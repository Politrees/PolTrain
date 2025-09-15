import os
import sys
import logging
import warnings

# Конфигурация среды выполнения
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

from random import shuffle
import librosa
import numpy as np
import soundfile as sf
import torch
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)
from rvc.lib.audio import load_audio
from rvc.lib.rmvpe import RMVPE
from rvc.train.preprocess.slicer import Slicer


class DataPreprocessor:
    """Унифицированный препроцессор для подготовки аудиоданных в системе RVC.

    Реализует полный пайплайн предобработки, включающий сегментацию аудио,
    извлечение фундаментальной частоты (F0) и акустических признаков HuBERT.

    """

    def __init__(
        self,
        exp_dir: str,
        sample_rate: int = 40000,
        percentage: float = 3.0,
        normalize: bool = True,
        arch_fairseq: str = "Fairseq",
        f0_method: str = "rmvpe",
        include_mutes: int = 2,
    ):
        """Инициализация препроцессора данных.

        Args:
            exp_dir: Корневая директория эксперимента для сохранения результатов
            sample_rate: Целевая частота дискретизации в Гц (поддерживается: 32000, 40000, 48000)
            percentage: Максимальная длительность аудиосегмента в секундах
            normalize: Применение нормализации амплитуды к аудиосигналу
            arch_fairseq: Версия архитектуры Fairseq ("Fairseq" или "Fairseq2")
            f0_method: Алгоритм извлечения фундаментальной частоты ("rmvpe" или "rmvpe+")
            include_mutes: Количество сэмплов тишины на каждого диктора для аугментации
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.exp_dir = exp_dir
        self.percentage = percentage
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.arch_fairseq = arch_fairseq
        self.f0_method = f0_method
        self.include_mutes = include_mutes

        # Инициализация файловой структуры проекта
        self.data_dir = os.path.join(self.exp_dir, "data")
        self.gt_wavs_dir = os.path.join(self.data_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(self.data_dir, "sliced_audios_16k")
        self.f0_quant_dir = os.path.join(self.data_dir, "f0_quantized")
        self.f0_voiced_dir = os.path.join(self.data_dir, "f0_voiced")
        self.features_dir = os.path.join(self.data_dir, "features")
        for path in [self.gt_wavs_dir, self.wavs16k_dir, self.f0_quant_dir, self.f0_voiced_dir, self.features_dir]:
            os.makedirs(path, exist_ok=True)

        # Инициализация модулей сегментации аудио
        self.slicer = Slicer(
            sr=self.sample_rate,
            threshold=-42,  # Порог детекции тишины в дБ
            min_length=1500,  # Минимальная длина сегмента в мс
            min_interval=400,  # Минимальный интервал между сегментами в мс
            hop_size=15,  # Размер шага анализа в мс
            max_sil_kept=500,  # Максимальная длина сохраняемой тишины в мс
        )
        self.overlap = 0.3
        self.tail = self.percentage + self.overlap

        # Butterworth ФВЧ для удаления низкочастотных артефактов
        self.b_high, self.a_high = signal.butter(N=5, Wn=48, btype="high", fs=self.sample_rate)

        # Параметры анализа фундаментальной частоты
        self.f0_bin = 256  # Количество бинов квантования F0
        self.f0_min = 50.0  # Минимальная частота F0 в Гц
        self.f0_max = 1100.0  # Максимальная частота F0 в Гц

        # Mel-шкала для квантования F0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # Инициализация моделей
        self.model_rmvpe = RMVPE(os.path.join(now_dir, "rvc", "models", "predictors", "rmvpe.pt"), self.device)
        self.hubert_model = self._load_hubert_model()

    def _load_hubert_model(self):
        """Загрузка предобученной модели HuBERT для извлечения семантических признаков."""
        hubert_model_path = os.path.join(now_dir, "rvc", "models", "embedders", "contentvec_base.pt")
        if not os.path.exists(hubert_model_path):
            raise FileNotFoundError(f"Модель HuBERT не найдена: {hubert_model_path}")

        if self.arch_fairseq == "Fairseq":
            from fairseq.checkpoint_utils import load_model_ensemble_and_task
            from fairseq.data.dictionary import Dictionary

            torch.serialization.add_safe_globals([Dictionary])
            models, _, _ = load_model_ensemble_and_task([hubert_model_path], suffix="")
            return models[0].to(self.device).eval()

        elif self.arch_fairseq == "Fairseq2":
            from rvc.lib.fairseq import load_model

            model = load_model(hubert_model_path)
            return model.to(self.device).eval()
        else:
            raise ValueError(f"Неподдерживаемая архитектура Fairseq: {self.arch_fairseq}")

    def _norm_res_write(self, tmp_audio, idx0, idx1):
        """Нормализация и сохранение аудиосегмента с ресемплингом."""
        # Проверка на клиппинг и артефакты
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            return

        # Применение адаптивной нормализации с сохранением динамического диапазона
        if self.normalize:
            tmp_audio = (tmp_audio / tmp_max * (0.9 * 0.75)) + (1 - 0.75) * tmp_audio

        # Сохранение с оригинальной частотой дискретизации
        wavfile.write(os.path.join(self.gt_wavs_dir, f"{idx0}_{idx1}.wav"), self.sample_rate, tmp_audio.astype(np.float32))

        # Высококачественный ресемплинг до 16kHz для моделей извлечения признаков
        tmp_audio_16k = librosa.resample(tmp_audio, orig_sr=self.sample_rate, target_sr=16000, res_type="soxr_vhq")
        wavfile.write(os.path.join(self.wavs16k_dir, f"{idx0}_{idx1}.wav"), 16000, tmp_audio_16k.astype(np.float32))

    def _calculation_f0(self, path):
        """Вычисление контура фундаментальной частоты методом RMVPE."""
        audio, _ = sf.read(path)
        if self.f0_method == "rmvpe+":
            return self.model_rmvpe.infer_from_audio_modified(audio, 0.02)
        return self.model_rmvpe.infer_from_audio(audio, 0.03)

    def _quantization_f0(self, f0):
        """Квантование значений F0 в дискретные бины на mel-шкале.

        Преобразует непрерывные значения частоты в дискретные индексы
        для эффективного представления в нейросетевых моделях.

        """
        # Преобразование в mel-шкалу
        f0_mel = 1127 * np.log(1 + f0 / 700)

        # Линейное квантование в заданном диапазоне
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1

        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, "Квантованные значения F0 вне допустимого диапазона"
        return f0_coarse

    def _extract_semantic_features(self, wav_path):
        """Извлечение семантических признаков с использованием модели HuBERT."""
        wav, _ = sf.read(wav_path)
        feats = torch.from_numpy(wav).float().view(1, -1).to(self.device)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(self.device)

        with torch.no_grad():
            # Извлечение признаков из 12-го слоя HuBERT
            logits = self.hubert_model.extract_features(source=feats, padding_mask=padding_mask, output_layer=12)
            return logits[0].squeeze(0).float().cpu().numpy()

    def segmentation_audios(self, input_root: str):
        """Выполнение сегментации аудиофайлов с детекцией пауз.

        Разделяет длинные аудиозаписи на короткие сегменты фиксированной длины,
        автоматически определяя границы по паузам в речи.

        """
        print("\nИнициализация процесса сегментации аудиоданных...")

        if not os.path.exists(input_root):
            raise FileNotFoundError(f"Директория не существует: {input_root}")

        # Сканирование директории на наличие поддерживаемых форматов
        audio_files = [name for name in sorted(os.listdir(input_root)) if name.endswith((".wav", ".mp3", ".flac", ".ogg"))]
        if not audio_files:
            raise FileNotFoundError(f"Аудиофайлы не обнаружены в директории: {input_root}")

        total_segments = 0
        with tqdm(audio_files, desc="Процесс сегментации") as pbar_files:
            for idx, filename in enumerate(pbar_files):
                path = os.path.join(input_root, filename)

                try:
                    # Загрузка и предварительная фильтрация аудио
                    audio = load_audio(path, self.sample_rate)
                    audio = signal.lfilter(self.b_high, self.a_high, audio)

                    idx1 = 0
                    # Итерация по сегментам, определенным алгоритмом VAD
                    for audio_segment in self.slicer.slice(audio):
                        i = 0
                        while True:
                            start = int(self.sample_rate * (self.percentage - self.overlap) * i)
                            i += 1

                            # Проверка достаточной длины оставшейся части
                            if len(audio_segment[start:]) > self.tail * self.sample_rate:
                                tmp_audio = audio_segment[start : start + int(self.percentage * self.sample_rate)]
                                self._norm_res_write(tmp_audio, idx, idx1)
                                idx1 += 1
                                total_segments += 1
                            else:
                                # Обработка последнего короткого сегмента
                                tmp_audio = audio_segment[start:]
                                if len(tmp_audio) > 0:  # Проверка на пустой сегмент
                                    self._norm_res_write(tmp_audio, idx, idx1)
                                    idx1 += 1
                                    total_segments += 1
                                break

                            # Обновление статистики в реальном времени
                            pbar_files.set_postfix({"Обработано сегментов": total_segments}, refresh=True)

                except Exception as e:
                    tqdm.write(f"⚠ Ошибка обработки файла {filename}: {str(e)}")
                    raise

                pbar_files.set_postfix({"Обработано сегментов": total_segments}, refresh=True)
        
        if total_segments == 0:
            raise RuntimeError("Не удалось создать ни одного сегмента из входных данных")
            
        print(f"✓ Сегментация успешно завершена!")

    def extract_acoustic_features(self):
        """Извлечение акустических признаков из сегментированных аудиофайлов.

        Выполняет извлечение контуров F0 и семантических представлений HuBERT
        для всех подготовленных аудиосегментов.

        """
        # Сканирование подготовленных 16kHz файлов
        files = sorted([f for f in os.listdir(self.wavs16k_dir) if f.endswith(".wav") and "spec" not in f])
        if not files:
            self._raise_no_files_error()
            sys.exit(1)  # Принудительное завершение процесса

        print(f"\nОбнаружено сегментов для извлечения признаков: {len(files)}")

        # Фаза 1: Извлечение контуров фундаментальной частоты
        for file in tqdm(files, desc="Извлечение F0 (фундаментальной частоты)"):
            try:
                inp_path = os.path.join(self.wavs16k_dir, file)
                f0_quant_path = os.path.join(self.f0_quant_dir, file)
                f0_voiced_path = os.path.join(self.f0_voiced_dir, file)

                # Пропуск уже обработанных файлов
                if not (os.path.exists(f0_quant_path + ".npy") and os.path.exists(f0_voiced_path + ".npy")):
                    f0 = self._calculation_f0(inp_path)
                    np.save(f0_voiced_path, f0, allow_pickle=False)
                    coarse_f0 = self._quantization_f0(f0)
                    np.save(f0_quant_path, coarse_f0, allow_pickle=False)

            except Exception as e:
                raise RuntimeError(f"Ошибка извлечения F0 для {file}: {e}")

        # Фаза 2: Извлечение семантических признаков HuBERT
        for file in tqdm(files, desc="Извлечение семантических признаков HuBERT"):
            try:
                wav_path = os.path.join(self.wavs16k_dir, file)
                out_path = os.path.join(self.features_dir, file.replace('.wav', '.npy'))

                if not os.path.exists(out_path):
                    feats = self._extract_semantic_features(wav_path)

                    # Валидация извлеченных признаков
                    if np.isnan(feats).sum() > 0:
                        raise ValueError(f"Обнаружены NaN значения в признаках файла {file}")

                    np.save(out_path, feats, allow_pickle=False)

            except Exception as e:
                raise RuntimeError(f"Ошибка извлечения признаков HuBERT для {file}: {e}")

        print("✓ Извлечение акустических признаков успешно завершено!")
    
    def generate_filelist(self):
        """Генерация манифеста данных для обучения модели.

        Создает текстовый файл со списком путей к обработанным данным
        и соответствующими метками дикторов.

        """
        mute_base_path = os.path.join(now_dir, "logs", "mute")

        # Сбор и валидация обработанных файлов
        gt_wavs_files = set(name.split(".")[0] for name in os.listdir(self.gt_wavs_dir))
        feature_files = set(name.split(".")[0] for name in os.listdir(self.features_dir))
        f0_files = set(name.split(".")[0] for name in os.listdir(self.f0_quant_dir))
        f0nsf_files = set(name.split(".")[0] for name in os.listdir(self.f0_voiced_dir))

        # Пересечение множеств для обеспечения полноты данных
        names = gt_wavs_files & feature_files & f0_files & f0nsf_files
        if not names:
            raise RuntimeError("Нет полностью обработанных файлов для создания манифеста")

        sids = []
        options = []

        # Формирование записей манифеста
        for name in names:
            sid = name.split("_")[0]  # Извлечение ID диктора
            if sid not in sids:
                sids.append(sid)
            options.append(
                f"{os.path.join(self.gt_wavs_dir, name)}.wav|"
                f"{os.path.join(self.features_dir, name)}.npy|"
                f"{os.path.join(self.f0_quant_dir, name)}.wav.npy|"
                f"{os.path.join(self.f0_voiced_dir, name)}.wav.npy|{sid}"
            )

        # Добавление сэмплов тишины для улучшения робастности модели
        if self.include_mutes > 0:
            mute_audio = os.path.join(mute_base_path, "sliced_audios", f"mute{self.sample_rate}.wav")
            mute_feature = os.path.join(mute_base_path, "features", "mute.npy")
            mute_f0 = os.path.join(mute_base_path, "f0_quantized", "mute.wav.npy")
            mute_f0nsf = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")
            for sid in sids * self.include_mutes:
                options.append(f"{mute_audio}|{mute_feature}|{mute_f0}|{mute_f0nsf}|{sid}")

        # Рандомизация порядка для улучшения обучения
        shuffle(options)
        with open(os.path.join(self.data_dir, "filelist.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(options))

    def process_dataset(self, input_root: str):
        """Выполнение полного пайплайна предобработки датасета.

        Последовательно выполняет все этапы подготовки данных:
        сегментацию, извлечение признаков и генерацию манифеста.

        """
        try:
            # 1: Сегментация аудиоданных
            self.segmentation_audios(input_root)

            # 2: Извлечение акустических признаков
            self.extract_acoustic_features()

            # 3: Генерация манифеста для обучения
            self.generate_filelist()

        except Exception as e:
            raise RuntimeError(f"\n❌ Критическая ошибка в процессе обработки: {str(e)}")

    def _raise_no_files_error(self):
        """Генерация детализированного сообщения об ошибке при отсутствии данных."""
        error_message = (
            "ОШИБКА: Отсутствуют файлы для обработки.\n\n"
            "Возможные причины:\n"
            "• Аудиофайлы содержат только тишину или имеют слишком низкий уровень сигнала\n"
            "• Общая длительность аудио менее минимального порога (3 секунды)\n"
            "• Единичный файл превышает максимальную длительность (1 час)\n"
            "• Некорректный формат или повреждение исходных файлов\n\n"
            "Рекомендации по устранению:\n"
            "1. Проверьте уровень сигнала в исходных файлах (рекомендуется -20 dB RMS)\n"
            "2. Увеличьте объем обучающих данных\n"
            "3. Разделите длинные записи на фрагменты по 10-30 минут\n"
            "4. Убедитесь в корректности аудиоформатов (WAV, MP3, FLAC, OGG)\n"
        )
        raise FileNotFoundError(error_message)


def main():
    """Точка входа для запуска препроцессора из командной строки.

    Обрабатывает аргументы командной строки и инициирует процесс предобработки.

    """
    if len(sys.argv) < 6:
        print("Использование:")
        print("python preparing_data.py <exp_dir> <input_root> <percentage> <sample_rate> <normalize> [arch_fairseq] [f0_method] [include_mutes]")
        print("\nПараметры:")
        print("  exp_dir      - директория для сохранения результатов")
        print("  input_root   - директория с исходными аудиофайлами")
        print("  percentage   - максимальная длина сегмента (секунды)")
        print("  sample_rate  - частота дискретизации (32000/40000/48000)")
        print("  normalize    - применять нормализацию (True/False)")
        print("  arch_fairseq - архитектура Fairseq (Fairseq/Fairseq2)")
        print("  f0_method    - метод извлечения F0 (rmvpe/rmvpe+)")
        print("  include_mutes - количество сэмплов тишины")
        sys.exit(1)

    # Парсинг обязательных аргументов
    exp_dir = sys.argv[1]
    input_root = sys.argv[2]
    percentage = float(sys.argv[3])
    sample_rate = int(sys.argv[4])
    normalize = sys.argv[5].lower() in ["true", "1", "yes"]

    # Парсинг опциональных аргументов с значениями по умолчанию
    arch_fairseq = sys.argv[6] if len(sys.argv) > 6 else "Fairseq"
    f0_method = sys.argv[7] if len(sys.argv) > 7 else "rmvpe"
    include_mutes = int(sys.argv[8]) if len(sys.argv) > 8 else 2
    if include_mutes < 0 or include_mutes > 10:
        raise ValueError("include_mutes не может быть отрицательным или более 10")

    # Инициализация и запуск препроцессора
    try:
        preprocessor = DataPreprocessor(
            exp_dir=exp_dir,
            sample_rate=sample_rate,
            percentage=percentage,
            normalize=normalize,
            arch_fairseq=arch_fairseq,
            f0_method=f0_method,
            include_mutes=include_mutes
        )
        preprocessor.process_dataset(input_root)
    except Exception as e:
        raise RuntimeError(f"\n❌ Ошибка выполнения: {e}")


if __name__ == "__main__":
    main()
