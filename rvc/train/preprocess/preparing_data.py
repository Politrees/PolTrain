import logging
import multiprocessing
import os
import sys
import traceback
import warnings
from random import shuffle

# Настройка окружения
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

sys.path.append(os.getcwd())
from rvc.lib.audio import load_audio
from rvc.lib.rmvpe import RMVPE
from rvc.train.preprocess.slicer import Slicer


class DataPreprocessor:
    """Унифицированный класс для полной предобработки данных RVC"""

    def __init__(
        self,
        exp_dir: str,
        sample_rate: int = 40000,
        percentage: float = 3.0,
        normalize: bool = True,
        arch_fairseq: str = "Fairseq",
        f0_method: str = "rmvpe",
        include_mutes: int = 2,
        num_processes: int = None
    ):
        """
        Инициализация препроцессора

        Args:
            exp_dir: Директория эксперимента
            sample_rate: Частота дискретизации (32000, 40000, 48000)
            percentage: Максимальная длина сегмента в секундах
            normalize: Флаг нормализации
            arch_fairseq: Архитектура Fairseq ("Fairseq" или "Fairseq2")
            f0_method: Метод извлечения F0 ("rmvpe" или "rmvpe+")
            include_mutes: Количество мьют файлов на спикера
            num_processes: Количество процессов для параллельной обработки
        """
        self.exp_dir = exp_dir
        self.sample_rate = sample_rate
        self.percentage = percentage
        self.normalize = normalize
        self.arch_fairseq = arch_fairseq
        self.f0_method = f0_method
        self.include_mutes = include_mutes
        self.num_processes = num_processes or max(1, os.cpu_count() - 1)

        # Создание структуры директорий
        self._setup_directories()

        # Инициализация компонентов для нарезки
        self._init_slicing_components()

        # Инициализация компонентов для извлечения признаков
        self._init_feature_components()

    def _setup_directories(self):
        """Создание необходимых директорий"""
        self.data_dir = os.path.join(self.exp_dir, "data")
        self.gt_wavs_dir = os.path.join(self.data_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(self.data_dir, "sliced_audios_16k")
        self.f0_quant_dir = os.path.join(self.data_dir, "f0_quantized")
        self.f0_voiced_dir = os.path.join(self.data_dir, "f0_voiced")
        self.features_dir = os.path.join(self.data_dir, "features")

        for dir_path in [self.gt_wavs_dir, self.wavs16k_dir, self.f0_quant_dir, self.f0_voiced_dir, self.features_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def _init_slicing_components(self):
        """Инициализация компонентов для нарезки аудио"""
        self.slicer = Slicer(
            sr=self.sample_rate,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )

        # Фильтр высоких частот
        self.b_high, self.a_high = signal.butter(N=5, Wn=48, btype="high", fs=self.sample_rate)

        self.overlap = 0.3
        self.tail = self.percentage + self.overlap

    def _init_feature_components(self):
        """Инициализация компонентов для извлечения признаков"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Настройки для F0
        self.f0_sample_rate = 16000
        self.hop_size = 160
        self.f0_bin = 256
        self.f0_min = 50.0
        self.f0_max = 1100.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # Ленивая инициализация моделей (будут загружены при первом использовании)
        self.model_rmvpe = RMVPE(os.path.join(os.getcwd(), "rvc", "models", "predictors", "rmvpe.pt"), self.device)
        self.hubert_model = self._load_hubert_model()

    def _load_hubert_model(self):
        """Загрузка модели HuBERT"""
        hubert_model_path = os.path.join(os.getcwd(), "rvc", "models", "embedders", "contentvec_base.pt")

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
            raise ValueError(f"Неизвестная архитектура: {self.arch_fairseq}")

    def _norm_write(self, tmp_audio, idx0, idx1):
        """Нормализация и сохранение аудио сегмента"""
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            return

        if self.normalize:
            tmp_audio = (tmp_audio / tmp_max * (0.9 * 0.75)) + (1 - 0.75) * tmp_audio

        # Сохранение с исходной частотой
        wavfile.write(f"{self.gt_wavs_dir}/{idx0}_{idx1}.wav", self.sample_rate, tmp_audio.astype(np.float32))

        # Ресемплирование и сохранение в 16kHz
        tmp_audio_16k = librosa.resample(tmp_audio, orig_sr=self.sample_rate, target_sr=16000, res_type="soxr_vhq")
        wavfile.write(f"{self.wavs16k_dir}/{idx0}_{idx1}.wav", 16000, tmp_audio_16k.astype(np.float32))

    def slice_audios(self, input_root: str):
        """
        Этап 1: сегментация аудиофайлов
        """
        print("\nПодготовка данных к сегментации...")
        
        try:
            # Сбор информации о файлах
            audio_files = [
                name for name in sorted(os.listdir(input_root))
                if name.endswith((".wav", ".mp3", ".flac", ".ogg"))
            ]

            if not audio_files:
                raise FileNotFoundError(f"Не найдено аудиофайлов в {input_root}")

            total_segments = 0
            with tqdm(audio_files, desc="Сегментация файлов") as pbar_files:
                for idx, filename in enumerate(pbar_files):
                    path = os.path.join(input_root, filename)
                    
                    try:
                        audio = load_audio(path, self.sample_rate)
                        audio = signal.lfilter(self.b_high, self.a_high, audio)

                        idx1 = 0
                        for audio_segment in self.slicer.slice(audio):
                            i = 0
                            while True:
                                start = int(self.sample_rate * (self.percentage - self.overlap) * i)
                                i += 1

                                if len(audio_segment[start:]) > self.tail * self.sample_rate:
                                    tmp_audio = audio_segment[start : start + int(self.percentage * self.sample_rate)]
                                    self._norm_write(tmp_audio, idx, idx1)
                                    idx1 += 1
                                    total_segments += 1
                                else:
                                    tmp_audio = audio_segment[start:]
                                    self._norm_write(tmp_audio, idx, idx1)
                                    idx1 += 1
                                    total_segments += 1
                                    break
                                
                                # Обновляем postfix в реальном времени
                                pbar_files.set_postfix({"Сегментов": total_segments}, refresh=True)

                    except Exception as e:
                        tqdm.write(f"⚠ Ошибка: {filename} - {str(e)}")
                        continue
                    
                    # Обновляем после обработки каждого файла
                    pbar_files.set_postfix({"Сегментов": total_segments}, refresh=True)

            print(f"✓ Сегментация завершена!")

        except Exception as e:
            raise RuntimeError(f"Ошибка при сегментации: {str(e)}")

    def _compute_f0(self, path):
        """Вычисление F0"""
        audio = load_audio(path, self.f0_sample_rate)

        if self.f0_method == "rmvpe+":
            return self.model_rmvpe.infer_from_audio_modified(audio, 0.02)
        return self.model_rmvpe.infer_from_audio(audio, 0.03)

    def _coarse_f0(self, f0):
        """Квантование F0"""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1
        return f0_coarse

    def _extract_hubert_features(self, wav_path):
        """Извлечение признаков HuBERT"""
        wav, sr = sf.read(wav_path)
        assert sr == 16000

        feats = torch.from_numpy(wav).float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1

        feats = feats.view(1, -1).to(self.device)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(self.device)

        with torch.no_grad():
            logits = self.hubert_model.extract_features(source=feats, padding_mask=padding_mask, output_layer=12)
            return logits[0].squeeze(0).float().cpu().numpy()

    def _generate_filelist(self):
        """Генерация filelist.txt"""
        mute_base_path = os.path.join(os.getcwd(), "logs", "mute")

        # Сбор файлов
        gt_wavs_files = set(name.split(".")[0] for name in os.listdir(self.gt_wavs_dir))
        feature_files = set(name.split(".")[0] for name in os.listdir(self.features_dir))
        f0_files = set(name.split(".")[0] for name in os.listdir(self.f0_quant_dir))
        f0nsf_files = set(name.split(".")[0] for name in os.listdir(self.f0_voiced_dir))

        names = gt_wavs_files & feature_files & f0_files & f0nsf_files

        sids = []
        options = []

        for name in names:
            sid = name.split("_")[0]
            if sid not in sids:
                sids.append(sid)

            options.append(
                f"{os.path.join(self.gt_wavs_dir, name)}.wav|"
                f"{os.path.join(self.features_dir, name)}.npy|"
                f"{os.path.join(self.f0_quant_dir, name)}.wav.npy|"
                f"{os.path.join(self.f0_voiced_dir, name)}.wav.npy|{sid}"
            )

        # Добавление mute файлов
        if self.include_mutes > 0:
            mute_audio = os.path.join(mute_base_path, "sliced_audios", f"mute{self.sample_rate}.wav")
            mute_feature = os.path.join(mute_base_path, "features", "mute.npy")
            mute_f0 = os.path.join(mute_base_path, "f0_quantized", "mute.wav.npy")
            mute_f0nsf = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")

            for sid in sids * self.include_mutes:
                options.append(f"{mute_audio}|{mute_feature}|{mute_f0}|{mute_f0nsf}|{sid}")

        shuffle(options)

        filelist_path = os.path.join(self.data_dir, "filelist.txt")
        with open(filelist_path, "w", encoding="utf-8") as f:
            f.write("\n".join(options))

        return len(options)

    def extract_features(self):
        """
        Этап 2: Извлечение F0 и признаков HuBERT
        """
        # Сбор файлов для обработки
        files = sorted([f for f in os.listdir(self.wavs16k_dir) if f.endswith(".wav") and "spec" not in f])

        if not files:
            self._raise_no_files_error()

        print(f"\nСегментов, готовых к обработке - {len(files)}")

        # Извлечение F0
        for file in tqdm(files, desc="Извлечение F0"):
            try:
                inp_path = os.path.join(self.wavs16k_dir, file)
                f0_quant_path = os.path.join(self.f0_quant_dir, file)
                f0_voiced_path = os.path.join(self.f0_voiced_dir, file)

                if not (os.path.exists(f0_quant_path + ".npy") and os.path.exists(f0_voiced_path + ".npy")):
                    f0 = self._compute_f0(inp_path)
                    np.save(f0_voiced_path, f0, allow_pickle=False)
                    coarse_f0 = self._coarse_f0(f0)
                    np.save(f0_quant_path, coarse_f0, allow_pickle=False)

            except Exception:
                raise RuntimeError(f"Ошибка извлечения F0!\nФайл: {inp_path}\n{traceback.format_exc()}")

        # Извлечение признаков HuBERT
        for file in tqdm(files, desc="Извлечение признаков HuBERT"):
            try:
                wav_path = os.path.join(self.wavs16k_dir, file)
                out_path = os.path.join(self.features_dir, file.replace('.wav', '.npy'))

                if not os.path.exists(out_path):
                    feats = self._extract_hubert_features(wav_path)

                    if np.isnan(feats).sum() > 0:
                        raise ValueError(f"Файл {file} содержит NaN значения")

                    np.save(out_path, feats, allow_pickle=False)

            except Exception:
                raise RuntimeError(f"Ошибка извлечения признаков HuBERT!\nФайл: {wav_path}\n{traceback.format_exc()}")

        print("✓ Обработка данных успешно завершена!")

    def _raise_no_files_error(self):
        """Вывод информативной ошибки при отсутствии файлов"""
        error_message = (
            "ОШИБКА: Не найдено файлов для обработки.\n"
            "Возможные причины:\n"
            "1. Датасет не содержит звука или слишком тихий\n"
            "2. Датасет слишком короткий (менее 3 сек)\n"
            "3. Датасет слишком длинный (более 1 часа одним файлом)\n\n"
            "Рекомендации:\n"
            "- Увеличьте громкость аудио или объем данных\n"
            "- Разделите длинные файлы на части\n"
        )
        raise FileNotFoundError(error_message)

    def process_dataset(self, input_root: str):
        """
        Полный пайплайн обработки датасета

        Args:
            input_root: Директория с исходными аудиофайлами

        Returns:
            dict: Статистика обработки
        """
        stats = {}

        try:
            # Этап 1: Нарезка
            self.slice_audios(input_root)
            stats['sliced_files'] = len(os.listdir(self.gt_wavs_dir))

            # Этап 2: Извлечение признаков
            self.extract_features()
            stats['features_extracted'] = len(os.listdir(self.features_dir))

            # Этап 3: Генерация filelist
            stats['filelist_entries'] = self._generate_filelist()
            return stats

        except Exception as e:
            print(f"\n❌ Критическая ошибка: {str(e)}")
            print(traceback.format_exc())
            raise


def main():
    """Основная функция для запуска из командной строки"""
    if len(sys.argv) < 6:
        print("Использование:")
        print("python preparing_data.py <exp_dir> <input_root> <percentage> <sample_rate> <normalize> [arch_fairseq] [f0_method] [include_mutes]")
        sys.exit(1)

    # Парсинг аргументов
    exp_dir = sys.argv[1]
    input_root = sys.argv[2]
    percentage = float(sys.argv[3])
    sample_rate = int(sys.argv[4])
    normalize = sys.argv[5] == "True"

    # Опциональные аргументы
    arch_fairseq = sys.argv[6] if len(sys.argv) > 6 else "Fairseq"
    f0_method = sys.argv[7] if len(sys.argv) > 7 else "rmvpe"
    include_mutes = int(sys.argv[8]) if len(sys.argv) > 8 else 2

    # Создание препроцессора и запуск
    preprocessor = DataPreprocessor(
        exp_dir=exp_dir,
        sample_rate=sample_rate,
        percentage=percentage,
        normalize=normalize,
        arch_fairseq=arch_fairseq,
        f0_method=f0_method,
        include_mutes=include_mutes
    )

    try:
        preprocessor.process_dataset(input_root)
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
