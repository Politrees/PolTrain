import os
import sys
import time
import traceback

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

sys.path.append(os.getcwd())

from rvc.lib.audio import load_audio
from rvc.train.preprocess.slicer import Slicer

# Парсинг аргументов командной строки
exp_dir = sys.argv[1]  # Директория для сохранения результатов
input_root = sys.argv[2]  # Директория с входными аудиофайлами
percentage = float(sys.argv[3])  # Максимальная длина сегмента в секундах / По умолчанию = 3.0 (от n до 3сек)
sample_rate = int(sys.argv[4])  # Частота дискретизации в которую преобразуются данные / 32000, 40000 и 48000
normalize = sys.argv[5] == "True"  # Флаг для включения/выключения нормализации

# Поддерживаемые аудио-расширения
AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".aifc",
    ".mp4",
    ".mkv",
    ".webm",
}


def is_audio_file(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in AUDIO_EXTENSIONS


class Postfix:
    # tqdm сам подставляет ", " перед постфиксом — этот класс обходит это,
    # чтобы строка выглядела как "4/4 [сегментов: 922]", а не "4/4, сегментов: 922"
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


class PreProcess:
    def __init__(self, sample_rate, exp_dir, percentage=3.0, normalize=True):
        # Директории для сохранения обработанных аудиофайлов
        self.gt_wavs_dir = os.path.join(exp_dir, "data", "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "data", "sliced_audios_16k")

        # Создаем директории, если они не существуют
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

        # Инициализация Slicer для нарезки аудио
        self.slicer = Slicer(
            sr=sample_rate,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sample_rate = sample_rate  # Частота дискретизации
        self.b_high, self.a_high = signal.butter(N=5, Wn=48, btype="high", fs=self.sample_rate)  # Фильтр высоких частот
        self.percentage = percentage  # Длина сегмента
        self.overlap = 0.3  # Перекрытие между сегментами
        self.tail = self.percentage + self.overlap  # Хвост для обработки
        self.normalize = normalize  # Флаг для включения/выключения нормализации

    def norm_write(self, tmp_audio, idx0, idx1):
        # Проверка на превышение максимального уровня сигнала
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            return 0  # Сегмент слишком громкий — пропускаем

        # Применение нормализации к аудио и сохранение в WAV
        if self.normalize:
            tmp_audio = (tmp_audio / tmp_max * (0.9 * 0.75)) + (1 - 0.75) * tmp_audio
        wavfile.write(f"{self.gt_wavs_dir}/{idx0}_{idx1}.wav", self.sample_rate, tmp_audio.astype(np.float32))

        # Ресемплирование аудио до 16 кГц и сохранение в WAV
        tmp_audio_16k = librosa.resample(tmp_audio, orig_sr=self.sample_rate, target_sr=16000, res_type="soxr_vhq")
        wavfile.write(f"{self.wavs16k_dir}/{idx0}_{idx1}.wav", 16000, tmp_audio_16k.astype(np.float32))
        return 1  # Сегмент записан

    def pipeline_inp_dir(self, input_root):
        try:
            # Собираем только аудиофайлы; всё остальное в папке датасета игнорируем
            names = sorted(
                name
                for name in os.listdir(input_root)
                if is_audio_file(os.path.join(input_root, name))
            )
            if not names:
                raise FileNotFoundError(
                    f"В папке '{input_root}' не найдено ни одного аудиофайла "
                    f"(поддерживаются: {', '.join(sorted(AUDIO_EXTENSIONS))})."
                )

            print(f"[1/3] - Запуск процесса сегментации аудиоданных...")

            total_segments = 0
            # Прогресс по файлам; счетчик сегментов тикает в постфиксе в реальном времени.
            # ncols фиксируем, чтобы при сломанном определении ширины терминала tqdm не игнорировал bar_format
            with tqdm(
                total=len(names),
                desc="Сегментация аудиоданных",
                bar_format="{desc}: {n}/{total}{postfix}",
                ncols=80,
            ) as pbar:
                last_paint = 0.0
                for idx, name in enumerate(names):
                    path = os.path.join(input_root, name)
                    try:
                        # Загрузка аудио
                        audio = load_audio(path, self.sample_rate)
                        # Применение фильтра высоких частот
                        audio = signal.lfilter(self.b_high, self.a_high, audio)

                        idx1 = 0
                        # Нарезка аудио на сегменты
                        for audio in self.slicer.slice(audio):
                            i = 0
                            while True:
                                # Вычисление начальной точки сегмента
                                start = int(self.sample_rate * (self.percentage - self.overlap) * i)
                                i += 1
                                # Проверка, остался ли хвост аудио
                                if len(audio[start:]) > self.tail * self.sample_rate:
                                    tmp_audio = audio[start : start + int(self.percentage * self.sample_rate)]
                                    total_segments += self.norm_write(tmp_audio, idx, idx1)
                                    idx1 += 1
                                    # Перерисовываем счетчик не чаще ~10 раз в секунду:
                                    # чаще — терминал не успевает и число в конце файла прыгает
                                    if time.monotonic() - last_paint >= 0.1:
                                        last_paint = time.monotonic()
                                        pbar.postfix = Postfix(f" [сегментов: {total_segments}]")
                                        pbar.refresh()
                                else:
                                    tmp_audio = audio[start:]
                                    if len(tmp_audio) > 0:  # Пустой хвост не записываем
                                        total_segments += self.norm_write(tmp_audio, idx, idx1)
                                        idx1 += 1
                                    break
                    except Exception:
                        raise RuntimeError(f"{path}\t-> {traceback.format_exc()}")
                    pbar.update(1)
                    pbar.postfix = Postfix(f" [сегментов: {total_segments}]")
                    pbar.refresh()

            print(f"✓ Сегментация успешно завершена!")
        except Exception:
            raise RuntimeError(f"Ошибка! {traceback.format_exc()}")


def preprocess_trainset(input_root, sample_rate, exp_dir, percentage, normalize):
    # Инициализация и запуск обработки
    pp = PreProcess(sample_rate, exp_dir, percentage, normalize)
    pp.pipeline_inp_dir(input_root)


if __name__ == "__main__":
    # Запуск препроцессинга
    preprocess_trainset(input_root, sample_rate, exp_dir, percentage, normalize)
