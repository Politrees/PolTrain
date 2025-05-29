import logging
import os
import sys
import traceback
import warnings

import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Отключаем мусорное логирование
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("fairseq").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Получаем путь к директории эксперимента
exp_dir = sys.argv[1]

# Настройка переменных окружения для работы с MPS (Metal Performance Shaders)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Определяем устройство для вычислений (CUDA, MPS или CPU)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

f = open(f"{exp_dir}/logfile.log", "a+")


# Функция для вывода и записи в лог
def printt(strr):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


# Путь к модели HuBERT
model_path = "assets/hubert/hubert_base.pt"

# Пути к директориям с аудиофайлами и для сохранения признаков
wavPath = f"{exp_dir}/data/sliced_audios_16k"
outPath = f"{exp_dir}/data/features"
os.makedirs(outPath, exist_ok=True)


# Функция для чтения аудиофайла и преобразования его в тензор
def readwave(wav_path):
    wav, sr = sf.read(wav_path)
    assert sr == 16000  # Проверяем, что частота дискретизации равна 16 кГц
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # Если аудио стерео, усредняем до моно
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()  # Проверяем, что тензор одномерный
    feats = feats.view(1, -1)  # Добавляем размерность батча
    return feats


# Проверяем, существует ли модель
if os.access(model_path, os.F_OK) == False:
    raise FileNotFoundError(
        f"Error: Extracting is shut down because {model_path} does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
    )

# Загружаем модель HuBERT
models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path], suffix="")
model = models[0]
model = model.to(device)
model.eval()

# Получаем список файлов для обработки
todo = sorted(list(os.listdir(wavPath)))[0::1]

# Проверяем, есть ли файлы для обработки
if len(todo) == 0:
    error_message = (
        "ОШИБКА: Не найдено ни одного признака для обработки.\n"
        "Возможные причины:\n"
        "1. Датасет не имеет звука.\n"
        "2. Датасет слишком тихий.\n"
        "3. Датасет слишком короткий (менее 3 секунд).\n"
        "4. Датасет слишком длинный (более 1 часа одним файлом).\n\n"
        "Попробуйте увеличить громкость или объем датасета. Если у вас один большой файл, можно разделить его на несколько более мелких."
    )
    raise FileNotFoundError(error_message)
else:
    try:
        printt(f"Фрагментов готовых к обработке - {len(todo)}")

        # Обрабатываем каждый файл
        for file in tqdm(todo, desc="Извлечение признаков"):
            try:
                if file.endswith(".wav"):
                    wav_path = f"{wavPath}/{file}"
                    out_path = f"{outPath}/{file.replace('wav', 'npy')}"

                    # Пропускаем файл, если признаки уже извлечены
                    if os.path.exists(out_path):
                        continue

                    # Читаем аудиофайл и преобразуем в тензор
                    feats = readwave(wav_path)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)

                    # Подготавливаем входные данные для модели
                    inputs = {
                        "source": feats.to(device),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 12,
                    }

                    # Извлекаем признаки с помощью модели
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats = logits[0]

                    # Сохраняем признаки в файл
                    feats = feats.squeeze(0).float().cpu().numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(out_path, feats, allow_pickle=False)
                    else:
                        raise TypeError(f"Ошибка: Файл {file} содержит некорректные значения (NaN).")
            except:
                raise RuntimeError(traceback.format_exc())

        printt("Все признаки извлечены!")
    except Exception as e:
        raise RuntimeError(f"Ошибка! {e}")
