import logging
import os
import sys
import traceback
import warnings
from random import shuffle

# Установка переменных окружения и отключение предупреждений
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())

from rvc.lib.audio import load_audio
from rvc.lib.rmvpe import RMVPE
# from rvc.lib.hpa_rmvpe import HPA_RMVPE

exp_dir = str(sys.argv[1])  # Директория с данными, подготовленными скриптом `preprocess.py`
arch_fairseq = str(sys.argv[2])  # Архитектура Fairseq / Fairseq, Fairseq2
f0_method = str(sys.argv[3])  # Метод извлечения F0 / rmvpe, rmvpe+, hpa-rmvpe
sample_rate = int(sys.argv[4])  # Частота дискретизации для генерации filelist.txt
include_mutes = int(sys.argv[5])  # Количество мьют файлов на одного спикера / По умолчанию = 2


class DataPreprocessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Настройки для F0
        self.sample_rate = 16000
        self.hop_size = 160
        self.f0_bin = 256
        self.f0_min = 50.0
        self.f0_max = 1100.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # Инициализация моделей
        self.model_rmvpe = RMVPE(os.path.join(os.getcwd(), "rvc", "models", "predictors", "rmvpe.pt"), "cuda")
        # self.model_hpa_rmvpe = HPA_RMVPE(os.path.join(os.getcwd(), "rvc", "models", "predictors", "hpa-rmvpe.pt"), "cuda", True)
        self.hubert_model = self._load_hubert_model(arch_fairseq)

    def _load_hubert_model(self, arch_fairseq):
        """Загрузка модели HuBERT"""
        hubert_model_path = os.path.join(os.getcwd(), "rvc", "models", "embedders", "contentvec_base.pt")
        if arch_fairseq == "Fairseq":
            from fairseq.checkpoint_utils import load_model_ensemble_and_task
            from fairseq.data.dictionary import Dictionary

            torch.serialization.add_safe_globals([Dictionary])
            models, _, _ = load_model_ensemble_and_task([hubert_model_path], suffix="")
            return models[0].to(self.device).eval()
        elif arch_fairseq == "Fairseq2":
            from rvc.lib.fairseq import load_model

            model = load_model(hubert_model_path)
            return model.to(self.device).eval()
        else:
            raise ValueError("Неизвестное значение для 'arch_fairseq'! Доступные варианты: 'Fairseq', 'Fairseq2'.")


    def compute_f0(self, path, f0_method):
        """Вычисление F0"""
        audio = load_audio(path, self.sample_rate)
        if f0_method == "rmvpe":
            return self.model_rmvpe.infer_from_audio(audio, 0.03)
        elif f0_method == "rmvpe+":
            return self.model_rmvpe.infer_from_audio_modified(audio, 0.02)
        # elif f0_method == "hpa-rmvpe":
        #     return self.model_hpa_rmvpe.infer_from_audio(audio, 0.03)

    def coarse_f0(self, f0):
        """Квантование F0"""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
        return f0_coarse

    def read_wave(self, wav_path):
        """Чтение аудиофайла"""
        wav, sr = sf.read(wav_path)
        assert sr == 16000
        feats = torch.from_numpy(wav).float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1
        return feats.view(1, -1)

    def extract_features(self, wav_path):
        """Извлечение признаков HuBERT"""
        feats = self.read_wave(wav_path).to(self.device)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(self.device)

        with torch.no_grad():
            logits = self.hubert_model.extract_features(source=feats, padding_mask=padding_mask, output_layer=12)
            return logits[0].squeeze(0).float().cpu().numpy()

    def process_files(self):
        """Основной метод обработки файлов"""
        # Подготовка путей
        inp_root = f"{exp_dir}/data/sliced_audios_16k"
        f0_quant_path = f"{exp_dir}/data/f0_quantized"
        f0_voiced_path = f"{exp_dir}/data/f0_voiced"
        features_path = f"{exp_dir}/data/features"

        os.makedirs(f0_quant_path, exist_ok=True)
        os.makedirs(f0_voiced_path, exist_ok=True)
        os.makedirs(features_path, exist_ok=True)

        # Сбор файлов для обработки
        files = sorted([f for f in os.listdir(inp_root) if f.endswith(".wav") and "spec" not in f])
        if not files:
            self._raise_no_files_error()

        print(f"\nДанных, готовых к обработке - {len(files)}")

        # Обработка файлов
        for file in tqdm(files, desc="Извлечение тона"):
            try:
                inp_path = f"{inp_root}/{file}"
                opt_path1 = f"{f0_quant_path}/{file}"
                opt_path2 = f"{f0_voiced_path}/{file}"

                if not (os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy")):
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(opt_path2, featur_pit, allow_pickle=False)
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(opt_path1, coarse_pit, allow_pickle=False)
            except:
                raise RuntimeError(f"Ошибка извлечения тона!\nФайл - {inp_path}\n{traceback.format_exc()}")

        for file in tqdm(files, desc="Извлечение признаков"):
            try:
                wav_path = f"{inp_root}/{file}"
                out_path = f"{features_path}/{file.replace('.wav', '.npy')}"

                if not os.path.exists(out_path):
                    feats = self.extract_features(wav_path)
                    if np.isnan(feats).sum() > 0:
                        raise TypeError(f"Файл {file} содержит некорректные значения (NaN).")
                    np.save(out_path, feats, allow_pickle=False)
            except:
                raise RuntimeError(f"Ошибка извлечения признаков!\nФайл - {wav_path}\n{traceback.format_exc()}")

        print("Обработка данных успешно завершена!")

    def _raise_no_files_error(self):
        error_message = (
            "ОШИБКА: Не найдено ни одного фрагмента для обработки.\n"
            "Возможные причины:\n"
            "1. Датасет не имеет звука.\n"
            "2. Датасет слишком тихий.\n"
            "3. Датасет слишком короткий (менее 3 секунд).\n"
            "4. Датасет слишком длинный (более 1 часа одним файлом).\n\n"
            "Попробуйте увеличить громкость или объем датасета. Если у вас один большой файл, можно разделить его на несколько более мелких."
        )
        raise FileNotFoundError(error_message)


def generate_filelist(model_path: str, sample_rate: int, include_mutes: int = 2):
    mute_base_path = os.path.join(os.getcwd(), "logs", "mute")

    f0_dir, f0nsf_dir = None, None
    gt_wavs_dir = os.path.join(model_path, "data", "sliced_audios")
    feature_dir = os.path.join(model_path, "data", "features")
    f0_dir = os.path.join(model_path, "data", "f0_quantized")
    f0nsf_dir = os.path.join(model_path, "data", "f0_voiced")

    gt_wavs_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir))
    feature_files = set(name.split(".")[0] for name in os.listdir(feature_dir))
    f0_files = set(name.split(".")[0] for name in os.listdir(f0_dir))
    f0nsf_files = set(name.split(".")[0] for name in os.listdir(f0nsf_dir))

    names = gt_wavs_files & feature_files & f0_files & f0nsf_files

    sids = []
    options = []
    for name in names:
        sid = name.split("_")[0]
        if sid not in sids:
            sids.append(sid)
        options.append(
            f"{os.path.join(gt_wavs_dir, name)}.wav|"
            f"{os.path.join(feature_dir, name)}.npy|"
            f"{os.path.join(f0_dir, name)}.wav.npy|"
            f"{os.path.join(f0nsf_dir, name)}.wav.npy|{sid}"
        )

    if include_mutes > 0:
        mute_audio_path = os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav")
        mute_feature_path = os.path.join(mute_base_path, "features", "mute.npy")
        mute_f0_path = os.path.join(mute_base_path, "f0_quantized", "mute.wav.npy")
        mute_f0nsf_path = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")

        # добавление (include_mutes) файлов для каждого sid
        for sid in sids * include_mutes:
            options.append(f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}")

    shuffle(options)

    with open(os.path.join(model_path, "data", "filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(options))


if __name__ == "__main__":
    try:
        preprocessor = DataPreprocessor()
        preprocessor.process_files()

        generate_filelist(exp_dir, sample_rate, include_mutes)
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)
