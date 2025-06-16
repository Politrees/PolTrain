import logging
import os
import sys
import traceback
import warnings
from tqdm import tqdm
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())

from rvc.lib.audio import load_audio
from rvc.lib.rmvpe import RMVPE
from rvc.train.preprocess.preparing_files import generate_config, generate_filelist

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("fairseq").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

exp_dir = str(sys.argv[1])  # Директория с данными
f0_method = str(sys.argv[2])  # Метод извлечения F0
sample_rate = int(sys.argv[3])  # Частота дискретизации
include_mutes = int(sys.argv[4])  # Количество мьют файлов

device = "cuda" if torch.cuda.is_available() else "cpu"


class DataPreprocessor:
    def __init__(self):
        # Настройки для F0
        self.sample_rate = 16000
        self.hop_size = 160
        self.f0_bin = 256
        self.f0_min = 50.0
        self.f0_max = 1100.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        
        # Инициализация моделей
        self.model_rmvpe = RMVPE("assets/rmvpe/rmvpe.pt", "cuda")
        self.hubert_model = self._load_hubert_model()

    def _load_hubert_model(self):
        """Загрузка модели HuBERT"""
        model_path = "assets/hubert/hubert_base.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Error: HuBERT model not found at {model_path}, "
                "download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            )
        
        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path], suffix="")
        model = models[0].to(device).eval()
        return model
  
    def compute_f0(self, path, f0_method):
        """Вычисление F0"""
        audio = load_audio(path, self.sample_rate)
        if f0_method == "rmvpe":
            return self.model_rmvpe.infer_from_audio(audio, 0.03)
        elif f0_method == "rmvpe+":
            return self.model_rmvpe.infer_from_audio_modified(audio, 0.02)

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
        feats = self.read_wave(wav_path)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        
        with torch.no_grad():
            logits = self.hubert_model.extract_features(
                source=feats.to(device),
                padding_mask=padding_mask.to(device),
                output_layer=12
            )
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

        print(f"\nФрагментов, готовых к обработке - {len(files)}")

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

if __name__ == "__main__":
    try:
        preprocessor = DataPreprocessor()
        preprocessor.process_files()
        
        generate_config(exp_dir, sample_rate)
        generate_filelist(exp_dir, sample_rate, include_mutes)
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)
