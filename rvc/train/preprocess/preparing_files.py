import os
import json
import shutil
from random import shuffle


def generate_config(model_path: str, sample_rate: int):
    config_path = os.path.join("rvc", "configs", f"{sample_rate}.json")
    config_save_path = os.path.join(model_path, "data", "config.json")
    if not os.path.exists(config_save_path):
        shutil.copyfile(config_path, config_save_path)


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

    with open(os.path.join(model_path, "data", "filelist.txt"), "w") as f:
        f.write("\n".join(options))
