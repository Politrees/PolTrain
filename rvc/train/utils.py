import argparse
import glob
import json
import logging
import os
import sys
from collections import OrderedDict

import soundfile as sf
import torch

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging


def replace_keys_in_dict(d, old_key_part, new_key_part):
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        new_key = key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        updated_dict[new_key] = replace_keys_in_dict(value, old_key_part, new_key_part) if isinstance(value, dict) else value
    return updated_dict


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_dict = replace_keys_in_dict(
        replace_keys_in_dict(checkpoint_dict, ".weight_v", ".parametrizations.weight.original1"),
        ".weight_g",
        ".parametrizations.weight.original0",
    )

    model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    new_state_dict = {k: checkpoint_dict["model"].get(k, v) for k, v in model_state_dict.items()}

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    if optimizer and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict.get("optimizer", {}))

    logger.info(f"Загружена контрольная точка '{checkpoint_path}' (эпоха {checkpoint_dict['iteration']})")
    return model, optimizer, checkpoint_dict.get("learning_rate", 0), checkpoint_dict["iteration"]


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    checkpoint_data = {
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }

    torch.save(
        replace_keys_in_dict(
            replace_keys_in_dict(checkpoint_data, ".parametrizations.weight.original1", ".weight_v"),
            ".parametrizations.weight.original0",
            ".weight_g",
        ),
        checkpoint_path,
    )

    logger.info(f"Сохранен чекпоинт '{checkpoint_path}' (эпоха {iteration})")


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    checkpoints = sorted(glob.glob(os.path.join(dir_path, regex)))
    return checkpoints[-1] if checkpoints else None


def load_wav_to_torch(full_path):
    data, sample_rate = sf.read(full_path, dtype="float32")
    return torch.FloatTensor(data), sample_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        return [line.strip().split(split) for line in f]


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_dir", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-te", "--total_epoch", type=int, required=True)
    parser.add_argument("-se", "--save_every_epoch", type=int, required=True)
    parser.add_argument("-bs", "--batch_size", type=int, required=True)
    parser.add_argument("-voc", "--vocoder", type=str, default="HiFi-GAN")
    parser.add_argument("-pg", "--pretrainG", type=str, default="")
    parser.add_argument("-pd", "--pretrainD", type=str, default="")
    parser.add_argument("-g", "--gpus", type=str, default="0")
    parser.add_argument("-sz", "--save_to_zip", type=str, default="False")

    args = parser.parse_args()
    experiment_dir = os.path.join(args.experiment_dir, args.model_name)

    config_save_path = os.path.join(experiment_dir, "data", "config.json")
    with open(config_save_path, "r") as f:
        config = json.load(f)

    hparams = HParams(**config)
    hparams.model_dir = experiment_dir
    hparams.model_name = args.model_name
    hparams.total_epoch = args.total_epoch
    hparams.save_every_epoch = args.save_every_epoch
    hparams.batch_size = args.batch_size
    hparams.vocoder = args.vocoder
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.gpus = args.gpus
    hparams.save_to_zip = args.save_to_zip
    hparams.data.training_files = f"{experiment_dir}/data/filelist.txt"
    return hparams


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = HParams(**v) if isinstance(v, dict) else v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)
