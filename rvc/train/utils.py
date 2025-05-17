import argparse
import glob
import json
import logging
import os
import sys

import soundfile as sf
import torch


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    saved_state_dict = checkpoint_dict["model"]
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                logger.warning(f"Несоответствие формы для {k}: нужно {state_dict[k].shape}, получено {saved_state_dict[k].shape}")
                raise KeyError
        except:
            logger.info(f"Ключ {k} не найден в контрольной точке при использовании начальных весов")
            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]

    if optimizer is not None and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])

    logger.info(f"Успешно загружена контрольная точка '{checkpoint_path}' (эпоха {iteration})")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(f"Сохранение состояния модели в '{checkpoint_path}' (эпоха {iteration})")

    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    x = f_list[-1]
    logger.debug(x)
    return x


def summarize(writer, tracking, scalars={}, images={}):
    for k, v in scalars.items():
        writer.add_scalar(k, v, tracking, double_precision=True)
    for k, v in images.items():
        writer.add_image(k, v, tracking, dataformats="HWC")


def load_wav_to_torch(full_path):
    data, sample_rate = sf.read(full_path, dtype="float32")
    return torch.FloatTensor(data), sample_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        return [line.strip().split(split) for line in f]


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-se", "--save_every_epoch", type=int, required=True)
    parser.add_argument("-te", "--total_epoch", type=int, required=True)
    parser.add_argument("-pg", "--pretrainG", type=str, default="")
    parser.add_argument("-pd", "--pretrainD", type=str, default="")
    parser.add_argument("-g", "--gpus", type=str, default="0")
    parser.add_argument("-bs", "--batch_size", type=int, required=True)
    parser.add_argument("-e", "--experiment_dir", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-sr", "--sample_rate", type=str, required=True)
    parser.add_argument("-sz", "--save_to_zip", type=str, required=True)

    args = parser.parse_args()
    name = args.model_name
    experiment_dir = os.path.join(args.experiment_dir, args.model_name)

    config_save_path = os.path.join(experiment_dir, "config.json")
    with open(config_save_path, "r") as f:
        config = json.load(f)

    hparams = HParams(**config)
    hparams.model_dir = hparams.experiment_dir = experiment_dir
    hparams.save_every_epoch = args.save_every_epoch
    hparams.name = name
    hparams.total_epoch = args.total_epoch
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.gpus = args.gpus
    hparams.batch_size = args.batch_size
    hparams.sample_rate = args.sample_rate
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
