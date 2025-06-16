import argparse
import glob
import json
import os
from collections import OrderedDict
from distutils.util import strtobool

import soundfile as sf
import torch
import torch_optimizer as torch_optim


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

    model_to_load = model.module if hasattr(model, "module") else model
    model_state_dict = model_to_load.state_dict()

    new_state_dict = {k: checkpoint_dict["model"].get(k, v) for k, v in model_state_dict.items()}
    model_to_load.load_state_dict(new_state_dict, strict=False)

    if optimizer and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict.get("optimizer", {}))

    print(f"Загружена контрольная точка '{checkpoint_path}' (эпоха {checkpoint_dict['iteration']})", flush=True)
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

    print(f"Сохранен чекпоинт '{checkpoint_path}' (эпоха {iteration})", flush=True)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    checkpoints = sorted(glob.glob(os.path.join(dir_path, regex)))
    return checkpoints[-1] if checkpoints else None


def optimizer_class(optimizer, params_g, params_d, lr, betas, eps):
    # torch.optim
    if optimizer == "Adam":
        optim_class = torch.optim.Adam
    elif optimizer == "AdamW":
        optim_class = torch.optim.AdamW # - нормик (дефолт)
    elif optimizer == "RAdam":
        optim_class = torch.optim.RAdam # - 50/50
    elif optimizer == "NAdam":
        optim_class = torch.optim.NAdam
    elif optimizer == "Adamax":
        optim_class = torch.optim.Adamax
    elif optimizer == "SparseAdam":
        optim_class = torch.optim.SparseAdam
    # torch_optimizer
    elif optimizer == "Lamb":
        optim_class = torch_optim.Lamb
    elif optimizer == "Yogi":
        optim_class = torch_optim.Yogi
    elif optimizer == "AdamP":
        optim_class = torch_optim.AdamP # - вроде норм
    elif optimizer == "SWATS":
        optim_class = torch_optim.SWATS
    elif optimizer == "AdaMod":
        optim_class = torch_optim.AdaMod
    elif optimizer == "QHAdam":
        optim_class = torch_optim.QHAdam
    elif optimizer == "Apollo":
        optim_class = torch_optim.Apollo
    elif optimizer == "DiffGrad":
        optim_class = torch_optim.DiffGrad # - 50/50
    elif optimizer == "NovoGrad":
        optim_class = torch_optim.NovoGrad
    elif optimizer == "AdaBound":
        optim_class = torch_optim.AdaBound
    elif optimizer == "AdaBelief":
        optim_class = torch_optim.AdaBelief # - какащке
    elif optimizer == "Adahessian":
        optim_class = torch_optim.Adahessian

    optim_g = optim_class(params=params_g, lr=lr, betas=betas, eps=eps)
    optim_d = optim_class(params=params_d, lr=lr, betas=betas, eps=eps)

    return optim_g, optim_d


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_dir", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-te", "--total_epoch", type=int, required=True)
    parser.add_argument("-se", "--save_every_epoch", type=int, required=True)
    parser.add_argument("-bs", "--batch_size", type=int, required=True)
    parser.add_argument("-voc", "--vocoder", type=str, default="HiFi-GAN")
    parser.add_argument("-opt", "--optimizer", type=str, default="AdamW")
    parser.add_argument("-pg", "--pretrainG", type=str, default="")
    parser.add_argument("-pd", "--pretrainD", type=str, default="")
    parser.add_argument("-g", "--gpus", type=str, default="0")
    parser.add_argument("-s", "--sex", type=float, default=0.0)
    parser.add_argument("-sz", "--save_to_zip", type=lambda x: bool(strtobool(x)), default=False)

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
    hparams.optimizer = args.optimizer
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.gpus = args.gpus
    hparams.sex = args.sex
    hparams.save_to_zip = args.save_to_zip
    hparams.data.training_files = f"{experiment_dir}/data/filelist.txt"
    return hparams


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
