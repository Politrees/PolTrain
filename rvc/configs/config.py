import json
import os

import torch

version_config_paths = [
    os.path.join("48000.json"),
    os.path.join("40000.json"),
    os.path.join("32000.json"),
]


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(int(self.device.split(":")[-1])) if self.device.startswith("cuda") else None
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def load_config_json(self):
        configs = {}
        for config_file in version_config_paths:
            config_path = os.path.join("rvc", "configs", config_file)
            with open(config_path, "r") as f:
                configs[config_file] = json.load(f)
        return configs

    def device_config(self):
        if self.device.startswith("cuda"):
            self.set_cuda_config()
        else:
            self.device = "cpu"

        # Configuration for 6GB GPU memory
        x_pad, x_query, x_center, x_max = (1, 6, 38, 41)
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            # Configuration for 5GB GPU memory
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (1024**3)
