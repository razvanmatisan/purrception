import random
from pathlib import Path
from time import localtime, strftime

import numpy as np
import torch
import yaml


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass

    return "cpu"


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_exp_name(model_type: str, backbone_type: str):
    timestamp = strftime("%Y-%m-%d_%H:%M:%S", localtime())
    exp_name = f"{model_type}-{backbone_type}-{timestamp}"

    return exp_name


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} does not exist")

    with path.open("r") as f:
        config = yaml.safe_load(f)

    return config
