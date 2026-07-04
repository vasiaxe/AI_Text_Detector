import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml


def load_config(config_path: str) -> dict:
    '''Load a YAML configuration file'''
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def set_seed(seed: int = 42) -> None:
    '''Set random seeds for reproducibility'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    '''Return CUDA device if available, otherwise CPU'''
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dir(path: str | Path) -> Path:
    '''Create a directory if it does not exist'''
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    '''Save dictionary as JSON'''
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, sort_keys=True)