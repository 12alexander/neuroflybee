"""Control centralizado de semillas para que cada experimento sea reproducible."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy y (si está instalado) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # el núcleo funciona sin PyTorch
        pass


def make_rng(seed: int) -> np.random.Generator:
    """Generador independiente: cada componente recibe el suyo y no comparte estado global."""
    return np.random.default_rng(seed)
