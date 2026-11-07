"""Hiperparámetros de entrenamiento (sin PyTorch, para poder validarlos en cualquier equipo)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ..config import from_dict


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4                 # early stopping sobre la métrica de validación
    grad_clip: float = 1.0
    class_weight_power: float = 0.5   # 0 = sin pesos; 1 = inverso de la frecuencia
    max_class_weight: float = 10.0
    num_workers: int = 2
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "auto"              # auto | cpu | cuda


def load_train_config(path: Optional[str | Path]) -> TrainConfig:
    if path is None:
        return TrainConfig()
    with open(path, encoding="utf-8") as fh:
        return from_dict(TrainConfig, yaml.safe_load(fh) or {})
