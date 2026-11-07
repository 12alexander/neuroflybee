"""Configuración de DAgger (sin PyTorch)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ..config import from_dict


@dataclass(frozen=True)
class DaggerConfig:
    model: str = "fly_cnn_gru"
    seeds: tuple[int, ...] = (0, 1, 2)
    iterations: int = 3
    episodes_per_iteration: int = 120
    first_new_seed: int = 400
    betas: tuple[float, ...] = (0.5, 0.25, 0.0)
    epochs_per_iteration: int = 5
    eval_families: tuple[str, ...] = ("nominal", "unknown_obstacles", "dynamic", "wind")
    eval_episodes_per_family: int = 10

    def beta(self, iteration: int) -> float:
        """β de la iteración (1-indexada); si faltan valores se repite el último."""
        return self.betas[min(iteration - 1, len(self.betas) - 1)]

    def new_seed_offset(self, iteration: int) -> int:
        return self.first_new_seed + (iteration - 1) * self.episodes_per_iteration


def load_dagger_config(path: Optional[str | Path]) -> DaggerConfig:
    if path is None:
        return DaggerConfig()
    with open(path, encoding="utf-8") as fh:
        return from_dict(DaggerConfig, yaml.safe_load(fh) or {})
