"""Modelo de viento por defecto: el del escenario, con ráfagas gaussianas (v1–v17).

Implementa `contracts.WindModel`. Reproduce exactamente el comportamiento anterior
(misma secuencia de números aleatorios), para que los resultados previos no cambien.
"""
from __future__ import annotations

import numpy as np


class ScenarioWind:
    def __init__(self, mean: np.ndarray, gust_ratio: float = 0.3):
        self.mean = np.asarray(mean, dtype=float)
        self.gust_ratio = gust_ratio

    def reset(self, rng: np.random.Generator) -> None:  # sin estado
        pass

    def velocity(self, t: int, position: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        speed = float(np.linalg.norm(self.mean))
        if speed == 0:
            return self.mean.copy()
        return self.mean + rng.normal(0, self.gust_ratio * speed, 2)
