"""Seguimiento de la misión con la posición REAL (sólo para evaluar; los
controladores mantienen su propia creencia, que puede estar equivocada)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Phase(str, Enum):
    INSPECT = "INSPECT"
    RETURN = "RETURN"
    LANDED = "LANDED"


@dataclass
class MissionTracker:
    waypoints: list[np.ndarray]
    base: np.ndarray
    capture_radius: float = 1.5
    home_radius: float = 2.0
    visited: list[bool] = field(init=False)

    def __post_init__(self) -> None:
        self.visited = [False] * len(self.waypoints)

    def update(self, true_position: np.ndarray) -> list[int]:
        """Marca los puntos alcanzados; devuelve los índices recién visitados."""
        newly = []
        for i, wp in enumerate(self.waypoints):
            if not self.visited[i] and np.linalg.norm(true_position - wp) <= self.capture_radius:
                self.visited[i] = True
                newly.append(i)
        return newly

    @property
    def coverage(self) -> float:
        return sum(self.visited) / max(len(self.waypoints), 1)

    def home_error(self, true_position: np.ndarray) -> float:
        return float(np.linalg.norm(true_position - self.base))

    def returned_home(self, true_position: np.ndarray) -> bool:
        return self.home_error(true_position) <= self.home_radius

    def success(self, true_position: np.ndarray, collided: bool, landed: bool) -> bool:
        """Misión exitosa: todos los puntos inspeccionados, sin choques y aterrizaje
        a menos de `home_radius` de la base."""
        return self.coverage == 1.0 and not collided and landed and self.returned_home(true_position)
