"""Estimación de posición: GPS cuando hay señal; integración de trayectoria
(odometría + brújula, como hacen los insectos) cuando no."""
from __future__ import annotations

import numpy as np

from ..sim.observation import Observation


class PositionEstimator:
    def __init__(self, gps_weight: float = 0.3):
        self.gps_weight = gps_weight
        self.position = np.zeros(2)
        self.using_gps = True
        self.steps_without_gps = 0

    def reset(self, obs: Observation) -> None:
        self.position = obs.base.astype(float).copy()
        self.using_gps = obs.gps is not None
        self.steps_without_gps = 0

    def update(self, obs: Observation) -> np.ndarray:
        predicted = self.position + obs.odometry          # integración de trayectoria
        if obs.gps is not None:
            self.position = self.gps_weight * obs.gps + (1 - self.gps_weight) * predicted
            self.using_gps, self.steps_without_gps = True, 0
        else:
            self.position = predicted
            self.using_gps = False
            self.steps_without_gps += 1
        return self.position.copy()

    def relocalize(self, position: np.ndarray, weight: float = 1.0) -> None:
        """Corrección externa (p. ej., la memoria visual de la abeja)."""
        self.position = weight * np.asarray(position) + (1 - weight) * self.position
