"""Modelo cinemático del dron (uniciclo planar, dt = 1 s)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from ..config import DroneConfig
from .geometry import unit, wrap_angle


class Action(IntEnum):
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    SHARP_LEFT = 3
    SHARP_RIGHT = 4
    HOVER = 5


ACTION_LABELS_ES = {
    Action.FORWARD: "Avanzar",
    Action.LEFT: "Girar izquierda",
    Action.RIGHT: "Girar derecha",
    Action.SHARP_LEFT: "Giro cerrado izq.",
    Action.SHARP_RIGHT: "Giro cerrado der.",
    Action.HOVER: "Mantener posición",
}
N_ACTIONS = len(Action)


@dataclass(frozen=True)
class DroneState:
    x: float
    y: float
    heading: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])


class Kinematics:
    """Traduce una acción discreta en un desplazamiento. Una sola responsabilidad:
    el viento y las colisiones se resuelven en el entorno, no aquí."""

    def __init__(self, cfg: DroneConfig):
        self.cfg = cfg
        soft, hard = np.radians(cfg.soft_turn_deg), np.radians(cfg.hard_turn_deg)
        v, vh = cfg.speed, cfg.speed * cfg.hard_turn_speed_factor
        self._table = {
            Action.FORWARD: (0.0, v),
            Action.LEFT: (soft, v),
            Action.RIGHT: (-soft, v),
            Action.SHARP_LEFT: (hard, vh),
            Action.SHARP_RIGHT: (-hard, vh),
            Action.HOVER: (0.0, 0.0),
        }

    def turn_and_speed(self, action: Action) -> tuple[float, float]:
        return self._table[Action(action)]

    def commanded_displacement(self, state: DroneState, action: Action) -> tuple[np.ndarray, float]:
        """Desplazamiento ordenado (sin viento) y nuevo rumbo."""
        turn, speed = self.turn_and_speed(action)
        heading = float(wrap_angle(state.heading + turn))
        return unit(heading) * speed, heading

    def apply(self, state: DroneState, action: Action, wind: np.ndarray | None = None) -> DroneState:
        disp, heading = self.commanded_displacement(state, action)
        if wind is not None:
            disp = disp + wind
        return DroneState(state.x + disp[0], state.y + disp[1], heading)
