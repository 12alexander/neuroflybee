"""Lo que un controlador puede ver. La posición real NO está aquí: sólo sensores."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PriorMap:
    """Mapa previo (sólo obstáculos conocidos), como el de un plan de vuelo."""
    occupancy: np.ndarray
    resolution: float
    width: float
    height: float


@dataclass
class Observation:
    t: int
    fly_patch: np.ndarray            # (S, S) ojo de mosca
    ranges: np.ndarray               # (n_rays,) distancias, relativas al rumbo
    panorama: np.ndarray             # (bins,) vista de la abeja (alineada al norte)
    heading: float                   # brújula
    odometry: np.ndarray             # desplazamiento medido en el último paso
    gps: Optional[np.ndarray]        # None si no hay señal
    base: np.ndarray
    waypoints: list[np.ndarray]
    home_snapshot: np.ndarray        # panorama memorizado al despegar
    prior_map: PriorMap
    extras: dict = field(default_factory=dict)
