"""Interfaz común de controladores (inversión de dependencias: el entorno, el
benchmark y el dashboard dependen de este protocolo, no de implementaciones)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ..domain.drone import Action
from ..domain.mission import Phase
from ..sim.observation import Observation


@dataclass
class Decision:
    action: Action
    land: bool = False
    phase: Phase = Phase.INSPECT
    source: str = ""                          # módulo que decidió: abeja, mosca, supervisor…
    probs: Optional[np.ndarray] = None        # distribución sobre acciones (si es neuronal)
    goal: Optional[np.ndarray] = None         # objetivo actual según la creencia del dron
    estimate: Optional[np.ndarray] = None     # posición estimada
    debug: dict = field(default_factory=dict)


@runtime_checkable
class Controller(Protocol):
    name: str

    def reset(self, obs: Observation) -> None: ...

    def act(self, obs: Observation) -> Decision: ...
