"""Navegador de misión (módulo "abeja" de alto nivel): recuerda la ruta, decide
la fase (inspeccionar / regresar / aterrizar) y el objetivo actual."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..domain.mission import Phase
from ..sim.observation import Observation
from .homing import HomingEstimate, HomingProvider


@dataclass
class NavigationTarget:
    goal: np.ndarray
    phase: Phase
    land: bool
    homing: Optional[HomingEstimate] = None
    homing_active: bool = False


class MissionNavigator:
    def __init__(self, homing: Optional[HomingProvider] = None, capture_radius: float = 0.7,
                 land_radius: float = 0.6, catchment_radius: float = 10.0,
                 homing_only_without_gps: bool = True, max_homing_steps: int = 80):
        self.homing = homing
        self.capture_radius = capture_radius
        self.land_radius = land_radius
        self.catchment_radius = catchment_radius
        self.homing_only_without_gps = homing_only_without_gps
        self.max_homing_steps = max_homing_steps

    def reset(self, obs: Observation) -> None:
        self.waypoints = [w.copy() for w in obs.waypoints]
        self.base = obs.base.copy()
        self.index = 0
        self.phase = Phase.INSPECT if self.waypoints else Phase.RETURN
        self.homing_steps = 0

    def update(self, obs: Observation, estimate: np.ndarray) -> NavigationTarget:
        if self.phase == Phase.INSPECT:
            while self.index < len(self.waypoints) and \
                    np.linalg.norm(estimate - self.waypoints[self.index]) <= self.capture_radius:
                self.index += 1
            if self.index >= len(self.waypoints):
                self.phase = Phase.RETURN
            else:
                return NavigationTarget(self.waypoints[self.index], self.phase, False)

        # Fase de retorno
        dist_pi = float(np.linalg.norm(estimate - self.base))
        homing_allowed = self.homing is not None and (obs.gps is None or not self.homing_only_without_gps)
        if homing_allowed and dist_pi <= self.catchment_radius:
            h = self.homing.estimate(obs)  # type: ignore[union-attr]
            if h is not None:
                self.homing_steps += 1
                step = min(h.distance if h.distance is not None else 2.0, 3.0)
                goal = estimate + step * np.array([np.cos(h.direction), np.sin(h.direction)])
                land = h.at_home or self.homing_steps >= self.max_homing_steps
                if land:
                    self.phase = Phase.LANDED
                return NavigationTarget(goal, self.phase, land, h, True)
        land = dist_pi <= self.land_radius
        if land:
            self.phase = Phase.LANDED
        return NavigationTarget(self.base, self.phase, land)
