"""Controlador compuesto: estimador + navegador (abeja) + control local (mosca/clásico).

Todos los controladores del benchmark se arman combinando piezas, sin herencia
profunda (composición sobre herencia)."""
from __future__ import annotations

from ..sim.observation import Observation
from .base import Decision
from .estimation import PositionEstimator
from .navigation import MissionNavigator
from .steering import Steering


class NavigationController:
    def __init__(self, name: str, navigator: MissionNavigator, steering: Steering,
                 estimator: PositionEstimator | None = None):
        self.name = name
        self.navigator = navigator
        self.steering = steering
        self.estimator = estimator or PositionEstimator()

    def reset(self, obs: Observation) -> None:
        self.estimator.reset(obs)
        self.navigator.reset(obs)
        self.steering.reset(obs)

    def act(self, obs: Observation) -> Decision:
        estimate = self.estimator.update(obs)
        target = self.navigator.update(obs, estimate)
        action, debug = self.steering.steer(obs, target.goal, estimate)
        debug.update({"gps": self.estimator.using_gps, "homing_active": target.homing_active})
        if target.homing is not None:
            debug["homing_dir"] = target.homing.direction
            debug["homing_dist"] = target.homing.distance
        probs = debug.pop("probs", None)
        source = "abeja" if target.homing_active else self.steering.source
        return Decision(action=action, land=target.land, phase=target.phase, source=source,
                        probs=probs, goal=target.goal, estimate=estimate, debug=debug)
