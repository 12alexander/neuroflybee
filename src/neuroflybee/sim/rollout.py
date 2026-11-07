"""Ejecución de un episodio completo, registrando cada paso."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..control.base import Controller, Decision
from ..domain.drone import Action
from .env import NavigationEnv, StepInfo


@dataclass
class StepRecord:
    t: int
    position: np.ndarray
    heading: float
    decision: Decision
    info: StepInfo
    latency_ms: float


@dataclass
class Trajectory:
    controller: str
    family: str
    seed: int
    steps: list[StepRecord] = field(default_factory=list)
    collided: bool = False
    landed: bool = False

    @property
    def positions(self) -> np.ndarray:
        """Posiciones reales (sólo para evaluar y dibujar)."""
        return np.array([s.position for s in self.steps])

    @property
    def estimates(self) -> np.ndarray:
        """Posiciones que el dron CREE tener (NaN si el controlador no estima)."""
        return np.array([s.decision.estimate if s.decision.estimate is not None else [np.nan, np.nan]
                         for s in self.steps], dtype=float)

    @property
    def gps_lost_from(self) -> int | None:
        """Índice del primer paso sin GPS, o None."""
        for i, s in enumerate(self.steps):
            if not s.info.gps_available:
                return i
        return None


def rollout(env: NavigationEnv, controller: Controller, keep_debug: bool = True) -> Trajectory:
    obs = env.reset()
    controller.reset(obs)
    traj = Trajectory(controller.name, env.scenario.family, env.scenario.seed)
    traj.steps.append(StepRecord(0, env.state.position, env.state.heading,
                                 Decision(action=Action.HOVER, source="inicio", estimate=env.state.position),
                                 StepInfo(0, env.state, False, env.gps_available), 0.0))
    done = False
    while not done:
        t0 = time.perf_counter()
        decision = controller.act(obs)
        latency = (time.perf_counter() - t0) * 1000
        result = env.step(decision.action, decision.land)
        if not keep_debug:
            decision.debug = {}
        traj.steps.append(StepRecord(result.info.t, env.state.position, env.state.heading,
                                     decision, result.info, latency))
        obs, done = result.observation, result.done
    traj.collided, traj.landed = env.collided, env.landed
    return traj
