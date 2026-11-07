"""Control local neuronal (módulo mosca) detrás de la interfaz `Steering`."""
from __future__ import annotations

import numpy as np

from ..domain.drone import Action
from ..inference.fly_runner import FlyPolicyRunner
from ..sim.observation import Observation


class FlySteering:
    """La red de la mosca decide la maniobra a partir del ojo y del objetivo relativo."""
    name, source = "mosca", "mosca"

    def __init__(self, runner: FlyPolicyRunner):
        self.runner = runner

    def reset(self, obs: Observation) -> None:
        self.runner.reset()

    def steer(self, obs: Observation, goal: np.ndarray, estimate: np.ndarray) -> tuple[Action, dict]:
        action = self.runner.act_towards(obs, goal, estimate)
        return action, {"probs": self.runner.last_probs.copy()}
