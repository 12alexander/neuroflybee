"""La red de la abeja como `HomingProvider`: el navegador la usa igual que al ALV clásico."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..control.homing import HomingEstimate
from ..models.bee import BeeHomingNet
from ..sim.observation import Observation


class BeeNetHoming:
    name = "abeja"

    def __init__(self, model: BeeHomingNet, device: str | torch.device = "cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def estimate(self, obs: Observation) -> Optional[HomingEstimate]:
        cur = torch.from_numpy(np.asarray(obs.panorama, dtype=np.float32))[None].to(self.device)
        home = torch.from_numpy(np.asarray(obs.home_snapshot, dtype=np.float32))[None].to(self.device)
        angle, distance = BeeHomingNet.decode(self.model(cur, home))
        d = float(distance[0])
        return HomingEstimate(float(angle[0]), d, d < self.model.cfg.land_distance)
