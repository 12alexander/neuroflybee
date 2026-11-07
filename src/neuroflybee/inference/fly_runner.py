"""Ejecuta una red de la mosca paso a paso: mantiene la memoria de los últimos cuadros
y devuelve la distribución sobre acciones. Lo usan DAgger (v10), el controlador (v12)
y el dashboard (v18)."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from ..data.generate import goal_features
from ..domain.drone import Action
from ..models.fly import FlyPolicyNet
from ..sim.observation import Observation


class FlyPolicyRunner:
    def __init__(self, model: FlyPolicyNet, device: str | torch.device = "cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.seq_len = model.cfg.seq_len
        self.frames: deque[np.ndarray] = deque(maxlen=self.seq_len)
        self.last_probs = np.full(model.cfg.n_actions, 1.0 / model.cfg.n_actions)

    def reset(self) -> None:
        self.frames.clear()

    def probabilities(self, obs: Observation, goal_feats: np.ndarray) -> np.ndarray:
        if obs.t == 0:
            self.reset()                                   # nuevo episodio
        frame = (obs.fly_patch > 0.5).astype(np.float32)
        if not self.frames:
            self.frames.extend([frame] * self.seq_len)     # relleno igual que en entrenamiento
        else:
            self.frames.append(frame)
        x = torch.from_numpy(np.stack(self.frames))[None].to(self.device)
        g = torch.from_numpy(np.asarray(goal_feats, dtype=np.float32))[None].to(self.device)
        self.last_probs = self.model.predict_proba(x, g)[0].cpu().numpy()
        return self.last_probs

    def act(self, obs: Observation, goal_feats: np.ndarray) -> Action:
        return Action(int(np.argmax(self.probabilities(obs, goal_feats))))

    # Firma de `BehaviorPolicy` (DAgger): política(obs, objetivo) → acción
    __call__ = act

    def act_towards(self, obs: Observation, goal: np.ndarray, estimate: np.ndarray) -> Action:
        return self.act(obs, goal_features(obs, goal, estimate))
