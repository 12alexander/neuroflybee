"""Supervisor de seguridad (patrón decorador): envuelve a CUALQUIER controlador y veta
las maniobras que, según el anillo de distancias, acercarían el dron a un obstáculo.

El margen (0.6 m además del radio) cubre la resolución del sensor: los rayos avanzan
de 0.25 m en 0.25 m sobre un mapa de 0.5 m por celda.

Es independiente de las redes y del GPS (sólo usa los rayos), por eso puede validar
lo que proponga una red experimental. Si veta, elige la maniobra segura más parecida a
la propuesta; si ninguna es segura, mantiene la posición.
"""
from __future__ import annotations

import numpy as np

from ..config import DroneConfig
from ..domain.drone import Action, Kinematics
from ..domain.geometry import wrap_angle
from ..sim.observation import Observation
from .base import Controller, Decision


class SafetySupervisor:
    def __init__(self, inner: Controller, drone: DroneConfig | None = None, margin: float = 0.6,
                 name: str | None = None):
        self.inner = inner
        self.name = name or f"{inner.name}+supervisor"
        self.kin = Kinematics(drone or DroneConfig())
        self.radius = (drone or DroneConfig()).radius
        self.margin = margin
        self.interventions = 0

    def reset(self, obs: Observation) -> None:
        self.inner.reset(obs)
        self.interventions = 0

    def clearance_for(self, obs: Observation, action: Action) -> float:
        """Distancia mínima entre la posición prevista tras la maniobra y los puntos donde
        los rayos tocaron un obstáculo (marco del dron; no necesita GPS)."""
        turn, speed = self.kin.turn_and_speed(action)
        if speed == 0:
            return np.inf
        rays = np.linspace(-np.pi, np.pi, len(obs.ranges), endpoint=False)
        hits = obs.ranges < obs.ranges.max() if np.ptp(obs.ranges) > 0 else obs.ranges < 0
        if not hits.any():
            return np.inf
        points = obs.ranges[hits, None] * np.stack([np.cos(rays[hits]), np.sin(rays[hits])], axis=1)
        moved = speed * np.array([np.cos(turn), np.sin(turn)])
        return float(np.linalg.norm(points - moved, axis=1).min())

    def is_safe(self, obs: Observation, action: Action) -> bool:
        return self.clearance_for(obs, action) >= self.radius + self.margin

    def act(self, obs: Observation) -> Decision:
        decision = self.inner.act(obs)
        if self.is_safe(obs, decision.action):
            decision.debug["supervisor"] = "ok"
            return decision
        proposed = decision.action
        turn = self.kin.turn_and_speed(proposed)[0]
        candidates = sorted((a for a in Action if a != Action.HOVER),
                            key=lambda a: abs(wrap_angle(self.kin.turn_and_speed(a)[0] - turn)))
        safe = [a for a in candidates if self.is_safe(obs, a)]
        decision.action = safe[0] if safe else Action.HOVER     # nada es seguro: mantener posición
        decision.source = "supervisor"
        decision.debug.update({"supervisor": "veto", "vetoed": int(proposed)})
        self.interventions += 1
        decision.debug["interventions"] = self.interventions
        return decision
