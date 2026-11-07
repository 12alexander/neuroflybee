"""Experto privilegiado para aprendizaje por imitación.

Conoce la posición real y el mapa completo, incluido el movimiento futuro de los
obstáculos móviles. Sólo se usa para ETIQUETAR datos; el estudiante (la mosca)
sólo ve su ojo egocéntrico y el objetivo relativo. Esta asimetría de información
es la que justifica la memoria temporal (GRU): inferir el movimiento a partir de
la secuencia de imágenes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import ProjectConfig
from ..control.steering import dilate, heading_to_action, nearest_free
from ..domain.drone import Action, DroneState, Kinematics
from ..domain.geometry import wrap_angle
from ..domain.world import World
from ..sim.grid import astar


class PrivilegedExpert:
    def __init__(self, world: World, cfg: ProjectConfig, inflate_m: float = 1.0,
                 lookahead_m: float = 1.5, horizon: int = 4, margin: float = 0.35):
        self.world, self.cfg = world, cfg
        self.kin = Kinematics(cfg.drone)
        self.inflate_cells = int(np.ceil(inflate_m / world.resolution))
        self.lookahead_m, self.horizon, self.margin = lookahead_m, horizon, margin
        self.path: Optional[list[tuple[int, int]]] = None
        self.path_goal: Optional[np.ndarray] = None
        self.t_plan = -10

    def _belief(self, t: int) -> np.ndarray:
        return dilate(self.world.occupancy(t), self.inflate_cells)

    def preferred_action(self, state: DroneState, goal: np.ndarray, t: int) -> Action:
        w = self.world
        need = (self.path is None or self.path_goal is None
                or np.linalg.norm(goal - self.path_goal) > 0.5 or t - self.t_plan >= 3)
        if need:
            belief = self._belief(t)
            start = nearest_free(belief, w.to_cell(state.position))
            target = nearest_free(belief, w.to_cell(goal))
            self.path = astar(belief, start, target)
            self.path_goal, self.t_plan = goal.copy(), t
        if not self.path:
            target_pt = goal
        else:
            pts = np.array([w.to_point(c) for c in self.path])
            nearest = int(np.argmin(np.linalg.norm(pts - state.position, axis=1)))
            k = min(len(pts) - 1, nearest + int(self.lookahead_m / w.resolution))
            target_pt = pts[k] if np.linalg.norm(goal - state.position) > self.lookahead_m else goal
        d = target_pt - state.position
        return heading_to_action(np.arctan2(d[1], d[0]) - state.heading, self.cfg.drone.soft_turn_deg)

    def _rollout_clearance(self, state: DroneState, first: Action, t: int) -> float:
        """Holgura mínima prevista: estáticos sólo en el paso siguiente (el A* ya los
        rodea con margen) y móviles en todo el horizonte (el experto ve su futuro)."""
        dynamic = self.world.dynamic_obstacles()
        s = self.kin.apply(state, first, self.world.wind)
        worst = float(self.world.clearance(s.position, t + 1, obstacles=self.world.static_obstacles())[0]) \
            + self.margin - 0.1
        for k in range(self.horizon):
            if k > 0:
                s = self.kin.apply(s, Action.FORWARD if first != Action.HOVER else Action.HOVER, self.world.wind)
            if dynamic:
                worst = min(worst, float(self.world.clearance(s.position, t + k + 1, obstacles=dynamic)[0]))
        return worst

    def act(self, state: DroneState, goal: np.ndarray, t: int) -> Action:
        preferred = self.preferred_action(state, goal, t)
        pref_turn = self.kin.turn_and_speed(preferred)[0]
        moving = [a for a in Action if a != Action.HOVER]
        ranked = sorted(moving, key=lambda a: (a != preferred,
                                               abs(wrap_angle(self.kin.turn_and_speed(a)[0] - pref_turn))))
        ranked.append(Action.HOVER)
        need = self.cfg.drone.radius + self.margin
        clearances = {}
        for a in ranked:
            c = self._rollout_clearance(state, a, t)
            clearances[a] = c
            if c >= need:
                return a
        return max(clearances, key=clearances.get)
