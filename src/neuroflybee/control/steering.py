"""Estrategias de control local: transforman (observación, objetivo) en una acción.

Cada estrategia es intercambiable (patrón Strategy); el controlador compuesto no
sabe si la acción la decide una regla, un planificador o una red neuronal.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from ..config import SensorConfig
from ..domain.drone import Action
from ..domain.geometry import wrap_angle
from ..sim.grid import astar
from ..sim.observation import Observation

RAY_RANGE_EPS = 0.05
DEFAULT_RAY_RANGE = SensorConfig.ray_range


class Steering(Protocol):
    name: str
    source: str

    def reset(self, obs: Observation) -> None: ...

    def steer(self, obs: Observation, goal: np.ndarray, estimate: np.ndarray) -> tuple[Action, dict]: ...


def heading_to_action(relative: float, soft_deg: float = 15.0) -> Action:
    """Acción discreta que mejor sigue un ángulo relativo al rumbo actual."""
    a = np.degrees(wrap_angle(relative))
    if abs(a) <= soft_deg / 2:
        return Action.FORWARD
    if abs(a) <= 2 * soft_deg:
        return Action.LEFT if a > 0 else Action.RIGHT
    return Action.SHARP_LEFT if a > 0 else Action.SHARP_RIGHT


def ray_directions(obs: Observation) -> np.ndarray:
    """Ángulo de cada rayo relativo al rumbo (mismo orden que `RangeSensor.angles`)."""
    return np.linspace(-np.pi, np.pi, len(obs.ranges), endpoint=False)


class DirectSteering:
    """Línea base convencional: va en línea recta al objetivo, sin evadir."""
    name, source = "directo", "directo"

    def reset(self, obs: Observation) -> None:
        pass

    def steer(self, obs, goal, estimate):
        d = goal - estimate
        return heading_to_action(np.arctan2(d[1], d[0]) - obs.heading), {}


class PotentialFieldSteering:
    """Controlador reactivo clásico (campos de potencial, Khatib 1986)."""
    name, source = "campo_potencial", "reactivo"

    def __init__(self, influence: float = 3.0, gain: float = 1.5):
        self.influence, self.gain = influence, gain

    def reset(self, obs: Observation) -> None:
        pass

    def steer(self, obs, goal, estimate):
        d = goal - estimate
        attract = d / (np.linalg.norm(d) + 1e-9)
        rel = ray_directions(obs) + obs.heading
        r = np.maximum(obs.ranges, 0.05)
        mag = np.where(r < self.influence, self.gain * (1 / r - 1 / self.influence), 0.0)
        repulse = -(mag[:, None] * np.stack([np.cos(rel), np.sin(rel)], axis=1)).sum(axis=0)
        v = attract + repulse
        return heading_to_action(np.arctan2(v[1], v[0]) - obs.heading), {"repulsion": float(np.linalg.norm(repulse))}


def safest_heading(obs: Observation, desired: float, min_clear: float = 2.0) -> float:
    """Entre las direcciones de los rayos con espacio libre >= `min_clear`, la más
    parecida a la deseada (marco del mundo). Si ninguna está libre, la más despejada."""
    world_angles = ray_directions(obs) + obs.heading
    free = obs.ranges >= min_clear
    if not free.any():
        return float(world_angles[int(np.argmax(obs.ranges))])
    gap = np.abs(wrap_angle(world_angles - desired))
    gap = np.where(free, gap, np.inf)
    return float(world_angles[int(np.argmin(gap))])


def nearest_free(occupied: np.ndarray, cell: tuple[int, int], max_radius: int = 6) -> tuple[int, int]:
    if not occupied[cell]:
        return cell
    rows, cols = occupied.shape
    for rad in range(1, max_radius + 1):
        best, best_d = None, np.inf
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                r, c = cell[0] + dr, cell[1] + dc
                if 0 <= r < rows and 0 <= c < cols and not occupied[r, c] and dr * dr + dc * dc < best_d:
                    best, best_d = (r, c), dr * dr + dc * dc
        if best is not None:
            return best
    return cell


def dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Dilatación binaria con disco (sin SciPy)."""
    out = mask.copy()
    rows, cols = mask.shape
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc > radius_cells * radius_cells or (dr == 0 and dc == 0):
                continue
            src = mask[max(0, -dr):rows - max(0, dr), max(0, -dc):cols - max(0, dc)]
            out[max(0, dr):rows - max(0, -dr), max(0, dc):cols - max(0, -dc)] |= src
    return out


class AStarSteering:
    """Planificador clásico: A* sobre el mapa previo + obstáculos detectados por el
    anillo de distancias, con replanificación. Es la línea base clásica más fuerte."""
    name, source = "a_estrella", "planificador"

    def __init__(self, inflate_m: float = 1.0, replan_every: int = 5, lookahead_m: float = 1.5,
                 memory_steps: int = 20, ray_range: float = DEFAULT_RAY_RANGE):
        self.ray_range = ray_range
        self.inflate_m, self.replan_every = inflate_m, replan_every
        self.lookahead_m, self.memory_steps = lookahead_m, memory_steps

    def reset(self, obs: Observation) -> None:
        pm = obs.prior_map
        self.res = pm.resolution
        self.prior = pm.occupancy
        self.sensed_age = np.full(pm.occupancy.shape, -10_000, dtype=int)
        self.path: Optional[list[tuple[int, int]]] = None
        self.last_goal: Optional[np.ndarray] = None
        self.steps = 0

    def _cell(self, p: np.ndarray) -> tuple[int, int]:
        rows, cols = self.prior.shape
        return (int(np.clip(p[1] / self.res, 0, rows - 1)), int(np.clip(p[0] / self.res, 0, cols - 1)))

    def _point(self, cell: tuple[int, int]) -> np.ndarray:
        return np.array([(cell[1] + 0.5) * self.res, (cell[0] + 0.5) * self.res])

    def _belief(self, t: int) -> np.ndarray:
        sensed = (t - self.sensed_age) <= self.memory_steps
        return dilate(self.prior | sensed, int(np.ceil(self.inflate_m / self.res)))

    def steer(self, obs, goal, estimate):
        self.steps += 1
        hits = obs.ranges < self.ray_range - RAY_RANGE_EPS   # rayos que tocaron algo
        if hits.any():
            ang = ray_directions(obs)[hits] + obs.heading
            pts = estimate + obs.ranges[hits][:, None] * np.stack([np.cos(ang), np.sin(ang)], axis=1)
            for p in pts:
                self.sensed_age[self._cell(p)] = obs.t
        goal_changed = self.last_goal is None or np.linalg.norm(goal - self.last_goal) > 0.5
        blocked = False
        belief = None
        if self.path:
            belief = self._belief(obs.t)
            blocked = any(belief[c] for c in self.path[:12])
        if goal_changed or blocked or self.path is None or self.steps % self.replan_every == 0:
            belief = belief if belief is not None else self._belief(obs.t)
            start = nearest_free(belief, self._cell(estimate))
            target = nearest_free(belief, self._cell(goal))
            self.path = astar(belief, start, target)
            self.last_goal = goal.copy()
        if not self.path:
            d = goal - estimate
            heading = safest_heading(obs, float(np.arctan2(d[1], d[0])))
            return heading_to_action(heading - obs.heading), {"plan": "sin ruta"}
        # avanzar sobre la ruta hasta el punto de anticipación
        pts = np.array([self._point(c) for c in self.path])
        nearest = int(np.argmin(np.linalg.norm(pts - estimate, axis=1)))
        self.path = self.path[nearest:]
        pts = pts[nearest:]
        k = min(len(pts) - 1, int(self.lookahead_m / self.res))
        target = pts[k] if np.linalg.norm(goal - estimate) > self.lookahead_m else goal
        d = target - estimate
        return heading_to_action(np.arctan2(d[1], d[0]) - obs.heading), {"plan_len": len(self.path)}
