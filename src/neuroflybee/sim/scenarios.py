"""Generación procedural y reproducible de escenarios.

* Seis condiciones (familias) que sólo difieren en lo que declaran en `FAMILIES`.
* Particiones entrenamiento / validación / prueba con rangos de semillas disjuntos:
  ningún escenario de prueba puede aparecer durante el entrenamiento.
* Cada escenario se verifica: siempre existe una ruta libre entre la base y todos
  los puntos de inspección (si no, se descarta y se genera otro).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

import numpy as np

from ..config import ProjectConfig, ScenarioConfig
from ..domain.world import CircleObstacle, Landmark, MovingObstacle, Obstacle, RectObstacle, World
from .grid import all_reachable


# --------------------------------------------------------------------- particiones
class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


SPLIT_RANGES: dict[Split, range] = {
    Split.TRAIN: range(0, 10_000),
    Split.VAL: range(10_000, 11_000),
    Split.TEST: range(20_000, 21_000),
}


def seeds_for(split: Split | str, n: int, offset: int = 0) -> list[int]:
    """Las `n` semillas de una partición, empezando en `offset`."""
    rng = SPLIT_RANGES[Split(split)]
    if offset + n > len(rng):
        raise ValueError(f"La partición {split} sólo tiene {len(rng)} semillas")
    return list(rng[offset:offset + n])


def split_of(seed: int) -> Split:
    for split, rng in SPLIT_RANGES.items():
        if seed in rng:
            return split
    raise ValueError(f"Semilla {seed} fuera de toda partición")


# --------------------------------------------------------------------- familias
FAMILIES: dict[str, dict] = {
    "nominal": {},
    "unknown_obstacles": {"n_unknown_obstacles": (4, 8)},
    "dynamic": {"n_moving_obstacles": (2, 4)},
    "wind": {"wind_speed": (0.10, 0.25)},
    "gps_loss": {"gps_loss_on_return": True, "wind_speed": (0.05, 0.15)},
    "combined": {"n_unknown_obstacles": (3, 6), "n_moving_obstacles": (1, 3),
                 "wind_speed": (0.05, 0.15), "gps_loss_on_return": True},
}
FAMILY_LABELS_ES = {
    "nominal": "Nominal",
    "unknown_obstacles": "Obstáculos no previstos",
    "dynamic": "Obstáculos móviles",
    "wind": "Viento",
    "gps_loss": "Pérdida de GPS al regresar",
    "combined": "Combinado (estrés)",
}


@dataclass(frozen=True)
class Scenario:
    family: str
    seed: int
    world: World
    gps_loss_on_return: bool
    max_steps: int

    @property
    def split(self) -> Split:
        return split_of(self.seed)

    @property
    def route(self) -> list[np.ndarray]:
        """Ruta planificada: base → puntos de inspección → base."""
        return [self.world.base, *self.world.waypoints, self.world.base]


# --------------------------------------------------------------------- generador
class ScenarioGenerator:
    """Construye un `Scenario` determinista a partir de (familia, semilla)."""

    MAX_ATTEMPTS = 50
    BORDER_MARGIN = 6.0         # la base no queda pegada al borde
    WAYPOINT_MIN_FROM_BASE = 12.0
    WAYPOINT_MIN_SPACING = 8.0
    FREE_AROUND_WAYPOINT = 2.5

    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg

    def family_config(self, family: str) -> ScenarioConfig:
        if family not in FAMILIES:
            raise KeyError(f"Familia desconocida: {family}. Opciones: {list(FAMILIES)}")
        return replace(self.cfg.scenario, **FAMILIES[family])

    def generate(self, family: str, seed: int) -> Scenario:
        scfg = self.family_config(family)
        rng = np.random.default_rng([seed, list(FAMILIES).index(family)])
        for _ in range(self.MAX_ATTEMPTS):
            world = self._try_world(scfg, rng)
            if world is not None:
                return Scenario(family, seed, world, scfg.gps_loss_on_return, scfg.max_steps)
        raise RuntimeError(f"No se pudo generar un escenario factible ({family}, {seed})")

    # ---------------------------------------------------------- pasos de construcción
    def _try_world(self, scfg: ScenarioConfig, rng: np.random.Generator) -> Optional[World]:
        W, H = self.cfg.world.width, self.cfg.world.height
        base = rng.uniform([self.BORDER_MARGIN] * 2, [W - self.BORDER_MARGIN, H - self.BORDER_MARGIN])
        waypoints = self._place_waypoints(scfg, rng, base)
        if waypoints is None:
            return None
        protected = [base, *waypoints]
        route = [base, *waypoints, base]
        obstacles: list[Obstacle] = []
        obstacles += self._place_known(scfg, rng, protected)
        obstacles += self._place_unknown(scfg, rng, protected, route)
        obstacles += self._place_moving(scfg, rng, route)
        if not self._is_feasible(base, waypoints, obstacles):
            return None
        return World(W, H, self.cfg.world.resolution, base, waypoints, obstacles,
                     self._place_landmarks(scfg, rng), self._sample_wind(scfg, rng))

    def _place_waypoints(self, scfg: ScenarioConfig, rng: np.random.Generator,
                         base: np.ndarray) -> Optional[list[np.ndarray]]:
        W, H = self.cfg.world.width, self.cfg.world.height
        n = int(rng.integers(scfg.n_waypoints[0], scfg.n_waypoints[1] + 1))
        points: list[np.ndarray] = []
        for _ in range(400):
            if len(points) == n:
                return self._order_route(base, points)
            p = rng.uniform([4, 4], [W - 4, H - 4])
            far_from_base = np.linalg.norm(p - base) >= self.WAYPOINT_MIN_FROM_BASE
            spaced = all(np.linalg.norm(p - q) >= self.WAYPOINT_MIN_SPACING for q in points)
            if far_from_base and spaced:
                points.append(p)
        return None

    def _place_known(self, scfg, rng, protected) -> list[Obstacle]:
        W, H = self.cfg.world.width, self.cfg.world.height
        n = int(rng.integers(scfg.n_known_obstacles[0], scfg.n_known_obstacles[1] + 1))
        placed: list[Obstacle] = []
        for _ in range(n * 20):
            if len(placed) >= n:
                break
            obs = self._random_static(rng, W, H)
            if self._keeps_free(obs, protected):
                placed.append(obs)
        return placed

    def _place_unknown(self, scfg, rng, protected, route) -> list[Obstacle]:
        """Obstáculos que NO están en el mapa previo, puestos cerca de la ruta para
        que el dron tenga que reaccionar."""
        n = int(rng.integers(scfg.n_unknown_obstacles[0], scfg.n_unknown_obstacles[1] + 1))
        placed: list[Obstacle] = []
        for _ in range(n * 30):
            if len(placed) >= n:
                break
            a, b = self._random_leg(rng, route)
            p = a + rng.uniform(0.3, 0.7) * (b - a) + self._normal(a, b) * rng.uniform(-1.5, 1.5)
            obs = CircleObstacle(p[0], p[1], rng.uniform(0.8, 1.8), height=rng.uniform(1.5, 4),
                                 known=False, kind="no previsto")
            if self._keeps_free(obs, protected):
                placed.append(obs)
        return placed

    def _place_moving(self, scfg, rng, route) -> list[Obstacle]:
        """Obstáculos que cruzan la ruta de forma perpendicular, de ida y vuelta."""
        n = int(rng.integers(scfg.n_moving_obstacles[0], scfg.n_moving_obstacles[1] + 1))
        placed: list[Obstacle] = []
        for _ in range(n):
            a, b = self._random_leg(rng, route)
            normal = self._normal(a, b)
            span, speed = rng.uniform(8, 12), rng.uniform(0.3, 0.6)
            start = a + rng.uniform(0.35, 0.65) * (b - a) - normal * span / 2
            placed.append(MovingObstacle(start[0], start[1], normal[0] * speed, normal[1] * speed,
                                         r=rng.uniform(0.5, 0.9), span=span))
        return placed

    def _is_feasible(self, base, waypoints, obstacles) -> bool:
        W, H, res = self.cfg.world.width, self.cfg.world.height, self.cfg.world.resolution
        static = [o for o in obstacles if not o.is_dynamic]
        probe = World(W, H, res, base, waypoints, static)
        occupied = probe.occupancy(inflate=self.cfg.drone.radius + 0.1)
        return all_reachable(occupied, probe.to_cell(base), [probe.to_cell(w) for w in waypoints])

    @staticmethod
    def _place_landmarks(scfg: ScenarioConfig, rng: np.random.Generator) -> list[Landmark]:
        """Cerros lejanos fuera del área de vuelo: forman la silueta del horizonte."""
        n = int(rng.integers(scfg.n_landmarks[0], scfg.n_landmarks[1] + 1))
        out = []
        for _ in range(n):
            ang, dist = rng.uniform(-np.pi, np.pi), rng.uniform(45, 75)
            x, y = 30 + dist * np.cos(ang), 30 + dist * np.sin(ang)
            out.append(Landmark(x, y, rng.uniform(4, 12), rng.uniform(10, 35)))
        return out

    @staticmethod
    def _sample_wind(scfg: ScenarioConfig, rng: np.random.Generator) -> np.ndarray:
        speed, direction = rng.uniform(*scfg.wind_speed), rng.uniform(-np.pi, np.pi)
        return speed * np.array([np.cos(direction), np.sin(direction)])

    # ------------------------------------------------------------------ utilidades
    @staticmethod
    def _order_route(base: np.ndarray, points: list[np.ndarray]) -> list[np.ndarray]:
        """Ordena los puntos por vecino más cercano (ruta de inspección razonable)."""
        remaining, route, current = list(points), [], base
        while remaining:
            idx = int(np.argmin([np.linalg.norm(p - current) for p in remaining]))
            current = remaining.pop(idx)
            route.append(current)
        return route

    @staticmethod
    def _random_leg(rng: np.random.Generator, route: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        i = int(rng.integers(len(route) - 1))
        return route[i], route[i + 1]

    @staticmethod
    def _normal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = b - a
        return np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)

    @staticmethod
    def _random_static(rng: np.random.Generator, W: float, H: float) -> Obstacle:
        if rng.random() < 0.7:
            return CircleObstacle(rng.uniform(2, W - 2), rng.uniform(2, H - 2), rng.uniform(0.8, 2.5),
                                  height=rng.uniform(2, 8), kind=str(rng.choice(["roca", "árbol", "poste"])))
        length, thick = rng.uniform(4, 10), rng.uniform(0.6, 1.2)
        x, y = rng.uniform(2, W - 12), rng.uniform(2, H - 12)
        if rng.random() < 0.5:
            return RectObstacle(x, y, x + length, y + thick, height=rng.uniform(3, 6))
        return RectObstacle(x, y, x + thick, y + length, height=rng.uniform(3, 6))

    def _keeps_free(self, obs: Obstacle, protected: list[np.ndarray]) -> bool:
        d = obs.signed_distance(np.array(protected))
        return bool(d[0] > self.cfg.world.base_clearance and np.all(d[1:] > self.FREE_AROUND_WAYPOINT))
