"""Modelo del mundo: obstáculos, hitos visuales, base y puntos de inspección.

Cada obstáculo implementa el mismo protocolo (`signed_distance`), de modo que el
mundo puede contener tipos nuevos sin modificarse (principio abierto/cerrado).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Obstacle(Protocol):
    known: bool       # ¿figura en el mapa previo con el que planifica un controlador clásico?
    height: float     # altura aparente (m) usada por la vista panorámica de la abeja

    def signed_distance(self, points: np.ndarray, t: int = 0) -> np.ndarray: ...
    def center(self, t: int = 0) -> np.ndarray: ...
    def footprint_radius(self) -> float: ...
    @property
    def is_dynamic(self) -> bool: ...


@dataclass(frozen=True)
class CircleObstacle:
    x: float
    y: float
    r: float
    height: float = 3.0
    known: bool = True
    kind: str = "roca"

    def center(self, t: int = 0) -> np.ndarray:
        return np.array([self.x, self.y])

    def signed_distance(self, points: np.ndarray, t: int = 0) -> np.ndarray:
        return np.linalg.norm(np.asarray(points) - self.center(t), axis=-1) - self.r

    def footprint_radius(self) -> float:
        return self.r

    @property
    def is_dynamic(self) -> bool:
        return False


@dataclass(frozen=True)
class RectObstacle:
    """Rectángulo alineado con los ejes (muros, compuertas, edificaciones)."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    height: float = 5.0
    known: bool = True
    kind: str = "muro"

    def center(self, t: int = 0) -> np.ndarray:
        return np.array([(self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2])

    def signed_distance(self, points: np.ndarray, t: int = 0) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        c = self.center()
        half = np.array([(self.xmax - self.xmin) / 2, (self.ymax - self.ymin) / 2])
        q = np.abs(p - c) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside

    def footprint_radius(self) -> float:
        return float(np.hypot(self.xmax - self.xmin, self.ymax - self.ymin) / 2)

    @property
    def is_dynamic(self) -> bool:
        return False


@dataclass(frozen=True)
class MovingObstacle:
    """Obstáculo circular que se desplaza y rebota en un segmento (aves, troncos flotando)."""
    x0: float
    y0: float
    vx: float
    vy: float
    r: float
    span: float = 10.0          # longitud del vaivén (m)
    height: float = 1.0
    known: bool = False
    kind: str = "móvil"

    def center(self, t: int = 0) -> np.ndarray:
        speed = float(np.hypot(self.vx, self.vy)) or 1e-9
        travelled = (speed * t) % (2 * self.span)
        s = travelled if travelled <= self.span else 2 * self.span - travelled
        direction = np.array([self.vx, self.vy]) / speed
        return np.array([self.x0, self.y0]) + direction * s

    def signed_distance(self, points: np.ndarray, t: int = 0) -> np.ndarray:
        return np.linalg.norm(np.asarray(points) - self.center(t), axis=-1) - self.r

    def footprint_radius(self) -> float:
        return self.r

    @property
    def is_dynamic(self) -> bool:
        return True


@dataclass(frozen=True)
class Landmark:
    """Referencia visual lejana (cerros, torres) que no obstruye el vuelo."""
    x: float
    y: float
    r: float
    height: float


@dataclass
class World:
    """Escenario completo. Se considera inmutable una vez construido: los mapas de
    ocupación estáticos se calculan una sola vez y se guardan en caché."""
    width: float
    height: float
    resolution: float
    base: np.ndarray
    waypoints: list[np.ndarray]
    obstacles: list[Obstacle] = field(default_factory=list)
    landmarks: list[Landmark] = field(default_factory=list)
    wind: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _occ_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------ consultas
    @property
    def shape(self) -> tuple[int, int]:
        return int(round(self.height / self.resolution)), int(round(self.width / self.resolution))

    @cached_property
    def cell_centers(self) -> np.ndarray:
        rows, cols = self.shape
        ys = (np.arange(rows) + 0.5) * self.resolution
        xs = (np.arange(cols) + 0.5) * self.resolution
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx, gy], axis=-1)

    def static_obstacles(self, known_only: bool = False) -> list[Obstacle]:
        return [o for o in self.obstacles if not o.is_dynamic and (o.known or not known_only)]

    def dynamic_obstacles(self) -> list[Obstacle]:
        return [o for o in self.obstacles if o.is_dynamic]

    def clearance(self, points: np.ndarray, t: int = 0, known_only: bool = False,
                  obstacles: Sequence[Obstacle] | None = None) -> np.ndarray:
        """Distancia (m) al obstáculo o borde más cercano; negativa si hay colisión."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        border = np.minimum.reduce([pts[:, 0], pts[:, 1], self.width - pts[:, 0], self.height - pts[:, 1]])
        obs = obstacles if obstacles is not None else [
            o for o in self.obstacles if o.known or not known_only]
        if not obs:
            return border
        d = np.min(np.stack([o.signed_distance(pts, t) for o in obs]), axis=0)
        return np.minimum(d, border)

    def occupancy(self, t: int = 0, known_only: bool = False, inflate: float = 0.0) -> np.ndarray:
        """Mapa de ocupación booleano (filas = y, columnas = x)."""
        static = self._static_occupancy(known_only, inflate)
        dyn = [] if known_only else self.dynamic_obstacles()
        if not dyn:
            return static
        grid = self.cell_centers.reshape(-1, 2)
        d = self.clearance(grid, t, obstacles=dyn).reshape(self.shape)
        return static | (d <= inflate)

    def _static_occupancy(self, known_only: bool, inflate: float) -> np.ndarray:
        key = (known_only, round(inflate, 3))
        if key not in self._occ_cache:
            grid = self.cell_centers.reshape(-1, 2)
            obs = self.static_obstacles(known_only)
            d = self.clearance(grid, obstacles=obs).reshape(self.shape)
            self._occ_cache[key] = d <= inflate
        return self._occ_cache[key]

    def to_cell(self, point: np.ndarray) -> tuple[int, int]:
        rows, cols = self.shape
        c = int(np.clip(point[0] / self.resolution, 0, cols - 1))
        r = int(np.clip(point[1] / self.resolution, 0, rows - 1))
        return r, c

    def to_point(self, cell: tuple[int, int]) -> np.ndarray:
        r, c = cell
        return np.array([(c + 0.5) * self.resolution, (r + 0.5) * self.resolution])
