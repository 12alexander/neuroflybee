"""Sensores simulados.

* `FlyEyeSensor`: "ojo" de la mosca. Parche egocéntrico de ocupación (abstracción de
  una cámara frontal con estimación de profundidad), con ruido de píxel.
* `RangeSensor`: anillo de distancias (tipo ToF / VL53L0X) usado por el supervisor.
* `PanoramaSensor`: vista panorámica de la abeja. Silueta de horizonte (elevación
  angular de los objetos) alineada con el norte gracias a una brújula celeste.
* `GpsSensor`, `OdometrySensor`, `CompassSensor`: estado propio con ruido.

Ninguno es una cámara RGB real: es una abstracción declarada en el artículo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import SensorConfig
from ..domain.drone import DroneState
from ..domain.geometry import rotate, wrap_angle
from ..domain.world import CircleObstacle, World


def _lookup(occupancy: np.ndarray, world: World, points: np.ndarray) -> np.ndarray:
    """Valor de ocupación en puntos del mundo; fuera del mapa cuenta como ocupado."""
    rows, cols = occupancy.shape
    c = np.floor(points[..., 0] / world.resolution).astype(int)
    r = np.floor(points[..., 1] / world.resolution).astype(int)
    inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    out = np.ones(points.shape[:-1], dtype=bool)
    out[inside] = occupancy[r[inside], c[inside]]
    return out


class FlyEyeSensor:
    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg
        s = cfg.patch_size
        # filas: de lejos (arriba) a cerca (abajo); columnas: izquierda a derecha
        fwd = np.linspace(cfg.patch_forward, -cfg.patch_back, s)
        lat = np.linspace(cfg.patch_half_width, -cfg.patch_half_width, s)
        f, l = np.meshgrid(fwd, lat, indexing="ij")
        self._body_points = np.stack([f, l], axis=-1)          # (s, s, 2) en marco del dron

    def read(self, occupancy: np.ndarray, world: World, state: DroneState,
             rng: Optional[np.random.Generator] = None, failed_fraction: float = 0.0) -> np.ndarray:
        pts = rotate(self._body_points, state.heading) + state.position
        patch = _lookup(occupancy, world, pts).astype(np.float32)
        if rng is not None and self.cfg.pixel_noise > 0:
            flips = rng.random(patch.shape) < self.cfg.pixel_noise
            patch = np.where(flips, 1.0 - patch, patch)
        if rng is not None and failed_fraction > 0:
            patch[rng.random(patch.shape) < failed_fraction] = 0.0   # píxeles muertos
        return patch

    def body_point_of_pixel(self, row: int, col: int) -> np.ndarray:
        return self._body_points[row, col]

    def footprint(self, state: DroneState) -> np.ndarray:
        """Esquinas (4, 2) del área que cubre el ojo, en coordenadas del mundo."""
        c = self.cfg
        corners = np.array([[c.patch_forward, c.patch_half_width], [c.patch_forward, -c.patch_half_width],
                            [-c.patch_back, -c.patch_half_width], [-c.patch_back, c.patch_half_width]])
        return rotate(corners, state.heading) + state.position


class RangeSensor:
    def __init__(self, cfg: SensorConfig, step: float = 0.25):
        self.cfg = cfg
        self.angles = np.linspace(-np.pi, np.pi, cfg.n_rays, endpoint=False)  # relativos al rumbo
        self._dists = np.arange(step, cfg.ray_range + 1e-9, step)

    def read(self, occupancy: np.ndarray, world: World, state: DroneState) -> np.ndarray:
        ang = self.angles + state.heading
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=-1)                    # (n, 2)
        pts = state.position + dirs[:, None, :] * self._dists[None, :, None]     # (n, k, 2)
        hit = _lookup(occupancy, world, pts)
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), len(self._dists) - 1)
        ranges = np.where(hit.any(axis=1), self._dists[first], self.cfg.ray_range)
        return ranges.astype(np.float32)

    def end_points(self, ranges: np.ndarray, state: DroneState) -> np.ndarray:
        """Punto final (n, 2) de cada rayo en coordenadas del mundo."""
        ang = self.angles + state.heading
        return state.position + ranges[:, None] * np.stack([np.cos(ang), np.sin(ang)], axis=-1)


class PanoramaSensor:
    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg
        self.noise = cfg.panorama_noise
        n = cfg.panorama_bins
        self.bin_centers = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n

    def objects(self, world: World) -> np.ndarray:
        """(m, 4): x, y, radio, altura de todo lo que se ve en el horizonte."""
        rows = [(lm.x, lm.y, lm.r, lm.height) for lm in world.landmarks]
        for o in world.static_obstacles():
            c = o.center()
            r = o.r if isinstance(o, CircleObstacle) else 0.8 * o.footprint_radius()
            rows.append((c[0], c[1], r, o.height))
        return np.array(rows, dtype=float).reshape(-1, 4)

    def view_from(self, objects: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Silueta (bins,) en [0, 1] vista desde `position`. Vectorizada sobre objetos."""
        if len(objects) == 0:
            return np.zeros(self.cfg.panorama_bins, dtype=np.float32)
        delta = objects[:, :2] - position
        d = np.maximum(np.linalg.norm(delta, axis=1), 1e-3)
        b = np.arctan2(delta[:, 1], delta[:, 0])
        half = np.arcsin(np.clip(objects[:, 2] / d, 0, 1)) + np.pi / self.cfg.panorama_bins
        elev = np.arctan2(objects[:, 3], d) / (np.pi / 2)
        diff = np.abs(wrap_angle(self.bin_centers[None, :] - b[:, None]))
        covered = diff <= half[:, None]
        return np.max(np.where(covered, elev[:, None], 0.0), axis=0).astype(np.float32)

    def read(self, world: World, state: DroneState, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        view = self.view_from(self.objects(world), state.position)
        if rng is not None and self.noise > 0:
            view = np.clip(view + rng.normal(0, self.noise, view.shape), 0, 1).astype(np.float32)
        return view


@dataclass
class GpsSensor:
    sigma: float

    def read(self, state: DroneState, available: bool, rng: np.random.Generator) -> Optional[np.ndarray]:
        if not available:
            return None
        return state.position + rng.normal(0, self.sigma, 2)


@dataclass
class CompassSensor:
    sigma_rad: float = np.radians(2.0)

    def read(self, state: DroneState, rng: np.random.Generator) -> float:
        return float(wrap_angle(state.heading + rng.normal(0, self.sigma_rad)))


@dataclass
class OdometrySensor:
    """Desplazamiento estimado por IMU/flujo óptico. No percibe el viento: por eso la
    integración de trayectoria acumula deriva cuando sopla viento."""
    relative_sigma: float

    def read(self, commanded: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        scale = 1.0 + rng.normal(0, self.relative_sigma)
        return commanded * scale + rng.normal(0, 0.01, 2)
