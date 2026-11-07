"""Configuración tipada del proyecto.

Los hiperparámetros viven en `configs/*.yaml`; este módulo los convierte en
dataclasses inmutables para que ningún componente lea diccionarios sueltos.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class WorldConfig:
    width: float = 60.0           # m
    height: float = 60.0          # m
    resolution: float = 0.5       # m por celda del mapa de ocupación
    base_clearance: float = 4.0   # radio libre alrededor de la base


@dataclass(frozen=True)
class DroneConfig:
    speed: float = 0.8            # m por paso (dt = 1 s)
    radius: float = 0.4           # m
    soft_turn_deg: float = 15.0
    hard_turn_deg: float = 45.0
    hard_turn_speed_factor: float = 0.3


@dataclass(frozen=True)
class SensorConfig:
    patch_size: int = 32          # píxeles del "ojo" de la mosca
    patch_forward: float = 10.0   # m visibles hacia adelante
    patch_back: float = 2.0       # m visibles hacia atrás
    patch_half_width: float = 6.0 # m a cada lado
    n_rays: int = 32
    ray_range: float = 8.0
    panorama_bins: int = 72       # 5° por bin
    gps_sigma: float = 0.5        # m
    odometry_sigma: float = 0.05  # fracción del desplazamiento
    pixel_noise: float = 0.02     # probabilidad de invertir un píxel
    panorama_noise: float = 0.01  # desviación del ruido del panorama (fracción de 90°)


@dataclass(frozen=True)
class ScenarioConfig:
    n_waypoints: tuple[int, int] = (3, 5)
    n_known_obstacles: tuple[int, int] = (10, 18)
    n_unknown_obstacles: tuple[int, int] = (0, 0)
    n_moving_obstacles: tuple[int, int] = (0, 0)
    n_landmarks: tuple[int, int] = (10, 16)
    wind_speed: tuple[float, float] = (0.0, 0.0)  # m por paso
    gps_loss_on_return: bool = False
    max_steps: int = 600


@dataclass(frozen=True)
class ProjectConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    extra: dict[str, Any] = field(default_factory=dict)


def from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    """Construye una dataclass (anidada) desde un diccionario, ignorando claves ajenas
    sólo en `extra`; cualquier otra clave desconocida es un error explícito."""
    data = dict(data or {})
    kwargs: dict[str, Any] = {}
    names = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(names)
    if unknown and "extra" not in names:
        raise KeyError(f"Claves desconocidas para {cls.__name__}: {sorted(unknown)}")
    for name, f in names.items():
        if name not in data:
            continue
        value = data.pop(name)
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        if is_dataclass(default) and isinstance(value, dict):
            value = from_dict(type(default), value)
        elif isinstance(default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    if "extra" in names:
        kwargs["extra"] = {**kwargs.get("extra", {}), **data}
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Carga un YAML; sin ruta devuelve los valores por defecto."""
    if path is None:
        return ProjectConfig()
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return from_dict(ProjectConfig, raw)
