"""Registro de controladores: agregar uno nuevo no exige modificar el benchmark
ni el dashboard (principio abierto/cerrado)."""
from __future__ import annotations

from typing import Callable

from .base import Controller

ControllerFactory = Callable[[], Controller]
_REGISTRY: dict[str, tuple[ControllerFactory, str]] = {}


def register(name: str, description: str = ""):
    def deco(factory: ControllerFactory) -> ControllerFactory:
        _REGISTRY[name] = (factory, description)
        return factory
    return deco


def create(name: str) -> Controller:
    if name not in _REGISTRY:
        raise KeyError(f"Controlador desconocido '{name}'. Disponibles: {available()}")
    return _REGISTRY[name][0]()


def available() -> list[str]:
    return list(_REGISTRY)


def describe() -> dict[str, str]:
    return {k: v[1] for k, v in _REGISTRY.items()}
