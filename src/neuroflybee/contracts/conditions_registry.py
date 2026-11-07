"""Catálogo de condiciones de vuelo con nombre (Abierto/Cerrado).

Cada módulo registra las suyas al importarse, sin editar este archivo::

    from neuroflybee.contracts import conditions_registry
    conditions_registry.register(Conditions("dryden_moderada", wind=..., description="..."))
"""
from __future__ import annotations

from .disturbances import Conditions

_CONDITIONS: dict[str, Conditions] = {}


def register(conditions: Conditions, replace: bool = False) -> Conditions:
    if conditions.name in _CONDITIONS and not replace:
        raise ValueError(f"Ya existe una condición llamada '{conditions.name}'")
    _CONDITIONS[conditions.name] = conditions
    return conditions


def get(name: str) -> Conditions:
    try:
        return _CONDITIONS[name]
    except KeyError:
        raise KeyError(f"Condición desconocida '{name}'. Disponibles: {available()}") from None


def available() -> list[str]:
    return sorted(_CONDITIONS)


def describe() -> dict[str, str]:
    return {n: c.description for n, c in sorted(_CONDITIONS.items())}


register(Conditions("escenario", description="Viento y sensores del escenario (v1–v17)"))
