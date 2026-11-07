"""Interfaces de perturbación del entorno (patrones Strategy y Decorator).

El entorno (`NavigationEnv`) depende de estas abstracciones y no de implementaciones
concretas (inversión de dependencias): agregar un modelo de viento o una falla de sensor
nuevo no obliga a modificar el simulador.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # evita importaciones circulares en tiempo de ejecución
    from ..sim.observation import Observation
    from ..sim.scenarios import Scenario


@runtime_checkable
class WindModel(Protocol):
    """Velocidad del viento [m/s] en el plano, en el paso `t` y la posición dada."""

    def reset(self, rng: np.random.Generator) -> None:
        """Se llama al inicio de cada episodio (estado interno, p. ej. filtros de turbulencia)."""

    def velocity(self, t: int, position: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Devuelve un vector (2,) en m/s."""


@runtime_checkable
class SensorPerturbation(Protocol):
    """Degrada una observación ya construida (decorador sobre los sensores).

    Debe devolver una observación NUEVA (o la misma si no cambia nada) y no mutar la
    entrada: el entorno guarda la observación limpia para depuración.
    """

    name: str

    def reset(self, rng: np.random.Generator) -> None:
        """Inicio de episodio (p. ej. vaciar el búfer de una latencia)."""

    def apply(self, observation: "Observation", rng: np.random.Generator) -> "Observation":
        """Devuelve la observación degradada."""


WindFactory = Callable[["Scenario"], WindModel]


@dataclass(frozen=True)
class Conditions:
    """Condiciones de vuelo inyectables: viento y perturbaciones de sensores.

    `wind=None` conserva el viento del escenario (comportamiento de v1–v17), así que todo lo
    medido antes sigue siendo reproducible.
    """

    name: str = "escenario"
    wind: Optional[WindFactory] = None
    perturbations: tuple[Callable[[], SensorPerturbation], ...] = field(default_factory=tuple)
    description: str = ""

    def build_wind(self, scenario: "Scenario") -> Optional[WindModel]:
        return self.wind(scenario) if self.wind else None

    def build_perturbations(self) -> list[SensorPerturbation]:
        return [make() for make in self.perturbations]
