"""Servicio de simulación (capa de aplicación).

No sabe nada de HTTP: recibe parámetros validados, simula y devuelve datos en el mismo
formato que el visor HTML. Así el mismo servicio podría exponerse por otro adaptador
(FastAPI, línea de comandos, el dron real) sin cambiar esta clase.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

from ..config import ProjectConfig
from ..control import registry
from ..control.base import Controller
from ..sim.scenarios import FAMILIES, FAMILY_LABELS_ES, SPLIT_RANGES, ScenarioGenerator, Split
from ..viz.export import CONTROLLER_STYLE, build_viewer_data


class ValidationError(ValueError):
    """Parámetros inválidos enviados por el usuario (se responden con 400)."""


@dataclass(frozen=True)
class SimulationRequest:
    family: str
    seed: int
    controllers: tuple[str, ...]


class SimulationService:
    MAX_CONTROLLERS = 4

    def __init__(self, cfg: ProjectConfig, make_controller: Callable[[str], Controller] = registry.create,
                 available: Callable[[], Sequence[str]] = registry.available):
        self.cfg = cfg
        self.make_controller = make_controller
        self.available = available
        self.generator = ScenarioGenerator(cfg)
        self.seed_range = SPLIT_RANGES[Split.TEST]

    # ----------------------------------------------------------------- consultas
    def catalog(self) -> dict:
        names = list(self.available())
        return {
            "families": [{"id": f, "label": FAMILY_LABELS_ES[f]} for f in FAMILIES],
            "controllers": [{"id": n, "label": CONTROLLER_STYLE.get(n, (n, "#ccc"))[0],
                             "color": CONTROLLER_STYLE.get(n, (n, "#ccc"))[1],
                             "description": registry.describe().get(n, "")} for n in names],
            "seed_min": self.seed_range.start, "seed_max": self.seed_range.stop - 1,
            "max_controllers": self.MAX_CONTROLLERS,
        }

    # ----------------------------------------------------------------- validación
    def parse(self, family: str, seed: str | int, controllers: str | Sequence[str]) -> SimulationRequest:
        if family not in FAMILIES:
            raise ValidationError(f"Condición desconocida '{family}'. Opciones: {list(FAMILIES)}")
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValidationError("La semilla debe ser un número entero") from None
        if seed not in self.seed_range:
            raise ValidationError(f"Use una semilla de PRUEBA entre {self.seed_range.start} y "
                                  f"{self.seed_range.stop - 1} (las demás se usan para entrenar)")
        names = [c for c in (controllers.split(",") if isinstance(controllers, str) else controllers) if c]
        if not 1 <= len(names) <= self.MAX_CONTROLLERS:
            raise ValidationError(f"Elija entre 1 y {self.MAX_CONTROLLERS} controladores")
        unknown = [c for c in names if c not in self.available()]
        if unknown:
            raise ValidationError(f"Controladores no disponibles: {unknown}")
        return SimulationRequest(family, seed, tuple(dict.fromkeys(names)))

    # ----------------------------------------------------------------- simulación
    def simulate(self, request: SimulationRequest) -> dict:
        t0 = time.perf_counter()
        scenario = self.generator.generate(request.family, request.seed)
        data = build_viewer_data(self.cfg, [scenario], request.controllers, self.make_controller)
        data["meta"]["elapsed_s"] = round(time.perf_counter() - t0, 2)
        return data
