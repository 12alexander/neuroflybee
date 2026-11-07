"""Pruebas de conformidad reutilizables: cada implementación nueva las llama en su test.

    from neuroflybee.contracts.conformance import check_wind_model
    def test_dryden_cumple_el_contrato():
        check_wind_model(lambda: DrydenWind(...))
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .disturbances import Conditions, SensorPerturbation, WindModel


def check_wind_model(make: Callable[[], WindModel], steps: int = 50) -> None:
    """Forma (2,), valores finitos y reproducibilidad con la misma semilla."""
    def series(seed: int) -> np.ndarray:
        model, rng = make(), np.random.default_rng(seed)
        model.reset(rng)
        return np.array([model.velocity(t, np.array([10.0, 10.0]), rng) for t in range(steps)])
    a, b = series(7), series(7)
    assert isinstance(make(), WindModel), "no implementa WindModel (reset, velocity)"
    assert a.shape == (steps, 2), f"velocity debe devolver (2,), devolvió {a.shape[1:]}"
    assert np.isfinite(a).all(), "valores no finitos"
    np.testing.assert_allclose(a, b, err_msg="con la misma semilla debe dar la misma serie")


def check_perturbation(make: Callable[[], SensorPerturbation], family: str = "nominal", seed: int = 20000) -> None:
    """Se puede inyectar en el entorno, no muta la observación limpia y respeta formas."""
    from ..config import ProjectConfig
    from ..domain.drone import Action
    from ..sim.env import NavigationEnv
    from ..sim.scenarios import ScenarioGenerator

    p = make()
    assert isinstance(p, SensorPerturbation), "no implementa SensorPerturbation (name, reset, apply)"
    cfg = ProjectConfig()
    scenario = ScenarioGenerator(cfg).generate(family, seed)
    clean = NavigationEnv(cfg, scenario)
    dirty = NavigationEnv(cfg, scenario, conditions=Conditions("prueba", perturbations=(make,)))
    for _ in range(5):
        a, b = clean.step(Action.FORWARD).observation, dirty.step(Action.FORWARD).observation
        for field in ("fly_patch", "ranges", "panorama"):
            assert getattr(a, field).shape == getattr(b, field).shape, f"cambió la forma de {field}"
    np.testing.assert_allclose(clean.state.position, dirty.state.position,
                               err_msg="una perturbación de sensores no debe mover al dron")
