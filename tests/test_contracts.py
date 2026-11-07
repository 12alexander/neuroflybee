"""Contratos de v18: viento y perturbaciones inyectables sin cambiar el núcleo."""
from dataclasses import replace

import numpy as np
import pytest

from neuroflybee.config import ProjectConfig
from neuroflybee.contracts import Conditions, conditions_registry
from neuroflybee.contracts.conformance import check_perturbation, check_wind_model
from neuroflybee.domain.drone import Action
from neuroflybee.sim.env import NavigationEnv
from neuroflybee.sim.scenarios import ScenarioGenerator
from neuroflybee.sim.wind import ScenarioWind

CFG = ProjectConfig()


class ConstantWind:
    def __init__(self, v):
        self.v = np.asarray(v, dtype=float)

    def reset(self, rng):
        pass

    def velocity(self, t, position, rng):
        return self.v.copy()


class BlindRays:
    """Rayos que fallan: todos devuelven el alcance máximo (no ven nada)."""
    name = "rayos_ciegos"

    def reset(self, rng):
        pass

    def apply(self, obs, rng):
        return replace(obs, ranges=np.full_like(obs.ranges, CFG.sensors.ray_range))


def _fly(env, n=15):
    for _ in range(n):
        env.step(Action.FORWARD)
    return env.state.position


def test_reference_implementations_satisfy_the_contracts():
    check_wind_model(lambda: ScenarioWind(np.array([0.5, -0.2])))
    check_wind_model(lambda: ConstantWind([0.3, 0.0]))
    check_perturbation(BlindRays)


def test_injected_wind_changes_the_flight():
    sc = ScenarioGenerator(CFG).generate("nominal", 20000)
    calm = _fly(NavigationEnv(CFG, sc))
    windy = _fly(NavigationEnv(CFG, sc, conditions=Conditions("viento", wind=lambda s: ConstantWind([0.0, 0.5]))))
    assert windy[1] - calm[1] == pytest.approx(15 * 0.5, rel=0.05)   # dt = 1 s: el viento se suma al desplazamiento


def test_default_conditions_reproduce_v17():
    sc = ScenarioGenerator(CFG).generate("wind", 20003)
    a = _fly(NavigationEnv(CFG, sc))
    b = _fly(NavigationEnv(CFG, sc, conditions=conditions_registry.get("escenario")))
    np.testing.assert_array_equal(a, b)


def test_sensor_perturbations_do_not_alter_the_gusts():
    """Flujo aleatorio separado: misma trayectoria con y sin ruido de sensores (con viento)."""
    sc = ScenarioGenerator(CFG).generate("wind", 20003)
    clean = NavigationEnv(CFG, sc)
    noisy = NavigationEnv(CFG, sc, conditions=Conditions("ruido", perturbations=(BlindRays,)))
    np.testing.assert_array_equal(_fly(clean), _fly(noisy))
    assert (noisy.last_observation.ranges == CFG.sensors.ray_range).all()


def test_registry():
    assert "escenario" in conditions_registry.available()
    c = conditions_registry.register(Conditions("prueba_registro", description="x"), replace=True)
    assert conditions_registry.get("prueba_registro") is c
    with pytest.raises(ValueError):
        conditions_registry.register(Conditions("prueba_registro"))
    with pytest.raises(KeyError):
        conditions_registry.get("no_existe")
