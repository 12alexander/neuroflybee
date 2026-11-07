from pathlib import Path

import pytest

from neuroflybee.config import ProjectConfig, ScenarioConfig, from_dict, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_loads():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert isinstance(cfg, ProjectConfig)
    assert cfg.sensors.patch_size == 32
    assert cfg.scenario.n_waypoints == (3, 5)


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError):
        from_dict(ScenarioConfig, {"no_existe": 1})


def test_seeding_is_reproducible():
    import numpy as np

    from neuroflybee.utils.seeding import make_rng, set_global_seed

    set_global_seed(123)
    a = np.random.rand(3)
    set_global_seed(123)
    assert np.allclose(a, np.random.rand(3))
    assert np.allclose(make_rng(7).random(2), make_rng(7).random(2))
