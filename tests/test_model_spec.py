"""Pruebas de la especificación (no requieren PyTorch)."""
from pathlib import Path

import pytest

from neuroflybee.models.spec import (FLY_MODEL_NAMES, FlyNetConfig, count_parameters, describe,
                                     load_fly_config)

ROOT = Path(__file__).resolve().parents[1]


def test_yaml_matches_defaults():
    assert load_fly_config(ROOT / "configs" / "fly_models.yaml") == FlyNetConfig()


def test_parameter_counts_by_hand():
    c = FlyNetConfig()
    # CNN: conv(160+4 640+18 496) + BN(32+64+128) + Linear(1024→128) + objetivo(128) + cabeza(10 694)
    assert count_parameters("fly_cnn", c) == 165_542
    assert count_parameters("fly_cnn_gru", c) == 165_542 + 3 * 128 * (128 + 128 + 2)
    assert count_parameters("fly_cnn_transformer", c) > count_parameters("fly_cnn_gru", c)


@pytest.mark.parametrize("name", FLY_MODEL_NAMES)
def test_describe_blocks(name):
    d = describe(name, FlyNetConfig())
    assert d.blocks[0].kind == "input" and d.blocks[-1].kind == "output"
    assert sum(b.params for b in d.blocks) == count_parameters(name, FlyNetConfig())


def test_unknown_model():
    with pytest.raises(KeyError):
        count_parameters("abeja", FlyNetConfig())
