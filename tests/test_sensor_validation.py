"""v21: validación del sensor panorámico 2,5D (IDF, rumbo, área de captura, figura y números)."""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neuroflybee.config import ProjectConfig
from neuroflybee.evaluation.sensor_validation import (SensorValidationConfig, arrival_radius, heading_error_deg,
                                                      load_sensor_validation_config, rms_difference,
                                                      rotational_idf, run_validation, spearman_rho,
                                                      summarize_validation, write_outputs)
from neuroflybee.reporting.numbers import NumberSet, sensor_validation

ROOT = Path(__file__).resolve().parents[1]
SMALL = SensorValidationConfig(scenarios=2, radii=(0.0, 1.0, 3.0, 6.0), directions=8, noise_draws=2,
                               rotation_radii=(2.0,), field_step=3.0, field_radius=6.0)


def test_config_file_matches_defaults():
    assert load_sensor_validation_config(ROOT / "configs" / "sensor_validation.yaml") == SensorValidationConfig()


def test_primitives():
    a = np.array([0.0, 1.0, 0.0, 0.0])
    assert rms_difference(a, a) == 0 and rms_difference(a, np.zeros(4)) == pytest.approx(0.5)
    view = np.random.default_rng(0).random(72)
    curve = rotational_idf(view, view)
    assert curve.shape == (72,) and int(np.argmin(curve)) == 0
    assert heading_error_deg(np.roll(view, 10), view, 10) == 0          # recupera el giro aplicado
    assert heading_error_deg(np.roll(view, 10), view, 12) == 10         # 2 sectores de 5°
    assert spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert np.isnan(spearman_rho([1, 2], [1, 2]))
    radii = np.array([0.0, 1.0, 2.0])
    assert arrival_radius(radii, np.array([0.0, 0.02, 0.04]), 0.03) == pytest.approx(1.5)
    assert arrival_radius(radii, np.array([0.0, 0.01, 0.02]), 0.03) == float("inf")


def test_sensor_behaves_like_real_panoramas():
    """Propiedades que deben cumplirse para que el retorno visual tenga sentido."""
    rows = run_validation(ProjectConfig(), SMALL)
    summary = summarize_validation(rows, SMALL.arrive_rms)
    for r in rows:
        assert r.idf_clean[0] == 0                                     # en casa la vista coincide
        assert r.idf_clean[-1] > r.idf_clean[1]                        # lejos difiere más que cerca
    assert summary.spearman_mean > 0.8
    assert summary.heading_error["2"] <= 10                             # el horizonte fija el rumbo
    assert 0.5 < summary.alv_catchment_mean <= 1


def test_outputs_figure_and_numbers(tmp_path):
    rows = run_validation(ProjectConfig(), SMALL)
    summary = summarize_validation(rows, SMALL.arrive_rms)
    write_outputs(rows, summary, tmp_path)
    assert (tmp_path / "v21_idf.csv").read_text(encoding="utf-8").count("\n") == 1 + len(SMALL.radii)
    assert json.loads((tmp_path / "v21_validacion_resumen.json").read_text(encoding="utf-8"))["scenarios"] == 2

    ns = NumberSet()
    sensor_validation(ns, tmp_path / "vacio")
    assert ns.is_pending("IdfSpearman")
    ns = NumberSet()
    sensor_validation(ns, tmp_path)
    assert not ns.is_pending("IdfSpearman") and ns.values["IdfEscenarios"][0] == "2"
    assert ns.is_pending("RumboErrorCuatro")                           # no se midió a 4 m


def test_script(tmp_path):
    spec = importlib.util.spec_from_file_location("vs", ROOT / "scripts" / "validate_sensor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--scenarios", "2", "--results", str(tmp_path)])
    assert (tmp_path / "figures" / "v21_validacion_sensor.png").stat().st_size > 50_000
    assert (tmp_path / "v21_validacion_sensor.csv").exists()


def test_noise_free_sensor_has_zero_floor():
    cfg = ProjectConfig()
    cfg = replace(cfg, sensors=replace(cfg.sensors, panorama_noise=0.0))
    rows = run_validation(cfg, SMALL)
    assert summarize_validation(rows, SMALL.arrive_rms).noise_floor == 0
