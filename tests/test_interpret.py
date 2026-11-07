"""Análisis de errores (sin PyTorch) e interpretabilidad (Grad-CAM / IG con PyTorch)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from neuroflybee.config import ProjectConfig
from neuroflybee.control import classical  # noqa: F401
from neuroflybee.control import registry
from neuroflybee.evaluation.error_analysis import CATEGORIES, category_shares, diagnose

ROOT = Path(__file__).resolve().parents[1]


def test_direct_controller_failures_are_collisions():
    diags = diagnose(ProjectConfig(), lambda: registry.create("direct"), ["unknown_obstacles"], [20_000, 20_001])
    assert all(d.category.startswith("choque") for d in diags)
    shares = category_shares(diags)
    assert set(shares) == set(CATEGORIES) and sum(shares.values()) == pytest.approx(1.0)


def test_gps_loss_failure_is_drift():
    diags = diagnose(ProjectConfig(), lambda: registry.create("astar"), ["gps_loss"], [20_000])
    assert diags[0].category == "deriva sin GPS (aterriza lejos)"


def test_explain_script_without_torch_weights(tmp_path):
    spec = importlib.util.spec_from_file_location("e", ROOT / "scripts" / "explain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--episodes", "1", "--results", str(tmp_path), "--checkpoints", str(tmp_path / "sin_pesos")])
    assert (tmp_path / "figures" / "v15_analisis_errores.png").stat().st_size > 10_000
    assert (tmp_path / "v15_errores.csv").read_text(encoding="utf-8").count("\n") == 1 + 4 * 6


def test_gradcam_and_integrated_gradients_with_torch():
    torch = pytest.importorskip("torch")
    from neuroflybee.interpret.gradcam import grad_cam
    from neuroflybee.interpret.integrated_gradients import deletion_test, integrated_gradients
    from neuroflybee.models.bee import BeeHomingNet
    from neuroflybee.models.fly import build_fly_model

    torch.manual_seed(0)
    frames = (torch.rand(4, 32, 32) > 0.8).float()
    for name in ("fly_cnn", "fly_cnn_gru", "fly_cnn_transformer"):
        cam, action, probs = grad_cam(build_fly_model(name), frames, torch.randn(3))
        assert cam.shape == (4, 32, 32) and float(cam.min()) >= 0 and float(cam.max()) <= 1 + 1e-6
        assert 0 <= action < 6 and probs.sum() == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(TypeError):
        grad_cam(build_fly_model("fly_mlp"), frames, torch.randn(3))

    bee = BeeHomingNet().eval()
    cur, home = torch.rand(72), torch.rand(72)
    attr, fx, fb = integrated_gradients(bee, cur, home, steps=256)
    assert attr.shape == (72,)
    assert attr.sum() == pytest.approx(fx - fb, rel=0.05, abs=1e-3)       # axioma de completitud
    top, rand = deletion_test(bee, cur.numpy(), home.numpy(), attr, np.array([0.0, 0.5]), np.random.default_rng(0))
    assert top[0] == pytest.approx(0.0) and top.shape == rand.shape == (2,)
