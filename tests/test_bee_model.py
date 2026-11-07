"""Abeja: especificación, métricas y figura sin PyTorch; red y entrenamiento con PyTorch."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from neuroflybee.evaluation.metrics import angular_errors_deg, homing_report
from neuroflybee.models.bee_spec import BeeNetConfig, count_bee_parameters, load_bee_config

ROOT = Path(__file__).resolve().parents[1]


def test_bee_config_and_param_count():
    assert load_bee_config(ROOT / "configs" / "bee_model.yaml") == BeeNetConfig()
    assert count_bee_parameters(BeeNetConfig()) == 172_547


def test_angular_metrics():
    assert np.allclose(angular_errors_deg(np.radians([10, 170]), np.radians([-10, -170])), [20, 20])
    t = np.array([[1.0, 0.0, np.log1p(4.0)]])
    rep = homing_report(t.copy(), t)
    assert rep["angle_error_median"] == 0 and rep["distance_mae"] == pytest.approx(0)


def test_v11_figure_without_network(tmp_path):
    spec = importlib.util.spec_from_file_location("s", ROOT / "scripts" / "show_bee_homing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--data", str(tmp_path / "sin_datos"), "--checkpoints", str(tmp_path / "sin_pesos"),
              "--results", str(tmp_path)])
    assert (tmp_path / "figures" / "v11_retorno_abeja.png").stat().st_size > 10_000
    assert (tmp_path / "v11_abeja_validacion.csv").read_text(encoding="utf-8").count("\n") == 2


def test_bee_network_with_torch(tmp_path):
    torch = pytest.importorskip("torch")
    from neuroflybee.config import ProjectConfig
    from neuroflybee.data.generate import collect_bee_samples
    from neuroflybee.inference.bee_homing import BeeNetHoming
    from neuroflybee.models.bee import BeeHomingLoss, BeeHomingNet
    from neuroflybee.sim.env import NavigationEnv
    from neuroflybee.sim.scenarios import ScenarioGenerator
    from neuroflybee.training.bee import load_bee_checkpoint, train_bee_model
    from neuroflybee.training.config import TrainConfig

    net = BeeHomingNet()
    assert sum(p.numel() for p in net.parameters()) == count_bee_parameters(BeeNetConfig())
    out = net(torch.rand(4, 72), torch.rand(4, 72))
    assert out.shape == (4, 3)
    target = torch.tensor([[1.0, 0.0, 1.0]] * 4)
    assert BeeHomingLoss()(target.clone(), target) == pytest.approx(0.0, abs=1e-6)

    cfg = ProjectConfig()
    data = collect_bee_samples(cfg, "train", 4, per_scenario=32)
    result = train_bee_model(data, data, BeeNetConfig(), TrainConfig(epochs=2, batch_size=32, num_workers=0,
                             device="cpu", patience=5), 0, tmp_path / "ck", tmp_path)
    assert len(result.history) == 2
    model, _ = load_bee_checkpoint(tmp_path / "ck" / "bee_s0.pt")
    obs = NavigationEnv(cfg, ScenarioGenerator(cfg).generate("nominal", 10_000)).reset()
    est = BeeNetHoming(model).estimate(obs)
    assert -np.pi <= est.direction <= np.pi and est.distance >= 0
