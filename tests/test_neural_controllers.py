"""Controladores neuronales con pesos sin entrenar (sólo verifica el cableado; requiere PyTorch)."""
import pytest

pytest.importorskip("torch")

from neuroflybee.config import ProjectConfig  # noqa: E402
from neuroflybee.control import registry  # noqa: E402
from neuroflybee.control.neural_registry import register_neural_controllers  # noqa: E402
from neuroflybee.models.bee import BeeHomingNet  # noqa: E402
from neuroflybee.models.fly import build_fly_model  # noqa: E402
from neuroflybee.sim.env import NavigationEnv  # noqa: E402
from neuroflybee.sim.rollout import rollout  # noqa: E402
from neuroflybee.sim.scenarios import ScenarioGenerator  # noqa: E402
from neuroflybee.training.bee import save_bee_checkpoint  # noqa: E402
from neuroflybee.training.checkpoints import save_fly_checkpoint  # noqa: E402


def test_all_neural_controllers_fly_an_episode(tmp_path):
    save_fly_checkpoint(tmp_path / "fly_cnn_gru_s0.pt", "fly_cnn_gru", build_fly_model("fly_cnn_gru"), 0, {}, 0)
    save_bee_checkpoint(tmp_path / "bee_s0.pt", BeeHomingNet(), 0, {}, 0)
    names = register_neural_controllers(tmp_path)
    assert set(names) == {"fly", "bee", "hybrid", "hybrid_no_supervisor"}
    cfg = ProjectConfig()
    sc = ScenarioGenerator(cfg).generate("gps_loss", 20_000)
    sc = type(sc)(sc.family, sc.seed, sc.world, sc.gps_loss_on_return, 40)      # episodios cortos
    for name in names:
        traj = rollout(NavigationEnv(cfg, sc), registry.create(name))
        assert len(traj.steps) > 1
        assert any(s.decision.probs is not None for s in traj.steps[1:]) or name == "bee"
