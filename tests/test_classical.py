import numpy as np
import pytest

from neuroflybee.config import ProjectConfig
from neuroflybee.control import classical  # noqa: F401  (registra controladores)
from neuroflybee.control import registry
from neuroflybee.control.steering import dilate, heading_to_action
from neuroflybee.domain.drone import Action
from neuroflybee.sim.env import NavigationEnv
from neuroflybee.sim.rollout import rollout
from neuroflybee.sim.scenarios import ScenarioGenerator


def test_heading_to_action():
    assert heading_to_action(0.0) == Action.FORWARD
    assert heading_to_action(np.radians(20)) == Action.LEFT
    assert heading_to_action(np.radians(-90)) == Action.SHARP_RIGHT


def test_dilate():
    m = np.zeros((7, 7), bool)
    m[3, 3] = True
    assert dilate(m, 1).sum() == 5 and dilate(m, 2)[3, 1]


def test_registry_lists_baselines():
    assert {"direct", "potential_field", "astar"} <= set(registry.available())


@pytest.mark.parametrize("seed", [20_000, 20_001, 20_002])
def test_astar_completes_nominal_missions(seed):
    cfg = ProjectConfig()
    env = NavigationEnv(cfg, ScenarioGenerator(cfg).generate("nominal", seed))
    traj = rollout(env, registry.create("astar"))
    assert not traj.collided and traj.landed
    assert env.tracker.coverage == 1.0


def test_alv_points_roughly_home():
    from neuroflybee.control.homing import AverageLandmarkVectorHoming
    from neuroflybee.domain.geometry import wrap_angle
    from neuroflybee.sim.sensors import PanoramaSensor
    cfg = ProjectConfig()
    ps = PanoramaSensor(cfg.sensors)
    errs = []
    for s in range(20_000, 20_010):
        w = ScenarioGenerator(cfg).generate("nominal", s).world
        objs = ps.objects(w)
        home = ps.view_from(objs, w.base)
        for a in np.linspace(-np.pi, np.pi, 8, endpoint=False):
            v = ps.view_from(objs, w.base + 5 * np.array([np.cos(a), np.sin(a)]))
            d = AverageLandmarkVectorHoming.alv(home) - AverageLandmarkVectorHoming.alv(v)
            errs.append(abs(wrap_angle(np.arctan2(d[1], d[0]) - (a + np.pi))))
    assert np.median(errs) < np.radians(90)   # mejor que el azar (90°)


def test_supervisor_vetoes_unsafe_forward_and_never_hits_static_obstacles():
    from neuroflybee.control.safety import SafetySupervisor
    cfg = ProjectConfig()
    crashes = 0
    for seed in (20_000, 20_001, 20_002):
        env = NavigationEnv(cfg, ScenarioGenerator(cfg).generate("unknown_obstacles", seed))
        traj = rollout(env, registry.create("direct_supervised"))
        crashes += traj.collided
    assert crashes == 0                                   # sin supervisor, "direct" choca en las 3


def test_supervisor_is_a_transparent_decorator():
    from neuroflybee.control.safety import SafetySupervisor
    cfg = ProjectConfig()
    sc = ScenarioGenerator(cfg).generate("nominal", 20_000)
    a = rollout(NavigationEnv(cfg, sc), registry.create("astar")).positions
    b = rollout(NavigationEnv(cfg, sc), SafetySupervisor(registry.create("astar"))).positions
    assert len(a) > 10 and np.allclose(a[:10], b[:10])   # si no hay peligro, no cambia nada


def test_neural_registry_without_weights(tmp_path):
    from neuroflybee.control.neural_registry import find_fly_checkpoint, register_neural_controllers
    for name in ("fly_cnn_gru_s0.pt", "fly_cnn_gru_dagger1_s0.pt", "fly_cnn_gru_dagger3_s0.pt"):
        (tmp_path / name).write_text("x")
    assert find_fly_checkpoint(tmp_path).name == "fly_cnn_gru_dagger3_s0.pt"
    assert register_neural_controllers(tmp_path / "vacio") == []
