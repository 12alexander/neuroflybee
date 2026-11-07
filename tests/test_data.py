import numpy as np
import pytest

from neuroflybee.config import ProjectConfig
from neuroflybee.control.navigation import MissionNavigator
from neuroflybee.data.expert import PrivilegedExpert
from neuroflybee.data.generate import (BeeDataset, FlyDataset, collect_bee_samples, collect_fly_episodes,
                                       goal_features)
from neuroflybee.domain.drone import N_ACTIONS
from neuroflybee.sim.env import NavigationEnv
from neuroflybee.sim.scenarios import ScenarioGenerator


@pytest.fixture(scope="module")
def cfg():
    return ProjectConfig()


@pytest.mark.parametrize("family", ["nominal", "dynamic"])
def test_expert_completes_missions(cfg, family):
    env = NavigationEnv(cfg, ScenarioGenerator(cfg).generate(family, 10_001))
    obs = env.reset()
    expert, nav = PrivilegedExpert(env.world, cfg), MissionNavigator()
    nav.reset(obs)
    done = False
    while not done:
        target = nav.update(obs, env.state.position)
        result = env.step(expert.act(env.state, target.goal, env.t), target.land)
        obs, done = result.observation, result.done
    assert env.tracker.success(env.state.position, env.collided, env.landed)


def test_fly_dataset_shapes_and_roundtrip(cfg, tmp_path):
    ds = collect_fly_episodes(cfg, "val", 2, max_steps=60)
    assert ds.patches.dtype == np.uint8 and ds.patches.shape[1:] == (32, 32)
    assert ds.goals.shape == (len(ds), 3) and ds.actions.max() < N_ACTIONS
    assert set(np.unique(ds.episode)) == {0, 1}
    ds.save(tmp_path / "fly.npz")
    again = FlyDataset.load(tmp_path / "fly.npz")
    assert np.array_equal(again.patches, ds.patches) and np.array_equal(again.family, ds.family)
    both = FlyDataset.concat([ds, again])
    assert len(both) == 2 * len(ds) and both.episode.max() == 3


def test_bee_samples(cfg, tmp_path):
    ds = collect_bee_samples(cfg, "val", 2, per_scenario=10)
    assert len(ds) == 20 and ds.current.shape == ds.home.shape == (20, 72)
    assert np.allclose(np.linalg.norm(ds.target[:, :2], axis=1), 1.0, atol=1e-5)
    assert np.all(ds.target[:, 2] >= 0)
    ds.save(tmp_path / "bee.npz")
    assert np.array_equal(BeeDataset.load(tmp_path / "bee.npz").target, ds.target)


def test_goal_features_are_in_body_frame(cfg):
    obs = NavigationEnv(cfg, ScenarioGenerator(cfg).generate("nominal", 10_000)).reset()
    obs.heading = 0.0
    g = goal_features(obs, np.array([5.0, 0.0]), np.zeros(2))
    assert np.allclose(g, [1.0, 0.0, 0.5], atol=1e-6)        # 5 m al frente
