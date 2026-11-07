import numpy as np

from neuroflybee.config import ProjectConfig
from neuroflybee.control.base import Decision
from neuroflybee.domain.drone import Action
from neuroflybee.sim.env import NavigationEnv
from neuroflybee.sim.rollout import rollout
from neuroflybee.sim.scenarios import ScenarioGenerator


class ForwardOnly:
    name = "forward"

    def reset(self, obs):
        pass

    def act(self, obs):
        return Decision(Action.FORWARD)


def make_env(family="nominal", seed=20_000):
    cfg = ProjectConfig()
    return NavigationEnv(cfg, ScenarioGenerator(cfg).generate(family, seed))


def test_observation_has_no_true_position():
    obs = make_env().reset()
    assert obs.fly_patch.shape == (32, 32) and obs.panorama.shape == (72,)
    assert obs.gps is not None and obs.home_snapshot.shape == (72,)
    assert not hasattr(obs, "true_state")


def test_forward_controller_ends_with_collision_or_timeout():
    env = make_env()
    traj = rollout(env, ForwardOnly())
    assert traj.collided or len(traj.steps) - 1 == env.scenario.max_steps
    assert env.done


def test_reset_is_reproducible():
    a, b = make_env(), make_env()
    pa = rollout(a, ForwardOnly()).positions
    pb = rollout(b, ForwardOnly()).positions
    assert np.allclose(pa, pb)


def test_gps_lost_after_all_waypoints_in_gps_loss_family():
    env = make_env("gps_loss")
    assert env.gps_available
    env.tracker.visited = [True] * len(env.tracker.visited)
    assert not env.gps_available


def test_trajectory_records_estimates():
    traj = rollout(make_env("nominal"), ForwardOnly())
    assert traj.estimates.shape == traj.positions.shape
    assert traj.gps_lost_from is None          # en la condición nominal nunca falta el GPS
