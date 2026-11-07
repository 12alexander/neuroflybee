import numpy as np
import pytest

from neuroflybee.config import DroneConfig
from neuroflybee.domain.drone import N_ACTIONS, Action, DroneState, Kinematics
from neuroflybee.domain.geometry import bearing, to_body_frame, wrap_angle
from neuroflybee.domain.mission import MissionTracker
from neuroflybee.domain.world import CircleObstacle, MovingObstacle, Obstacle, RectObstacle, World


@pytest.fixture
def world() -> World:
    return World(width=20, height=20, resolution=0.5, base=np.array([2.0, 2.0]),
                 waypoints=[np.array([15.0, 15.0])],
                 obstacles=[CircleObstacle(10, 10, 2), RectObstacle(4, 12, 6, 18, known=False)])


# --------------------------------------------------------------- geometría
@pytest.mark.parametrize("angle, expected", [(0.1, 0.1), (2 * np.pi + 0.1, 0.1), (-np.pi - 0.1, np.pi - 0.1)])
def test_wrap_angle(angle, expected):
    assert np.isclose(wrap_angle(angle), expected)


def test_wrap_angle_vectorized():
    out = wrap_angle(np.array([0.0, 3 * np.pi]))
    assert isinstance(out, np.ndarray) and np.all(out >= -np.pi) and np.all(out < np.pi)


def test_body_frame_and_bearing():
    assert np.allclose(to_body_frame(np.array([0.0, 1.0]), np.pi / 2), [1.0, 0.0])
    assert np.isclose(bearing(np.zeros(2), np.array([0.0, 5.0])), np.pi / 2)


# --------------------------------------------------------------- obstáculos
def test_obstacles_follow_protocol():
    for o in (CircleObstacle(0, 0, 1), RectObstacle(0, 0, 1, 1), MovingObstacle(0, 0, 1, 0, 0.5)):
        assert isinstance(o, Obstacle)


def test_signed_distances():
    c = CircleObstacle(0, 0, 1)
    assert np.isclose(c.signed_distance(np.array([[3.0, 0.0]]))[0], 2.0)
    r = RectObstacle(0, 0, 2, 2)
    assert r.signed_distance(np.array([[1.0, 1.0]]))[0] < 0          # dentro
    assert np.isclose(r.signed_distance(np.array([[4.0, 1.0]]))[0], 2.0)


def test_moving_obstacle_bounces():
    m = MovingObstacle(0, 0, 1, 0, 0.5, span=4)
    assert np.allclose(m.center(2), [2, 0])
    assert np.allclose(m.center(4), [4, 0])
    assert np.allclose(m.center(8), [0, 0])
    assert m.is_dynamic and not m.known


# --------------------------------------------------------------- mundo
def test_clearance_and_occupancy(world):
    assert world.clearance(np.array([10.0, 10.0]))[0] < 0                     # dentro de la roca
    assert world.clearance(np.array([5.0, 15.0]), known_only=True)[0] > 0      # muro no previsto
    assert world.clearance(np.array([5.0, 15.0]))[0] < 0
    occ_known, occ_all = world.occupancy(known_only=True), world.occupancy()
    assert occ_known.shape == world.shape == (40, 40)
    assert occ_all.sum() > occ_known.sum()
    assert occ_all[world.to_cell(np.array([10.0, 10.0]))]


def test_cell_roundtrip(world):
    p = np.array([7.3, 12.9])
    assert np.linalg.norm(world.to_point(world.to_cell(p)) - p) <= world.resolution


def test_dynamic_occupancy_changes_with_time():
    w = World(20, 20, 0.5, np.array([1.0, 1.0]), [], [MovingObstacle(5, 10, 1, 0, 1.0, span=8)])
    assert not np.array_equal(w.occupancy(t=0), w.occupancy(t=4))


# --------------------------------------------------------------- dron
def test_kinematics():
    k = Kinematics(DroneConfig())
    s = k.apply(DroneState(0, 0, 0), Action.FORWARD)
    assert np.isclose(s.x, 0.8) and s.y == 0
    assert k.apply(s, Action.HOVER) == s
    left = k.apply(DroneState(0, 0, 0), Action.LEFT)
    assert np.isclose(left.heading, np.radians(15))
    windy = k.apply(DroneState(0, 0, 0), Action.FORWARD, wind=np.array([0, 0.2]))
    assert np.isclose(windy.y, 0.2)
    assert N_ACTIONS == 6


# --------------------------------------------------------------- misión
def test_mission_tracker():
    m = MissionTracker([np.array([5.0, 5.0]), np.array([9.0, 9.0])], base=np.array([0.0, 0.0]))
    assert m.update(np.array([5.5, 5.0])) == [0]
    assert m.update(np.array([5.5, 5.0])) == []          # no se cuenta dos veces
    assert m.coverage == 0.5
    assert m.returned_home(np.array([1.0, 1.0]))
    assert not m.returned_home(np.array([3.0, 3.0]))
