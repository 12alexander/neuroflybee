import numpy as np

from neuroflybee.config import SensorConfig
from neuroflybee.domain.drone import DroneState
from neuroflybee.domain.world import CircleObstacle, Landmark, World
from neuroflybee.sim.sensors import FlyEyeSensor, OdometrySensor, PanoramaSensor, RangeSensor


def world():
    return World(30, 30, 0.5, np.array([12.0, 15.0]), [np.array([25.0, 15.0])],
                 obstacles=[CircleObstacle(17, 15, 1.0, height=4)],
                 landmarks=[Landmark(80, 15, 10, 30)])


def test_fly_eye_sees_obstacle_ahead():
    w, cfg = world(), SensorConfig(pixel_noise=0)
    patch = FlyEyeSensor(cfg).read(w.occupancy(), w, DroneState(12, 15, 0.0))
    assert patch.shape == (32, 32)
    # obstáculo a 5 m al frente: filas superiores-centrales ocupadas, columnas laterales no
    assert patch[: 20, 14:18].sum() > 0
    assert patch[:, :4].sum() == 0
    # girado 180°: el obstáculo queda detrás y fuera de la vista frontal
    back = FlyEyeSensor(cfg).read(w.occupancy(), w, DroneState(12, 15, np.pi))
    assert back[:20, 14:18].sum() == 0


def test_range_sensor():
    w = world()
    rs = RangeSensor(SensorConfig())
    r = rs.read(w.occupancy(), w, DroneState(12, 15, 0.0))
    front = r[np.argmin(np.abs(rs.angles))]
    assert 3.5 <= front <= 4.5


def test_panorama_points_to_landmarks():
    w, ps = world(), PanoramaSensor(SensorConfig())
    view = ps.read(w, DroneState(12, 15, 1.0))
    east = np.argmin(np.abs(ps.bin_centers))
    assert view[east] > 0.2 and view.max() <= 1
    # la vista no depende del rumbo (está alineada con la brújula)
    assert np.allclose(view, ps.read(w, DroneState(12, 15, -2.0)))


def test_odometry_ignores_wind():
    rng = np.random.default_rng(0)
    m = OdometrySensor(0.0).read(np.array([1.0, 0.0]), rng)
    assert np.allclose(m, [1, 0], atol=0.05)


def test_fly_eye_footprint_and_ray_endpoints():
    cfg = SensorConfig()
    corners = FlyEyeSensor(cfg).footprint(DroneState(12, 15, 0.0))
    assert np.allclose(corners[0], [12 + cfg.patch_forward, 15 + cfg.patch_half_width])
    rs = RangeSensor(cfg)
    ends = rs.end_points(np.full(cfg.n_rays, 2.0, dtype=np.float32), DroneState(12, 15, 0.0))
    assert np.allclose(np.linalg.norm(ends - [12, 15], axis=1), 2.0)


def test_panorama_changes_with_distance_from_home():
    w, ps = world(), PanoramaSensor(SensorConfig())
    objs = ps.objects(w)
    home = ps.view_from(objs, w.base)
    near = np.abs(ps.view_from(objs, w.base + [0.5, 0]) - home).mean()
    far = np.abs(ps.view_from(objs, w.base + [0, 6.0]) - home).mean()
    assert far > near
