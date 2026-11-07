"""Entorno de navegación: integra mundo, cinemática, perturbaciones y sensores."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import ProjectConfig
from ..domain.drone import Action, DroneState, Kinematics
from ..domain.geometry import bearing
from ..domain.mission import MissionTracker
from .observation import Observation, PriorMap
from .scenarios import Scenario
from .wind import ScenarioWind
from ..contracts.disturbances import Conditions
from .sensors import (CompassSensor, FlyEyeSensor, GpsSensor, OdometrySensor, PanoramaSensor,
                      RangeSensor)


@dataclass
class StepInfo:
    t: int
    true_state: DroneState
    collision: bool
    gps_available: bool
    newly_visited: list[int] = field(default_factory=list)
    clearance: float = np.inf


@dataclass
class StepResult:
    observation: Observation
    done: bool
    info: StepInfo


class SensorSuite:
    """Agrupa los sensores (composición) para que el entorno no conozca sus detalles."""

    def __init__(self, cfg: ProjectConfig):
        s = cfg.sensors
        self.fly = FlyEyeSensor(s)
        self.ranges = RangeSensor(s)
        self.panorama = PanoramaSensor(s)
        self.gps = GpsSensor(s.gps_sigma)
        self.compass = CompassSensor()
        self.odometry = OdometrySensor(s.odometry_sigma)


class NavigationEnv:
    def __init__(self, cfg: ProjectConfig, scenario: Scenario, seed: int = 0,
                 sensor_failure: float = 0.0, conditions: Optional[Conditions] = None):
        self.cfg = cfg
        self.conditions = conditions or Conditions()
        # Inversión de dependencias (v18): el viento y las fallas de sensores se inyectan.
        self.wind_model = self.conditions.build_wind(scenario) or ScenarioWind(scenario.world.wind)
        self.perturbations = self.conditions.build_perturbations()
        self.scenario = scenario
        self.world = scenario.world
        self.kin = Kinematics(cfg.drone)
        self.sensors = SensorSuite(cfg)
        self.sensor_failure = sensor_failure
        self._seed = seed
        self._panorama_objects = self.sensors.panorama.objects(self.world)
        prior = self.world.occupancy(known_only=True)
        self.prior_map = PriorMap(prior, self.world.resolution, self.world.width, self.world.height)
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self) -> Observation:
        self.rng = np.random.default_rng([self._seed, self.scenario.seed])
        self.wind_model.reset(self.rng)
        self.last_wind = np.zeros(2)
        # Flujo aleatorio propio para las perturbaciones: agregar ruido de sensores no altera
        # las ráfagas ni el resto del episodio (comparación controlada, v18).
        self.perturbation_rng = np.random.default_rng([self._seed, self.scenario.seed, 1])
        for p in self.perturbations:
            p.reset(self.perturbation_rng)
        w = self.world
        first = w.waypoints[0] if w.waypoints else w.base + np.array([1.0, 0.0])
        self.state = DroneState(w.base[0], w.base[1], bearing(w.base, first))
        self.t = 0
        self.done = False
        self.tracker = MissionTracker(w.waypoints, w.base)
        self.collided = False
        self.landed = False
        self._last_odo = np.zeros(2)
        self.home_snapshot = self.sensors.panorama.view_from(self._panorama_objects, w.base)
        self.trajectory: list[np.ndarray] = [self.state.position]
        self.last_observation = self._observe()
        return self.last_observation

    @property
    def gps_available(self) -> bool:
        return not (self.scenario.gps_loss_on_return and all(self.tracker.visited))

    def step(self, action: Action, land: bool = False) -> StepResult:
        if self.done:
            raise RuntimeError("El episodio terminó; llame a reset()")
        commanded, _ = self.kin.commanded_displacement(self.state, action)
        wind = self.wind_model.velocity(self.t, self.state.position, self.rng)
        self.last_wind = wind
        self.state = self.kin.apply(self.state, action, wind)
        self._last_odo = self.sensors.odometry.read(commanded, self.rng)
        self.t += 1
        self.trajectory.append(self.state.position)

        clearance = float(self.world.clearance(self.state.position, self.t)[0])
        collision = clearance < self.cfg.drone.radius
        newly = self.tracker.update(self.state.position)
        self.collided |= collision
        self.landed = bool(land) and not collision
        self.done = collision or self.landed or self.t >= self.scenario.max_steps
        info = StepInfo(self.t, self.state, collision, self.gps_available, newly, clearance)
        self.last_observation = self._observe()
        return StepResult(self.last_observation, self.done, info)

    # ------------------------------------------------------------- internos
    def _observe(self) -> Observation:
        occ = self.world.occupancy(self.t)
        s = self.sensors
        obs = Observation(
            t=self.t,
            fly_patch=s.fly.read(occ, self.world, self.state, self.rng, self.sensor_failure),
            ranges=s.ranges.read(occ, self.world, self.state),
            panorama=np.clip(s.panorama.view_from(self._panorama_objects, self.state.position)
                             + self.rng.normal(0, s.panorama.noise, self.cfg.sensors.panorama_bins), 0, 1
                             ).astype(np.float32),
            heading=s.compass.read(self.state, self.rng),
            odometry=self._last_odo.copy(),
            gps=s.gps.read(self.state, self.gps_available, self.rng),
            base=self.world.base.copy(),
            waypoints=[w.copy() for w in self.world.waypoints],
            home_snapshot=self.home_snapshot,
            prior_map=self.prior_map,
        )
        for perturbation in self.perturbations:
            obs = perturbation.apply(obs, self.perturbation_rng)
        return obs
