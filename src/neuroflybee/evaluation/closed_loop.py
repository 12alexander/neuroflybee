"""Evaluación en lazo cerrado: el controlador vuela episodios completos (sin PyTorch;
cualquier `Controller` sirve). La reutiliza el benchmark de la v13."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from ..config import ProjectConfig
from ..contracts.disturbances import Conditions
from ..control.base import Controller
from ..sim.env import NavigationEnv
from ..sim.rollout import rollout
from ..sim.scenarios import ScenarioGenerator


@dataclass(frozen=True)
class EpisodeResult:
    controller: str
    family: str
    seed: int
    success: bool
    collided: bool
    landed: bool
    coverage: float
    home_error: float
    steps: int
    path_length: float
    mean_latency_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def run_episode(cfg: ProjectConfig, controller: Controller, family: str, seed: int,
                generator: ScenarioGenerator | None = None,
                conditions: Conditions | None = None) -> EpisodeResult:
    gen = generator or ScenarioGenerator(cfg)
    env = NavigationEnv(cfg, gen.generate(family, seed), conditions=conditions)
    traj = rollout(env, controller, keep_debug=False)
    pos = traj.positions
    latency = [s.latency_ms for s in traj.steps[1:]]
    return EpisodeResult(controller.name, family, seed,
                         env.tracker.success(env.state.position, traj.collided, traj.landed),
                         traj.collided, traj.landed, env.tracker.coverage,
                         env.tracker.home_error(env.state.position), len(traj.steps) - 1,
                         float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
                         float(np.mean(latency)) if latency else 0.0)


def evaluate(cfg: ProjectConfig, make_controller: Callable[[], Controller], families: Sequence[str],
             seeds: Iterable[int], conditions: Conditions | None = None) -> list[EpisodeResult]:
    gen, seeds = ScenarioGenerator(cfg), list(seeds)
    return [run_episode(cfg, make_controller(), family, seed, gen, conditions)
            for family in families for seed in seeds]


def summarize(results: Sequence[EpisodeResult]) -> dict:
    if not results:
        return {}
    arr = lambda k: np.array([getattr(r, k) for r in results], dtype=float)  # noqa: E731
    return {"episodes": len(results), "success": float(arr("success").mean()),
            "collision": float(arr("collided").mean()), "coverage": float(arr("coverage").mean()),
            "home_error_median": float(np.median(arr("home_error"))), "steps_mean": float(arr("steps").mean())}
