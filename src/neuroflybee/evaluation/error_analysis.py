"""Análisis sistemático de errores (R5), sin PyTorch.

Cada episodio fallido se clasifica por su CAUSA observable en la simulación:

* choque con obstáculo del mapa previo / no previsto / móvil / borde del área
* deriva sin GPS (aterriza lejos de la base tras perder el GPS)
* atascado (se acaba el tiempo sin completar la ruta)
* no encuentra la base (completa la ruta pero no logra aterrizar)
* aterriza sin inspeccionar todo
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from ..config import ProjectConfig
from ..control.base import Controller
from ..sim.env import NavigationEnv
from ..sim.rollout import Trajectory, rollout
from ..sim.scenarios import ScenarioGenerator

CATEGORIES = [
    "éxito",
    "choque: obstáculo del mapa previo",
    "choque: obstáculo no previsto",
    "choque: obstáculo móvil",
    "choque: borde del área",
    "deriva sin GPS (aterriza lejos)",
    "aterriza lejos (con GPS)",
    "atascado (no completa la ruta)",
    "no encuentra la base",
    "aterriza sin inspeccionar todo",
]
CATEGORY_COLORS = ["#27ae60", "#5b6770", "#d1495b", "#edae49", "#34495e", "#8e44ad", "#a569bd",
                   "#e67e22", "#2980b9", "#95a5a6"]


@dataclass(frozen=True)
class Diagnosis:
    controller: str
    family: str
    seed: int
    category: str
    position: tuple[float, float]
    step: int


def collision_cause(env: NavigationEnv) -> str:
    w, p, t = env.world, env.state.position, env.t
    border = min(p[0], p[1], w.width - p[0], w.height - p[1])
    if not w.obstacles:
        return "choque: borde del área"
    dists = [float(o.signed_distance(p[None], t)[0]) for o in w.obstacles]
    k = int(np.argmin(dists))
    if border < dists[k]:
        return "choque: borde del área"
    o = w.obstacles[k]
    if o.is_dynamic:
        return "choque: obstáculo móvil"
    return "choque: obstáculo del mapa previo" if o.known else "choque: obstáculo no previsto"


def classify(env: NavigationEnv, traj: Trajectory) -> str:
    tr = env.tracker
    if traj.collided:
        return collision_cause(env)
    if tr.success(env.state.position, traj.collided, traj.landed):
        return "éxito"
    if traj.landed:
        if tr.coverage < 1:
            return "aterriza sin inspeccionar todo"
        return "deriva sin GPS (aterriza lejos)" if traj.gps_lost_from is not None else "aterriza lejos (con GPS)"
    return "atascado (no completa la ruta)" if tr.coverage < 1 else "no encuentra la base"


def diagnose(cfg: ProjectConfig, make_controller: Callable[[], Controller], families: Sequence[str],
             seeds: Iterable[int]) -> list[Diagnosis]:
    gen, seeds, out = ScenarioGenerator(cfg), list(seeds), []
    for family in families:
        for seed in seeds:
            controller = make_controller()
            env = NavigationEnv(cfg, gen.generate(family, seed))
            traj = rollout(env, controller, keep_debug=False)
            out.append(Diagnosis(controller.name, family, seed, classify(env, traj),
                                 (float(env.state.position[0]), float(env.state.position[1])), env.t))
    return out


def category_shares(diagnoses: Sequence[Diagnosis]) -> dict[str, float]:
    counts = Counter(d.category for d in diagnoses)
    n = max(len(diagnoses), 1)
    return {c: counts.get(c, 0) / n for c in CATEGORIES}
