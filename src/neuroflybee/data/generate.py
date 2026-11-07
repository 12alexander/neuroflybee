"""Generación de conjuntos de datos para la mosca (imitación) y la abeja (retorno).

Todo se guarda como `.npz` comprimido; los parches binarios se guardan en uint8.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from ..config import ProjectConfig
from ..control.estimation import PositionEstimator
from ..control.navigation import MissionNavigator
from ..domain.drone import Action
from ..domain.geometry import to_body_frame
from ..sim.env import NavigationEnv
from ..sim.observation import Observation
from ..sim.scenarios import FAMILIES, ScenarioGenerator, Split, seeds_for
from ..sim.sensors import PanoramaSensor
from .expert import PrivilegedExpert

FLY_TRAIN_FAMILIES = ("nominal", "unknown_obstacles", "dynamic", "wind")
GOAL_SCALE = 10.0

# Política de comportamiento opcional (DAgger): recibe la observación y el vector
# objetivo y devuelve la acción que realmente se ejecuta.
BehaviorPolicy = Callable[[Observation, np.ndarray], Action]


def goal_features(obs: Observation, goal: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Objetivo en el marco del dron: (cos, sin, distancia normalizada)."""
    rel = to_body_frame(goal - estimate, obs.heading)
    dist = float(np.linalg.norm(rel))
    ang = float(np.arctan2(rel[1], rel[0]))
    return np.array([np.cos(ang), np.sin(ang), min(dist, GOAL_SCALE) / GOAL_SCALE], dtype=np.float32)


@dataclass
class FlyDataset:
    """Muestras de imitación para la mosca. Las secuencias temporales se arman al
    entrenar a partir de `episode` y `t`, sin duplicar cuadros en disco."""
    patches: np.ndarray     # (N, S, S) uint8 — ojo de la mosca
    goals: np.ndarray       # (N, 3) float32 — objetivo relativo (cos, sin, distancia)
    actions: np.ndarray     # (N,) int64 — etiqueta del experto
    episode: np.ndarray     # (N,) int32
    t: np.ndarray           # (N,) int32
    family: np.ndarray      # (N,) int8 — índice en FAMILIES (para analizar errores)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, patches=self.patches, goals=self.goals, actions=self.actions,
                            episode=self.episode, t=self.t, family=self.family)

    @classmethod
    def load(cls, path: str | Path) -> "FlyDataset":
        z = np.load(path)
        return cls(z["patches"], z["goals"], z["actions"], z["episode"], z["t"], z["family"])

    @classmethod
    def concat(cls, parts: Sequence["FlyDataset"]) -> "FlyDataset":
        """Une conjuntos (p. ej., iteraciones de DAgger) renumerando los episodios."""
        offset, episodes = 0, []
        for p in parts:
            episodes.append(p.episode + offset)
            if len(p.episode):
                offset = int(episodes[-1].max()) + 1
        return cls(np.concatenate([p.patches for p in parts]), np.concatenate([p.goals for p in parts]),
                   np.concatenate([p.actions for p in parts]), np.concatenate(episodes),
                   np.concatenate([p.t for p in parts]), np.concatenate([p.family for p in parts]))

    def __len__(self) -> int:
        return len(self.actions)

    def action_counts(self) -> np.ndarray:
        return np.bincount(self.actions, minlength=len(Action))


def collect_fly_episodes(cfg: ProjectConfig, split: Split | str, n_episodes: int,
                         families: Sequence[str] = FLY_TRAIN_FAMILIES, offset: int = 0,
                         noise_eps: float = 0.15, behavior: Optional[BehaviorPolicy] = None,
                         beta: float = 1.0, seed: int = 0, max_steps: int = 400) -> FlyDataset:
    """Ejecuta episodios y etiqueta cada estado con el experto privilegiado.

    El objetivo que ve el estudiante es el que el dron CREE tener (navegador con la
    posición estimada, igual que en la prueba); el experto calcula la mejor acción hacia
    ese mismo objetivo desde la posición real.

    * `behavior=None`: imitación directa con ruido de acción `noise_eps` (DART).
    * `behavior=política`, `beta<1`: iteración DAgger (con prob. 1-beta actúa el estudiante;
      la política se consulta en cada paso y debe reiniciar su memoria cuando `obs.t == 0`).
    """
    gen = ScenarioGenerator(cfg)
    family_index = {name: i for i, name in enumerate(FAMILIES)}
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {k: [] for k in ("patches", "goals", "actions", "episode", "t", "family")}
    for ep, scen_seed in enumerate(seeds_for(split, n_episodes, offset)):
        family = families[ep % len(families)]
        env = NavigationEnv(cfg, gen.generate(family, scen_seed), seed=seed)
        expert = PrivilegedExpert(env.world, cfg)
        obs = env.reset()
        estimator, navigator = PositionEstimator(), MissionNavigator()
        estimator.reset(obs)
        navigator.reset(obs)
        for _ in range(max_steps):
            estimate = estimator.update(obs)
            target = navigator.update(obs, estimate)
            label = expert.act(env.state, target.goal, env.t)
            g = goal_features(obs, target.goal, estimate)
            rows["patches"].append((obs.fly_patch > 0.5).astype(np.uint8))
            rows["goals"].append(g)
            rows["actions"].append(int(label))
            rows["episode"].append(ep)
            rows["t"].append(env.t)
            rows["family"].append(family_index[family])
            executed = _behavior_action(rng, label, obs, g, behavior, beta, noise_eps)
            result = env.step(executed, target.land)
            obs = result.observation
            if result.done:
                break
    return FlyDataset(np.array(rows["patches"], dtype=np.uint8), np.array(rows["goals"], dtype=np.float32),
                      np.array(rows["actions"], dtype=np.int64), np.array(rows["episode"], dtype=np.int32),
                      np.array(rows["t"], dtype=np.int32), np.array(rows["family"], dtype=np.int8))


def _behavior_action(rng: np.random.Generator, label: Action, obs: Observation, goal: np.ndarray,
                     behavior: Optional[BehaviorPolicy], beta: float, noise_eps: float) -> Action:
    """Acción que realmente se ejecuta (la etiqueta guardada siempre es la del experto).

    Con DAgger, el estudiante se consulta en TODOS los pasos (así su memoria de cuadros
    queda completa) y se ejecuta su acción con probabilidad 1 − beta."""
    if behavior is not None:
        student = behavior(obs, goal)
        return student if rng.random() >= beta else label
    if rng.random() < noise_eps:
        return Action(int(rng.integers(len(Action))))
    return label


@dataclass
class BeeDataset:
    current: np.ndarray     # (N, B) panorama actual
    home: np.ndarray        # (N, B) panorama memorizado en la base
    target: np.ndarray      # (N, 3): cos, sin (dirección a la base), log(1 + distancia)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, current=self.current, home=self.home, target=self.target)

    @classmethod
    def load(cls, path: str | Path) -> "BeeDataset":
        z = np.load(path)
        return cls(z["current"], z["home"], z["target"])

    def __len__(self) -> int:
        return len(self.target)


def collect_bee_samples(cfg: ProjectConfig, split: Split | str, n_scenarios: int, per_scenario: int = 64,
                        max_radius: float = 12.0, offset: int = 0, seed: int = 0,
                        families: Sequence[str] = ("nominal", "unknown_obstacles")) -> BeeDataset:
    """Vistas tomadas alrededor de la base, en espacio libre, con su vector de retorno real."""
    gen = ScenarioGenerator(cfg)
    ps = PanoramaSensor(cfg.sensors)
    rng = np.random.default_rng(seed)
    cur, home, target = [], [], []
    for i, scen_seed in enumerate(seeds_for(split, n_scenarios, offset)):
        world = gen.generate(families[i % len(families)], scen_seed).world
        objs = ps.objects(world)
        snapshot = ps.view_from(objs, world.base)
        got = 0
        while got < per_scenario:
            r = max_radius * np.sqrt(rng.uniform(0.002, 1.0))
            a = rng.uniform(-np.pi, np.pi)
            p = world.base + r * np.array([np.cos(a), np.sin(a)])
            if not (0 < p[0] < world.width and 0 < p[1] < world.height):
                continue
            if world.clearance(p)[0] < 0.5:
                continue
            view = np.clip(ps.view_from(objs, p) + rng.normal(0, ps.noise, snapshot.shape), 0, 1)
            cur.append(view.astype(np.float32))
            home.append(snapshot.astype(np.float32))
            back = a + np.pi
            target.append([np.cos(back), np.sin(back), np.log1p(r)])
            got += 1
    return BeeDataset(np.array(cur), np.array(home), np.array(target, dtype=np.float32))


# --------------------------------------------------------------------------- configuración
@dataclass(frozen=True)
class FlyDataConfig:
    train_episodes: int = 400
    val_episodes: int = 60
    noise_eps: float = 0.15
    max_steps: int = 400


@dataclass(frozen=True)
class BeeDataConfig:
    train_scenarios: int = 600
    val_scenarios: int = 100
    test_scenarios: int = 100        # v20: partición de PRUEBA para el informe final de la abeja
    per_scenario: int = 64
    max_radius: float = 12.0

    def scenarios_for(self, split: Split | str) -> int:
        """Escenarios configurados para una partición."""
        return {Split.TRAIN: self.train_scenarios, Split.VAL: self.val_scenarios,
                Split.TEST: self.test_scenarios}[Split(split)]


@dataclass(frozen=True)
class DataConfig:
    seed: int = 0
    out_dir: str = "data/generated"
    fly: FlyDataConfig = FlyDataConfig()
    bee: BeeDataConfig = BeeDataConfig()


def load_or_collect_bee_split(cfg: ProjectConfig, bee: BeeDataConfig, split: Split | str, data_dir: str | Path,
                              seed: int = 0, scale: float = 1.0) -> BeeDataset:
    """Lee `bee_<split>.npz` si existe; si no, lo genera (determinista) y lo guarda.

    Permite correr la ablación de v20 aunque los datos se hayan generado con una versión
    anterior que no tenía la partición de prueba de la abeja."""
    path = Path(data_dir) / f"bee_{Split(split).value}.npz"
    if path.exists():
        return BeeDataset.load(path)
    n = max(1, int(round(bee.scenarios_for(split) * scale)))
    data = collect_bee_samples(cfg, split, n, bee.per_scenario, bee.max_radius, seed=seed)
    data.save(path)
    return data
