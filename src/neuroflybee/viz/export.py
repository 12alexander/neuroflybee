"""Exporta episodios al formato del visor HTML (`results/visor/datos.js`).

El visor no ejecuta Python: reproduce lo que se exporta aquí. Por cada paso se guarda la
posición real y la estimada, la acción, el módulo que decidió, la fase, el GPS, los vetos
del supervisor, las probabilidades de la red (si hay) y los tres sensores (ojo de la mosca
codificado en hexadecimal, rayos y panorama).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import ProjectConfig
from ..control.base import Controller
from ..domain.drone import ACTION_LABELS_ES, Action
from ..domain.world import CircleObstacle, MovingObstacle, RectObstacle
from ..sim.env import NavigationEnv
from ..sim.rollout import Trajectory, rollout
from ..sim.scenarios import FAMILY_LABELS_ES, Scenario

CONTROLLER_STYLE = {
    "direct": ("Directo", "#9e9e9e"),
    "direct_supervised": ("Directo + supervisor", "#7f8c8d"),
    "potential_field": ("Campos de potencial", "#e67e22"),
    "astar": ("A* (clásico)", "#3498db"),
    "astar_alv": ("A* + ALV (clásico)", "#9b59b6"),
    "astar_supervised": ("A* + supervisor", "#5dade2"),
    "fly": ("Mosca (CNN+GRU)", "#e74c3c"),
    "bee": ("Abeja (red siamesa)", "#f1c40f"),
    "hybrid": ("Híbrido abeja+mosca", "#2ecc71"),
    "hybrid_no_supervisor": ("Híbrido sin supervisor", "#16a085"),
}


def pack_bits(patch: np.ndarray) -> str:
    """Parche binario (S, S) → texto hexadecimal (S*S/4 caracteres)."""
    bits = (np.asarray(patch) > 0.5).astype(np.uint8).ravel()
    return np.packbits(bits).tobytes().hex()


def unpack_bits(text: str, size: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(text), dtype=np.uint8))[: size * size]
    return bits.reshape(size, size)


def _r(x: float, nd: int = 2) -> float:
    return round(float(x), nd)


def world_payload(scenario: Scenario, horizon: int) -> dict:
    w = scenario.world
    static, dynamic = [], []
    for o in w.obstacles:
        if isinstance(o, MovingObstacle):
            dynamic.append({"r": _r(o.r), "path": [[_r(v) for v in o.center(t)] for t in range(horizon + 1)]})
        elif isinstance(o, CircleObstacle):
            static.append({"type": "circle", "x": _r(o.x), "y": _r(o.y), "r": _r(o.r), "known": o.known})
        elif isinstance(o, RectObstacle):
            static.append({"type": "rect", "x0": _r(o.xmin), "y0": _r(o.ymin), "x1": _r(o.xmax), "y1": _r(o.ymax),
                           "known": o.known})
    return {"width": w.width, "height": w.height, "base": [_r(v) for v in w.base],
            "waypoints": [[_r(v) for v in p] for p in w.waypoints], "obstacles": static, "dynamic": dynamic,
            "wind": [_r(v, 3) for v in w.wind], "gps_loss": scenario.gps_loss_on_return}


def outcome(env: NavigationEnv, traj: Trajectory) -> dict:
    err = env.tracker.home_error(env.state.position)
    success = env.tracker.success(env.state.position, traj.collided, traj.landed)
    if traj.collided:
        text = f"Choque en el paso {len(traj.steps) - 1}"
    elif success:
        text = f"Misión completa · aterrizó a {err:.1f} m de la base"
    elif traj.landed:
        text = f"Aterrizó a {err:.1f} m de la base (fuera del radio de 2 m)"
    else:
        text = f"No terminó en {len(traj.steps) - 1} pasos · inspeccionó {env.tracker.coverage:.0%}"
    return {"text": text, "success": bool(success), "collided": bool(traj.collided),
            "coverage": _r(env.tracker.coverage), "home_error": _r(err), "steps": len(traj.steps) - 1}


def run_payload(cfg: ProjectConfig, scenario: Scenario, controller: Controller) -> tuple[dict, NavigationEnv]:
    env = NavigationEnv(cfg, scenario)
    observations = []

    class Recorder:                       # envuelve al controlador para guardar los sensores
        name = controller.name

        def reset(self, obs):
            observations.clear()
            controller.reset(obs)

        def act(self, obs):
            observations.append(obs)
            return controller.act(obs)

    traj = rollout(env, Recorder())
    observations.append(env.last_observation)
    steps = []
    for i, st in enumerate(traj.steps):
        d, obs = st.decision, observations[min(i, len(observations) - 1)]
        steps.append({
            "p": [_r(st.position[0]), _r(st.position[1])], "h": _r(st.heading, 3),
            "e": None if d.estimate is None else [_r(d.estimate[0]), _r(d.estimate[1])],
            "a": int(d.action), "s": d.source or "", "ph": getattr(d.phase, "value", str(d.phase)),
            "g": bool(st.info.gps_available), "v": d.debug.get("supervisor") == "veto",
            "pr": None if d.probs is None else [_r(p, 3) for p in d.probs],
            "eye": pack_bits(obs.fly_patch), "rg": [_r(r, 1) for r in obs.ranges],
            "pn": [_r(v, 2) for v in obs.panorama],
        })
    label, color = CONTROLLER_STYLE.get(controller.name, (controller.name, "#ecf0f1"))
    return {"controller": controller.name, "label": label, "color": color, "outcome": outcome(env, traj),
            "steps": steps}, env


def build_viewer_data(cfg: ProjectConfig, scenarios: Sequence[Scenario], controllers: Sequence[str],
                      make_controller) -> dict:
    out = []
    for sc in scenarios:
        runs, home = [], None
        for name in controllers:
            run, env = run_payload(cfg, sc, make_controller(name))
            runs.append(run)
            home = env.home_snapshot
        horizon = max(len(r["steps"]) for r in runs)
        out.append({"id": f"{sc.family}-{sc.seed}", "family": sc.family, "label": FAMILY_LABELS_ES[sc.family],
                    "seed": sc.seed, "world": world_payload(sc, horizon),
                    "home_panorama": [_r(v, 3) for v in home], "runs": runs})
    s = cfg.sensors
    return {"meta": {"title": "NeuroFly-Bee Perú — visor de la simulación",
                     "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                     "actions": [ACTION_LABELS_ES[a] for a in Action],
                     "eye": {"size": s.patch_size, "forward": s.patch_forward, "back": s.patch_back,
                             "half_width": s.patch_half_width},
                     "ray_range": s.ray_range, "drone_radius": cfg.drone.radius},
            "scenarios": out}


def write_viewer_data(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    Path(path).write_text("window.NFB_DATA = " + text + ";\n", encoding="utf-8")
