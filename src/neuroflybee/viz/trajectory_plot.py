"""Dibujo de episodios: recorrido real, recorrido estimado y desenlace."""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from ..sim.rollout import Trajectory
from .sensor_plot import draw_drone

TRUE_COLOR = "#1f4e94"
BELIEF_COLOR = "#c0392b"


def draw_trajectory(ax: Axes, traj: Trajectory, show_belief: bool = True) -> None:
    pos = traj.positions
    ax.plot(pos[:, 0], pos[:, 1], color=TRUE_COLOR, lw=2, label="recorrido real", zorder=7)
    lost = traj.gps_lost_from
    if show_belief and lost is not None:
        est = traj.estimates[lost:]
        ax.plot(est[:, 0], est[:, 1], color=BELIEF_COLOR, lw=1.6, ls="--",
                label="lo que el dron cree (sin GPS)", zorder=7)
        ax.plot(*pos[lost], marker="D", ms=7, color=BELIEF_COLOR, ls="", zorder=8,
                label="se pierde el GPS")
    last = traj.steps[-1]
    if traj.collided:
        ax.plot(*last.position, marker="X", ms=14, color="#d1495b", mec="black", ls="", zorder=9,
                label="choque")
    else:
        draw_drone(ax, last.info.true_state)


def outcome_text(traj: Trajectory, home_error: float, coverage: float) -> str:
    steps = len(traj.steps) - 1
    if traj.collided:
        return f"Choque en el paso {steps} · inspeccionó {coverage:.0%}"
    if traj.landed:
        return f"Aterrizó a {home_error:.1f} m de la base · inspeccionó {coverage:.0%} · {steps} pasos"
    return f"Sin terminar en {steps} pasos · inspeccionó {coverage:.0%}"


def path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0
