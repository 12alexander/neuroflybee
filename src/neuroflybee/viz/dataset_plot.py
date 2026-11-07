"""Figuras de los conjuntos de datos (mosca y abeja)."""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from ..domain.drone import ACTION_LABELS_ES, Action
from .sensor_plot import EYE_COLOR


def draw_eye_frame(ax: Axes, patch: np.ndarray, forward: float, back: float, half_width: float,
                   goal: np.ndarray | None = None, goal_scale: float = 10.0) -> None:
    """Un cuadro del ojo; si se da `goal` (cos, sin, dist_norm) se dibuja la flecha al objetivo."""
    ax.imshow(patch, cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
              extent=(-half_width, half_width, -back, forward))
    ax.plot(0, 0, marker="^", ms=6, color=EYE_COLOR)
    if goal is not None:
        d = min(goal[2] * goal_scale, forward)
        # columna izquierda de la imagen = izquierda del dron → x de la figura = -lateral
        ax.annotate("", xy=(-goal[1] * d, goal[0] * d), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#30a14e", lw=2))
    ax.set_xticks([])
    ax.set_yticks([])


def draw_action_distribution(ax: Axes, counts: dict[str, np.ndarray]) -> None:
    labels = [ACTION_LABELS_ES[a] for a in Action]
    width = 0.8 / len(counts)
    x = np.arange(len(labels))
    colors = ["#1f4e94", "#e67e22", "#30a14e"]
    for i, (name, c) in enumerate(counts.items()):
        frac = c / max(c.sum(), 1)
        bars = ax.bar(x + (i - (len(counts) - 1) / 2) * width, frac, width, label=name, color=colors[i % 3])
        for b, f in zip(bars, frac):
            ax.text(b.get_x() + b.get_width() / 2, f + 0.005, f"{f:.1%}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
