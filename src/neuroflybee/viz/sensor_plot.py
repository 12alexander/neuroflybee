"""Dibujo de lo que percibe el dron: ojo de la mosca, anillo de distancias y
panorama de la abeja."""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon

from ..domain.drone import DroneState
from .style import COLORS

DRONE_COLOR = "#111111"
EYE_COLOR = "#1f77b4"
RAY_COLOR = "#e67e22"
BEE_COLOR = "#f1c40f"


def draw_drone(ax: Axes, state: DroneState, size: float = 1.4) -> None:
    """Triángulo que apunta en la dirección de vuelo."""
    h = state.heading
    tip = state.position + size * np.array([np.cos(h), np.sin(h)])
    left = state.position + 0.6 * size * np.array([np.cos(h + 2.5), np.sin(h + 2.5)])
    right = state.position + 0.6 * size * np.array([np.cos(h - 2.5), np.sin(h - 2.5)])
    ax.add_patch(Polygon([tip, left, right], closed=True, color=DRONE_COLOR, zorder=8))


def draw_eye_footprint(ax: Axes, corners: np.ndarray) -> None:
    ax.add_patch(Polygon(corners, closed=True, fill=True, fc=EYE_COLOR, alpha=0.12,
                         ec=EYE_COLOR, lw=1.5, ls="--", zorder=6))


def draw_rays(ax: Axes, origin: np.ndarray, end_points: np.ndarray, ranges: np.ndarray,
              max_range: float) -> None:
    for end, r in zip(end_points, ranges):
        hit = r < max_range - 1e-3
        ax.plot([origin[0], end[0]], [origin[1], end[1]], lw=0.8,
                color=RAY_COLOR if hit else "#bbbbbb", alpha=0.9 if hit else 0.5, zorder=7)
        if hit:
            ax.plot(*end, marker="o", ms=2.5, color=RAY_COLOR, zorder=7)


def draw_fly_eye(ax: Axes, patch: np.ndarray, forward: float, back: float, half_width: float) -> None:
    """Parche egocéntrico: arriba = adelante, izquierda = izquierda del dron."""
    ax.imshow(patch, cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
              extent=(-half_width, half_width, -back, forward))
    ax.plot(0, 0, marker="^", ms=10, color=EYE_COLOR)
    ax.set_xlabel("lateral (m)  ← izquierda · derecha →")
    ax.set_ylabel("adelante (m)")


def draw_ranges_polar(ax: Axes, angles: np.ndarray, ranges: np.ndarray, max_range: float) -> None:
    """`ax` debe ser polar. 0° = frente del dron."""
    a = np.append(angles, angles[0])
    r = np.append(ranges, ranges[0])
    ax.set_theta_zero_location("N")
    ax.fill(a, r, color=RAY_COLOR, alpha=0.25)
    ax.plot(a, r, color=RAY_COLOR, lw=1.5)
    ax.set_ylim(0, max_range)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["frente", "izq.", "atrás", "der."])


def draw_panorama(ax: Axes, bin_centers: np.ndarray, current: np.ndarray, home: np.ndarray) -> None:
    deg = np.degrees(bin_centers)
    ax.fill_between(deg, 0, current * 90, step="mid", color=BEE_COLOR, alpha=0.55,
                    label="vista actual")
    ax.step(deg, home * 90, where="mid", color=COLORS["base"], lw=1.6, label="memorizada en la base")
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["O", "S", "E", "N", "O"])
    ax.set_xlabel("dirección (brújula)")
    ax.set_ylabel("elevación del horizonte (°)")
    ax.legend(loc="upper right", fontsize=7)
