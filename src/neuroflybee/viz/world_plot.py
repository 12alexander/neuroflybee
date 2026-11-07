"""Dibujo de escenarios: obstáculos, base, ruta, viento y condiciones especiales."""
from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle

from ..domain.world import CircleObstacle, MovingObstacle, RectObstacle, World
from .style import COLORS


def draw_obstacles(ax: Axes, world: World, t: int = 0, show_unknown: bool = True) -> None:
    for o in world.obstacles:
        if not o.known and not show_unknown:
            continue
        if isinstance(o, MovingObstacle):
            start, end = o.center(0), o.center(0) + np.array([o.vx, o.vy]) / np.hypot(o.vx, o.vy) * o.span
            ax.plot([start[0], end[0]], [start[1], end[1]], ls=":", lw=1.2, color=COLORS["moving"])
            ax.add_patch(Circle(o.center(t), o.r, color=COLORS["moving"], ec="black", lw=0.6, zorder=3))
        elif isinstance(o, CircleObstacle):
            color = COLORS["known"] if o.known else COLORS["unknown"]
            ax.add_patch(Circle((o.x, o.y), o.r, color=color, alpha=0.9, lw=0))
        elif isinstance(o, RectObstacle):
            color = COLORS["known"] if o.known else COLORS["unknown"]
            ax.add_patch(Rectangle((o.xmin, o.ymin), o.xmax - o.xmin, o.ymax - o.ymin,
                                   color=color, alpha=0.9, lw=0))


def draw_route(ax: Axes, world: World, show_order: bool = True) -> None:
    route = np.array([world.base, *world.waypoints, world.base])
    ax.plot(route[:, 0], route[:, 1], ls="--", lw=1, color=COLORS["route"], alpha=0.6)
    ax.plot(*world.base, marker="s", ms=10, color=COLORS["base"], ls="", zorder=4)
    for i, wp in enumerate(world.waypoints):
        ax.plot(*wp, marker="o", ms=9, mfc="white", mec=COLORS["waypoint"], mew=2.2, ls="", zorder=4)
        if show_order:
            ax.annotate(str(i + 1), wp, ha="center", va="center", fontsize=7, fontweight="bold",
                        color=COLORS["waypoint"], zorder=5)


def draw_wind(ax: Axes, world: World) -> None:
    speed = float(np.linalg.norm(world.wind))
    if speed < 1e-6:
        return
    d = world.wind / speed
    origin = np.array([world.width - 6.0, 6.0])
    ax.add_patch(Circle(origin, 5.2, color="white", alpha=0.55, zorder=5))
    ax.add_patch(FancyArrowPatch(origin - d * 3.5, origin + d * 3.5, arrowstyle="-|>", mutation_scale=14,
                                 color=COLORS["wind"], lw=2, zorder=6))
    ax.text(origin[0], origin[1] + 5.8, f"viento {speed:.2f} m/s", ha="center", fontsize=7,
            color=COLORS["wind"], zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def draw_world(ax: Axes, world: World, t: int = 0, title: str = "", gps_loss: bool = False,
               show_unknown: bool = True) -> None:
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.set_facecolor(COLORS["background"])
    draw_obstacles(ax, world, t, show_unknown)
    draw_route(ax, world)
    draw_wind(ax, world)
    if gps_loss:
        ax.text(1.5, 1.5, "GPS se pierde al regresar", fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["unknown"], ec="none"))
    ax.set_title(title)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([0, 20, 40, 60])


def legend_handles() -> list:
    return [
        Line2D([], [], marker="s", ls="", color=COLORS["base"], ms=8, label="Base"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=COLORS["waypoint"], mew=2, ms=8,
               label="Punto de inspección"),
        Line2D([], [], ls="--", color=COLORS["route"], label="Ruta planificada"),
        Patch(color=COLORS["known"], label="Obstáculo en el mapa previo"),
        Patch(color=COLORS["unknown"], label="Obstáculo no previsto"),
        Line2D([], [], marker="o", ls=":", color=COLORS["moving"], mec="black", ms=8,
               label="Obstáculo móvil (y su recorrido)"),
        Line2D([], [], color=COLORS["wind"], lw=2, label="Viento"),
    ]


def _hex_to_rgb(color: str) -> np.ndarray:
    return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def draw_occupancy(ax: Axes, world: World, known_only: bool, title: Optional[str] = None) -> None:
    """Mapa de ocupación: gris = en el mapa previo; rojo = no previsto (sólo en la realidad)."""
    known = world.occupancy(known_only=True)
    image = np.ones((*known.shape, 3))
    image[known] = _hex_to_rgb(COLORS["known"])
    if not known_only:
        image[world.occupancy() & ~known] = _hex_to_rgb(COLORS["unknown"])
    ax.imshow(image, origin="lower", extent=(0, world.width, 0, world.height), interpolation="nearest")
    ax.plot(*world.base, marker="s", ms=8, color=COLORS["base"], ls="")
    for wp in world.waypoints:
        ax.plot(*wp, marker="o", ms=7, mfc="white", mec=COLORS["waypoint"], mew=2, ls="")
    ax.set_title(title or ("Mapa previo (lo que conoce el planificador)" if known_only
                           else "Realidad (lo que encontrará el dron)"))
    ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([0, 20, 40, 60])
