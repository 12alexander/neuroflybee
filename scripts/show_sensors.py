"""Figuras de la v4: lo que percibe el dron.

Uso:
    python scripts/show_sensors.py [--seed 20000] [--progress 0.3]
Genera:
    results/figures/v4_sensores.png             ojo de mosca, distancias y panorama
    results/figures/v4_panorama_vs_distancia.png  la vista cambia al alejarse de la base
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from neuroflybee.config import load_config  # noqa: E402
from neuroflybee.domain.drone import DroneState  # noqa: E402
from neuroflybee.domain.geometry import bearing  # noqa: E402
from neuroflybee.sim.scenarios import ScenarioGenerator, seeds_for  # noqa: E402
from neuroflybee.sim.sensors import FlyEyeSensor, PanoramaSensor, RangeSensor  # noqa: E402
from neuroflybee.viz.sensor_plot import (draw_drone, draw_eye_footprint, draw_fly_eye,  # noqa: E402
                                         draw_panorama, draw_ranges_polar, draw_rays)
from neuroflybee.viz.style import apply_style  # noqa: E402
from neuroflybee.viz.world_plot import draw_world  # noqa: E402


def pose_on_route(world, leg: int, progress: float) -> DroneState:
    route = [world.base, *world.waypoints, world.base]
    a, b = route[leg], route[leg + 1]
    p = a + progress * (b - a)
    return DroneState(p[0], p[1], bearing(a, b))


def interesting_pose(world, rays: RangeSensor, occ: np.ndarray, max_range: float) -> DroneState:
    """Primer punto de la ruta, en espacio libre, con un obstáculo al frente a 3–7 m:
    es el momento en que la mosca debe reaccionar."""
    front = int(np.argmin(np.abs(rays.angles)))
    for leg in range(len(world.waypoints) + 1):
        for progress in np.linspace(0.05, 0.95, 46):
            state = pose_on_route(world, leg, progress)
            if world.clearance(state.position)[0] < 1.5 or np.linalg.norm(state.position - world.base) < 8:
                continue
            r = rays.read(occ, world, state)
            if 3.0 <= r[front] <= 6.0:
                return state
    return pose_on_route(world, 0, 0.3)


def plot_sensors(cfg, seed: int, progress: float | None, out: Path) -> None:
    sc = ScenarioGenerator(cfg).generate("unknown_obstacles", seed)
    w, s = sc.world, cfg.sensors
    rng = np.random.default_rng(0)
    occ = w.occupancy()
    fly, rays, pano = FlyEyeSensor(s), RangeSensor(s), PanoramaSensor(s)
    state = pose_on_route(w, 0, progress) if progress is not None else interesting_pose(w, rays, occ, s.ray_range)
    patch = fly.read(occ, w, state, rng)
    ranges = rays.read(occ, w, state)
    objs = pano.objects(w)
    view, home = pano.view_from(objs, state.position), pano.view_from(objs, w.base)

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1], hspace=0.5, wspace=0.3, top=0.86)
    ax_map = fig.add_subplot(grid[:, 0])
    draw_world(ax_map, w, title="Mapa: dron, área del ojo (azul) y rayos\n(naranja = rayo que toca un obstáculo)")
    draw_eye_footprint(ax_map, fly.footprint(state))
    draw_rays(ax_map, state.position, rays.end_points(ranges, state), ranges, s.ray_range)
    draw_drone(ax_map, state)

    ax_eye = fig.add_subplot(grid[0, 1])
    draw_fly_eye(ax_eye, patch, s.patch_forward, s.patch_back, s.patch_half_width)
    ax_eye.set_title(f"Ojo de la mosca ({s.patch_size}×{s.patch_size}, ruido {s.pixel_noise:.0%})\nentrada de la CNN")

    ax_polar = fig.add_subplot(grid[0, 2], projection="polar")
    draw_ranges_polar(ax_polar, rays.angles, ranges, s.ray_range)
    ax_polar.set_title(f"Anillo de distancias ({s.n_rays} rayos, ≤{s.ray_range:.0f} m)\nsupervisor de seguridad",
                       pad=14)

    ax_pano = fig.add_subplot(grid[1, 1:])
    draw_panorama(ax_pano, pano.bin_centers, view, home)
    dist = float(np.linalg.norm(state.position - w.base))
    ax_pano.set_title(f"Panorama de la abeja ({s.panorama_bins} direcciones) a {dist:.1f} m de la base:"
                      " la diferencia con la vista memorizada guía el regreso")

    fig.suptitle("v4 — Lo que percibe el dron (escenario con obstáculos no previstos)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_panorama_vs_distance(cfg, n_scenarios: int, out: Path) -> None:
    gen, pano = ScenarioGenerator(cfg), PanoramaSensor(cfg.sensors)
    rng = np.random.default_rng(1)
    distances = np.arange(0.5, 15.1, 0.5)
    diffs = np.zeros((n_scenarios, len(distances)))
    for i, seed in enumerate(seeds_for("test", n_scenarios)):
        w = gen.generate("nominal", seed).world
        objs = pano.objects(w)
        home = pano.view_from(objs, w.base)
        directions = rng.uniform(-np.pi, np.pi, 8)
        for j, d in enumerate(distances):
            views = [pano.view_from(objs, w.base + d * np.array([np.cos(a), np.sin(a)])) for a in directions]
            diffs[i, j] = np.mean([np.sqrt(np.mean((v - home) ** 2)) for v in views]) * 90
    mean, std = diffs.mean(axis=0), diffs.std(axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(distances, mean, color="#b8860b", lw=2)
    ax.fill_between(distances, mean - std, mean + std, color="#f1c40f", alpha=0.3, label="± 1 d.e.")
    ax.axhline(cfg.sensors.panorama_noise * 90, color="grey", ls="--", lw=1, label="nivel de ruido del sensor")
    ax.set_xlabel("distancia a la base (m)")
    ax.set_ylabel("diferencia RMS con la vista memorizada (°)")
    ax.set_title(f"La vista panorámica codifica la distancia a la base ({n_scenarios} escenarios de prueba)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--progress", type=float, default=None,
                        help="avance en el primer tramo (0–1); por defecto se elige un momento con obstáculo al frente")
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = parser.parse_args(argv)

    apply_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    plot_sensors(cfg, args.seed, args.progress, out / "v4_sensores.png")
    plot_panorama_vs_distance(cfg, args.scenarios, out / "v4_panorama_vs_distancia.png")
    print(f"Figuras guardadas en {out}")


if __name__ == "__main__":
    main()
