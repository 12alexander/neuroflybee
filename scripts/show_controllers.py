"""Figuras de la v12: arquitectura híbrida, efecto del supervisor y controladores neuronales.

Uso:
    python scripts/show_controllers.py [--episodes 10]
Genera siempre (sin PyTorch):
    results/figures/v12_arquitectura_hibrida.png   cómo se combinan abeja, mosca y supervisor
    results/figures/v12_supervisor.png             el supervisor evita choques (con controladores clásicos)
    results/v12_supervisor.csv
Si hay PyTorch y pesos en checkpoints/:
    results/figures/v12_controladores.png          A*, mosca, abeja e híbrido en los mismos escenarios
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from neuroflybee.config import load_config  # noqa: E402
from neuroflybee.control import classical  # noqa: E402,F401
from neuroflybee.control import registry  # noqa: E402
from neuroflybee.control.neural_registry import register_neural_controllers  # noqa: E402
from neuroflybee.evaluation.closed_loop import evaluate  # noqa: E402
from neuroflybee.sim.env import NavigationEnv  # noqa: E402
from neuroflybee.sim.rollout import rollout  # noqa: E402
from neuroflybee.sim.scenarios import FAMILY_LABELS_ES, ScenarioGenerator, seeds_for  # noqa: E402
from neuroflybee.viz.style import apply_style  # noqa: E402
from neuroflybee.viz.world_plot import draw_world  # noqa: E402

LABELS = {"direct": "Directo", "direct_supervised": "Directo + supervisor", "astar": "A*",
          "astar_supervised": "A* + supervisor", "fly": "Mosca", "bee": "Abeja", "hybrid": "Híbrido"}
COLORS = {"direct": "#9e9e9e", "direct_supervised": "#555555", "astar": "#1f4e94",
          "astar_supervised": "#5dade2", "fly": "#c0392b", "bee": "#d4ac0d", "hybrid": "#27ae60"}


# ------------------------------------------------------------------------- diagrama
def plot_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 6.6))
    boxes = [  # x, y, ancho, alto, título, detalle, color
        (0.3, 2.0, 2.6, 2.3, "Sensores", "ojo 32×32 · rayos\npanorama 72 · brújula\nGPS (puede faltar)\nodometría", "#dfe6e9"),
        (3.6, 4.3, 2.4, 1.2, "Estimador", "GPS + integración\nde trayectoria", "#ecf0f1"),
        (6.8, 3.4, 3.2, 2.1, "ABEJA (alto nivel)", "memoria de la ruta · fase\nred siamesa: dirección y\ndistancia a la base sin GPS", "#f9e79f"),
        (6.8, 0.6, 3.2, 2.1, "MOSCA (bajo nivel)", "CNN + GRU sobre 4 cuadros\nmaniobra para seguir el\nobjetivo evitando obstáculos", "#f5b7b1"),
        (10.8, 0.6, 2.4, 2.1, "SUPERVISOR", "veta maniobras que\nacercan a un obstáculo\n(sólo rayos, sin redes)", "#abebc6"),
        (13.7, 2.3, 1.4, 1.6, "Dron", "simulado o\nreal (v19)", "#d7bde2"),
    ]
    for x, y, w, h, title, detail, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=color, ec="#555555", lw=1.2))
        ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + w / 2, y + (h - 0.45) / 2, detail, ha="center", va="center", fontsize=8.5)

    def path(points, text="", color="#333333", ls="-", text_at=0, offset=0.17):
        xs, ys = zip(*points)
        ax.plot(xs[:-1], ys[:-1], color=color, lw=1.5, ls=ls)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=14,
                                     color=color, lw=1.5, ls=ls))
        if text:
            (x0, y0), (x1, y1) = points[text_at], points[text_at + 1]
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + offset, text, ha="center", fontsize=8, color=color,
                    bbox=dict(fc="white", ec="none", alpha=0.85))

    path([(2.9, 4.0), (3.6, 4.7)], "odometría / GPS")
    path([(6.0, 4.9), (6.8, 4.9)], "posición")
    path([(2.9, 3.4), (6.8, 3.9)], "panorama", color="#b7950b")
    path([(2.9, 2.4), (6.8, 1.6)], "ojo", color="#c0392b")
    path([(8.4, 3.4), (8.4, 2.7)])
    ax.text(8.55, 3.05, "objetivo relativo", fontsize=8, va="center")
    path([(10.0, 1.65), (10.8, 1.65)])
    ax.text(10.4, 1.95, "acción\npropuesta", fontsize=7.5, ha="center")
    path([(1.6, 2.0), (1.6, 0.25), (12.0, 0.25), (12.0, 0.6)], "rayos (independiente de las redes)",
         color="#1e8449", ls="--", text_at=1, offset=-0.2)
    path([(13.2, 1.65), (14.4, 1.65), (14.4, 2.3)])
    ax.text(13.8, 1.35, "acción final", fontsize=8, ha="center")
    path([(14.4, 3.9), (14.4, 6.1), (1.6, 6.1), (1.6, 4.3)], "nueva observación (ciclo cerrado)",
         color="#888888", ls=":", text_at=1)
    ax.set_xlim(0, 15.4)
    ax.set_ylim(-0.2, 6.5)
    ax.axis("off")
    ax.set_title("v12 — Controlador híbrido: la abeja decide HACIA DÓNDE, la mosca CÓMO, el supervisor SI ES SEGURO",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------------- supervisor
def outcome(env, traj) -> str:
    if traj.collided:
        return "choque"
    if env.tracker.success(env.state.position, traj.collided, traj.landed):
        return "✓ misión completa"
    return "sin choque, no completa"


def draw_run(ax, cfg, scenario, name: str) -> dict:
    env = NavigationEnv(cfg, scenario)
    traj = rollout(env, registry.create(name))
    pos = traj.positions
    ax.plot(pos[:, 0], pos[:, 1], lw=2, color=COLORS.get(name, "black"),
            label=f"{LABELS.get(name, name)} — {outcome(env, traj)}")
    vetoes = np.array([s.position for s in traj.steps if s.decision.debug.get("supervisor") == "veto"])
    if len(vetoes):
        ax.scatter(vetoes[:, 0], vetoes[:, 1], s=10, color="#e67e22", zorder=8)
    if traj.collided:
        ax.plot(*pos[-1], marker="X", ms=13, color=COLORS.get(name, "black"), mec="black", zorder=9)
    return {"collided": traj.collided, "vetoes": len(vetoes)}


def plot_supervisor(cfg, episodes: int, out_png: Path, out_csv: Path) -> None:
    gen = ScenarioGenerator(cfg)
    families = ["nominal", "unknown_obstacles", "dynamic", "wind"]
    names = ["direct", "direct_supervised", "astar", "astar_supervised"]
    seeds = seeds_for("test", episodes)
    rows = {}
    for name in names:
        res = evaluate(cfg, lambda n=name: registry.create(n), families, seeds)
        for f in families:
            sel = [r for r in res if r.family == f]
            rows[(name, f)] = (np.mean([r.collided for r in sel]), np.mean([r.success for r in sel]))

    fig = plt.figure(figsize=(16, 6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    scenario = gen.generate("unknown_obstacles", seeds[0])
    for k, pair in enumerate((("direct", "direct_supervised"), ("astar", "astar_supervised"))):
        ax = fig.add_subplot(grid[0, k])
        draw_world(ax, scenario.world, title=f"{LABELS[pair[0]]} vs {LABELS[pair[1]]}\n(obstáculos no previstos)")
        for name in pair:
            draw_run(ax, cfg, scenario, name)
        ax.scatter([], [], s=10, color="#e67e22", label="veto del supervisor")
        ax.legend(loc="upper left", fontsize=7)
    ax = fig.add_subplot(grid[0, 2])
    x = np.arange(len(families))
    width = 0.2
    for i, name in enumerate(names):
        ax.bar(x + (i - 1.5) * width, [rows[(name, f)][0] for f in families], width, color=COLORS[name],
               label=LABELS[name])
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS_ES[f] for f in families], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("episodios con choque")
    ax.set_title(f"Choques con y sin supervisor ({episodes} escenarios de prueba por condición)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("v12 — El supervisor de seguridad evita choques con obstáculos fijos; los móviles y el viento "
                 "requieren anticipación (la mosca)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["controlador", "condicion", "episodios", "choques", "exito"])
        for (name, f), (col, suc) in rows.items():
            w.writerow([name, f, episodes, f"{col:.3f}", f"{suc:.3f}"])


# ------------------------------------------------------------------------- neuronales
def plot_neural(cfg, names: list[str], seed: int, out: Path) -> None:
    gen = ScenarioGenerator(cfg)
    families = ["dynamic", "gps_loss", "combined"]
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 6.4))
    for ax, family in zip(axes, families):
        scenario = gen.generate(family, seed)
        draw_world(ax, scenario.world, title=FAMILY_LABELS_ES[family], gps_loss=scenario.gps_loss_on_return)
        for name in ["astar", *names]:
            draw_run(ax, cfg, scenario, name)
        ax.scatter([], [], s=10, color="#e67e22", label="veto del supervisor")
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"v12 — Controladores neuronales vs A* (semilla de prueba {seed})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    apply_style()
    cfg = load_config(args.config)
    results = Path(args.results)
    figs = results / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    plot_architecture(figs / "v12_arquitectura_hibrida.png")
    plot_supervisor(cfg, args.episodes, figs / "v12_supervisor.png", results / "v12_supervisor.csv")
    neural = [n for n in register_neural_controllers(args.checkpoints) if n in ("fly", "bee", "hybrid")]
    if neural:
        plot_neural(cfg, neural, args.seed, figs / "v12_controladores.png")
        print("Controladores neuronales dibujados:", neural)
    else:
        print("Sin PyTorch o sin pesos: v12_controladores.png se genera tras entrenar (v9–v11)")


if __name__ == "__main__":
    main()
