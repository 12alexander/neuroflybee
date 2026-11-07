"""Figuras de la v6: líneas base clásicas.

Uso:
    python scripts/show_baselines.py [--seed 20000] [--episodes 10]
Genera:
    results/figures/v6_lineas_base.png   las 4 líneas base en los mismos escenarios
    results/figures/v6_preliminar.png    éxito y choques por condición (resultado preliminar)
    results/v6_preliminar.csv            los mismos números en tabla
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

from neuroflybee.config import load_config  # noqa: E402
from neuroflybee.control import classical  # noqa: E402,F401  (registra los controladores)
from neuroflybee.control import registry  # noqa: E402
from neuroflybee.sim.env import NavigationEnv  # noqa: E402
from neuroflybee.sim.rollout import rollout  # noqa: E402
from neuroflybee.sim.scenarios import FAMILIES, FAMILY_LABELS_ES, ScenarioGenerator, seeds_for  # noqa: E402
from neuroflybee.viz.bars import CONTROLLER_COLORS, CONTROLLER_LABELS_ES, grouped_bars  # noqa: E402
from neuroflybee.viz.style import apply_style  # noqa: E402
from neuroflybee.viz.world_plot import draw_world  # noqa: E402

BASELINES = ["direct", "potential_field", "astar", "astar_alv"]
LINE_STYLES = {"direct": "-", "potential_field": "-", "astar": "-", "astar_alv": "--"}


def outcome_label(env, traj) -> str:
    """Desenlace corto y honesto de un episodio."""
    err = env.tracker.home_error(env.state.position)
    if traj.collided:
        return "choque"
    if env.tracker.coverage < 1:
        return f"inspeccionó {env.tracker.coverage:.0%}"
    if not traj.landed:
        return "no aterrizó"
    return f"✓ aterrizó a {err:.1f} m" if err <= env.tracker.home_radius else f"aterrizó a {err:.1f} m de la base"


def plot_trajectories(cfg, seed: int, families: list[str], out: Path) -> None:
    gen = ScenarioGenerator(cfg)
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 6.4))
    for ax, family in zip(np.atleast_1d(axes), families):
        scenario = gen.generate(family, seed)
        draw_world(ax, scenario.world, title=FAMILY_LABELS_ES[family], gps_loss=scenario.gps_loss_on_return)
        for name in BASELINES:
            env = NavigationEnv(cfg, scenario)
            traj = rollout(env, registry.create(name), keep_debug=False)
            pos = traj.positions
            ax.plot(pos[:, 0], pos[:, 1], lw=2.6 if name == "astar" else 1.6, ls=LINE_STYLES[name],
                    color=CONTROLLER_COLORS[name], label=f"{CONTROLLER_LABELS_ES[name]} — {outcome_label(env, traj)}")
            if traj.collided:
                ax.plot(*pos[-1], marker="X", ms=12, color=CONTROLLER_COLORS[name], mec="black")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.suptitle(f"v6 — Líneas base clásicas en los mismos escenarios (semilla {seed}).  "
                 "Con GPS, A* y A*+ALV vuelan igual (el ALV sólo actúa sin GPS)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def evaluate(cfg, n_episodes: int) -> dict[tuple[str, str], np.ndarray]:
    """(controlador, condición) → matriz (episodios × [éxito, choque, error de retorno])."""
    gen = ScenarioGenerator(cfg)
    results: dict[tuple[str, str], np.ndarray] = {}
    for family in FAMILIES:
        scenarios = [gen.generate(family, s) for s in seeds_for("test", n_episodes)]
        for name in BASELINES:
            rows = []
            for scenario in scenarios:
                env = NavigationEnv(cfg, scenario)
                traj = rollout(env, registry.create(name), keep_debug=False)
                rows.append([env.tracker.success(env.state.position, traj.collided, traj.landed),
                             traj.collided, env.tracker.home_error(env.state.position)])
            results[(name, family)] = np.array(rows, dtype=float)
    return results


def plot_summary(results, n_episodes: int, out_png: Path, out_csv: Path) -> None:
    families = list(FAMILIES)
    labels = [FAMILY_LABELS_ES[f] for f in families]
    success = {n: np.array([results[(n, f)][:, 0].mean() for f in families]) for n in BASELINES}
    crashes = {n: np.array([results[(n, f)][:, 1].mean() for f in families]) for n in BASELINES}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    grouped_bars(axes[0], labels, success, ylabel="misiones exitosas")
    axes[0].set_title("Éxito de la misión (inspecciona todo, sin choques, aterriza a ≤ 2 m de la base)")
    grouped_bars(axes[1], labels, crashes, ylabel="episodios con choque")
    axes[1].set_title("Choques")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"v6 — Resultado PRELIMINAR de las líneas base ({n_episodes} escenarios de prueba por condición)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["controlador", "condicion", "episodios", "exito", "choques", "error_retorno_mediano_m"])
        for (name, family), r in results.items():
            w.writerow([name, family, len(r), f"{r[:, 0].mean():.3f}", f"{r[:, 1].mean():.3f}",
                        f"{np.median(r[:, 2]):.2f}"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--families", default="unknown_obstacles,gps_loss")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = parser.parse_args(argv)

    apply_style()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_trajectories(cfg, args.seed, args.families.split(","), out / "v6_lineas_base.png")
    results = evaluate(cfg, args.episodes)
    plot_summary(results, args.episodes, out / "v6_preliminar.png", out.parent / "v6_preliminar.csv")
    print(f"Figuras guardadas en {out}")


if __name__ == "__main__":
    main()
