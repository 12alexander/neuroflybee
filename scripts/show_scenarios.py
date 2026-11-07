"""Figuras de la v3: las 6 condiciones de evaluación y el mapa previo vs la realidad.

Uso:
    python scripts/show_scenarios.py                 # semilla de prueba 20000
    python scripts/show_scenarios.py --seed 20007
Genera:
    results/figures/v3_escenarios.png
    results/figures/v3_mapa_previo_vs_real.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from neuroflybee.config import load_config  # noqa: E402
from neuroflybee.sim.scenarios import FAMILIES, FAMILY_LABELS_ES, ScenarioGenerator  # noqa: E402
from neuroflybee.viz.style import apply_style  # noqa: E402
from neuroflybee.viz.world_plot import draw_occupancy, draw_world, legend_handles  # noqa: E402


def plot_families(gen: ScenarioGenerator, seed: int, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 9.6), gridspec_kw={"hspace": 0.38})
    for ax, family in zip(axes.flat, FAMILIES):
        sc = gen.generate(family, seed)
        n_unknown = sum(not o.known and not o.is_dynamic for o in sc.world.obstacles)
        n_moving = sum(o.is_dynamic for o in sc.world.obstacles)
        title = f"{FAMILY_LABELS_ES[family]}\n{len(sc.world.waypoints)} puntos · {n_unknown} no previstos · {n_moving} móviles"
        draw_world(ax, sc.world, title=title, gps_loss=sc.gps_loss_on_return)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"NeuroFly-Bee Perú — 6 condiciones de evaluación (semilla {seed}, partición de prueba)",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.12, wspace=0.15)
    fig.savefig(out)
    plt.close(fig)


def plot_prior_vs_real(gen: ScenarioGenerator, seed: int, out: Path) -> None:
    sc = gen.generate("unknown_obstacles", seed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    draw_occupancy(axes[0], sc.world, known_only=True)
    draw_occupancy(axes[1], sc.world, known_only=False)
    n_unknown = sum(not o.known for o in sc.world.obstacles)
    fig.suptitle(f"Obstáculos no previstos: el mapa previo no coincide con la realidad "
                 f"({n_unknown} obstáculos nuevos en rojo)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = parser.parse_args(argv)

    apply_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = ScenarioGenerator(load_config(args.config))
    plot_families(gen, args.seed, out / "v3_escenarios.png")
    plot_prior_vs_real(gen, args.seed, out / "v3_mapa_previo_vs_real.png")
    print(f"Figuras guardadas en {out}")


if __name__ == "__main__":
    main()
