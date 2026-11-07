"""Demostración de inferencia de punta a punta (v23): un escenario de PRUEBA, todos los controladores.

Carga los pesos entrenados (si hay PyTorch y checkpoints/), hace volar en el MISMO escenario
las líneas base clásicas y los controladores neuronales, y dibuja sus recorridos con el
desenlace de cada uno. Sin pesos funciona igual con los clásicos (útil para probar el entorno).

Uso:
    python scripts/demo_inference.py                              # combinada, semilla 20003
    python scripts/demo_inference.py --familia gps_loss --semilla 20010
Genera:
    results/figures/v23_demo_<familia>_<semilla>.png
    results/v23_demo_<familia>_<semilla>.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from neuroflybee.config import load_config  # noqa: E402
from neuroflybee.control import classical  # noqa: E402,F401  (registra los clásicos)
from neuroflybee.control import registry  # noqa: E402
from neuroflybee.control.neural_registry import register_neural_controllers  # noqa: E402
from neuroflybee.sim.env import NavigationEnv  # noqa: E402
from neuroflybee.sim.rollout import Trajectory, rollout  # noqa: E402
from neuroflybee.sim.scenarios import FAMILIES, FAMILY_LABELS_ES, ScenarioGenerator, Split, split_of  # noqa: E402
from neuroflybee.viz.export import CONTROLLER_STYLE  # noqa: E402
from neuroflybee.viz.style import apply_style  # noqa: E402
from neuroflybee.viz.trajectory_plot import draw_trajectory, outcome_text, path_length  # noqa: E402
from neuroflybee.viz.world_plot import draw_world  # noqa: E402

CLASSICAL = ("astar", "astar_alv")
NEURAL = ("fly", "bee", "hybrid")


@dataclass(frozen=True)
class DemoRun:
    """Desenlace de un controlador en el escenario de la demostración."""

    controller: str
    success: bool
    collided: bool
    landed: bool
    home_error: float
    coverage: float
    steps: int
    path_length: float
    mean_latency_ms: float


def fly(cfg, scenario, name: str) -> tuple[DemoRun, Trajectory]:
    env = NavigationEnv(cfg, scenario)
    traj = rollout(env, registry.create(name))
    latency = [s.latency_ms for s in traj.steps[1:]]
    run = DemoRun(name, env.tracker.success(env.state.position, traj.collided, traj.landed), traj.collided,
                  traj.landed, env.tracker.home_error(env.state.position), env.tracker.coverage,
                  len(traj.steps) - 1, path_length(traj.positions), float(np.mean(latency)) if latency else 0.0)
    return run, traj


def label(name: str) -> str:
    return CONTROLLER_STYLE.get(name, (name,))[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--familia", default="combined", choices=FAMILIES)
    parser.add_argument("--semilla", type=int, default=20_003, help="semilla de la partición de PRUEBA")
    parser.add_argument("--semilla-entrenamiento", type=int, default=0, help="qué pesos usar (0, 1 o 2)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args(argv)
    if split_of(args.semilla) != Split.TEST:
        parser.error("la demostración sólo usa semillas de prueba (20000–20999): nunca se entrenó con ellas")

    apply_style()
    cfg = load_config(args.config)
    neural = [n for n in register_neural_controllers(args.checkpoints, seed=args.semilla_entrenamiento)
              if n in NEURAL]
    names = [*CLASSICAL, *neural]
    scenario = ScenarioGenerator(cfg).generate(args.familia, args.semilla)

    t0 = time.time()
    runs = [fly(cfg, scenario, n) for n in names]
    fig, axes = plt.subplots(1, len(runs), figsize=(max(4.6 * len(runs), 10), 5.6), squeeze=False)
    for ax, (run, traj) in zip(axes[0], runs):
        draw_world(ax, scenario.world, gps_loss=scenario.gps_loss_on_return,
                   title=f"{label(run.controller)}\n{'✓ misión completa' if run.success else '✗ no completa'}")
        draw_trajectory(ax, traj)
        ax.set_xlabel(outcome_text(traj, run.home_error, run.coverage), fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    status = "" if neural else "\n(sin pesos o sin PyTorch: sólo controladores clásicos)"
    fig.suptitle(f"v23 — Demostración: {FAMILY_LABELS_ES[args.familia]}, semilla de prueba {args.semilla}{status}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    stem = f"v23_demo_{args.familia}_{args.semilla}"
    results = Path(args.results)
    out_png = results / "figures" / f"{stem}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    with open(results / f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["controlador", "exito", "choque", "aterrizo", "error_retorno_m", "cobertura", "pasos",
                    "longitud_m", "latencia_ms"])
        for run, _ in runs:
            w.writerow([run.controller, int(run.success), int(run.collided), int(run.landed), f"{run.home_error:.2f}",
                        f"{run.coverage:.2f}", run.steps, f"{run.path_length:.1f}", f"{run.mean_latency_ms:.2f}"])
    print(f"{len(runs)} vuelos en {time.time() - t0:.1f} s → {out_png}")
    for run, _ in runs:
        print(f"  {label(run.controller):28s} {'✓' if run.success else '✗'}  choque={run.collided!s:5s} "
              f"error de retorno {run.home_error:5.1f} m  latencia {run.mean_latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
