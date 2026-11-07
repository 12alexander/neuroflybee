"""Figuras de la v7: datos de entrenamiento y calidad del experto.

Uso:
    python scripts/show_dataset.py            # usa data/generated/ si existe; si no, genera una muestra
Genera:
    results/figures/v7_dataset_mosca.png   secuencias del ojo + acción del experto, distribución
                                            de acciones y éxito del experto por condición
    results/figures/v7_dataset_abeja.png   muestras alrededor de la base con su vector de retorno
    results/v7_experto.csv                 éxito del experto por condición (semillas de validación)
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
from neuroflybee.control.navigation import MissionNavigator  # noqa: E402
from neuroflybee.data.expert import PrivilegedExpert  # noqa: E402
from neuroflybee.data.generate import (GOAL_SCALE, BeeDataset, FlyDataset,  # noqa: E402
                                       collect_bee_samples, collect_fly_episodes)
from neuroflybee.domain.drone import ACTION_LABELS_ES, Action  # noqa: E402
from neuroflybee.sim.env import NavigationEnv  # noqa: E402
from neuroflybee.sim.scenarios import FAMILIES, FAMILY_LABELS_ES, ScenarioGenerator, seeds_for  # noqa: E402
from neuroflybee.viz.dataset_plot import draw_action_distribution, draw_eye_frame  # noqa: E402
from neuroflybee.viz.style import apply_style  # noqa: E402
from neuroflybee.viz.world_plot import draw_world  # noqa: E402

SEQ = 4   # cuadros por secuencia (los que verá la GRU)


def load_or_make(cfg, data_dir: Path, quick: bool) -> tuple[FlyDataset, FlyDataset, BeeDataset]:
    if not quick and (data_dir / "fly_train.npz").exists():
        return (FlyDataset.load(data_dir / "fly_train.npz"), FlyDataset.load(data_dir / "fly_val.npz"),
                BeeDataset.load(data_dir / "bee_train.npz"))
    print("data/generated no existe: se genera una muestra pequeña (use scripts/generate_data.py)")
    return (collect_fly_episodes(cfg, "train", 8), collect_fly_episodes(cfg, "val", 4),
            collect_bee_samples(cfg, "train", 20, per_scenario=32))


def expert_success(cfg, n: int) -> dict[str, float]:
    gen, out = ScenarioGenerator(cfg), {}
    for family in FAMILIES:
        ok = []
        for seed in seeds_for("val", n):
            env = NavigationEnv(cfg, gen.generate(family, seed))
            obs = env.reset()
            expert, nav = PrivilegedExpert(env.world, cfg), MissionNavigator()
            nav.reset(obs)
            done = False
            while not done:
                target = nav.update(obs, env.state.position)
                result = env.step(expert.act(env.state, target.goal, env.t), target.land)
                obs, done = result.observation, result.done
            ok.append(env.tracker.success(env.state.position, env.collided, env.landed))
        out[family] = float(np.mean(ok))
    return out


def pick_sequences(ds: FlyDataset, wanted: list[Action], seed: int = 3) -> list[int]:
    """Índices finales de secuencias completas (SEQ cuadros del mismo episodio) con
    obstáculos a la vista y la acción pedida."""
    rng = np.random.default_rng(seed)
    chosen = []
    for action in wanted:
        cand = np.where((ds.actions == action) & (ds.t >= SEQ))[0]
        cand = [i for i in cand if ds.episode[i - SEQ + 1] == ds.episode[i]
                and ds.patches[i, :20].sum() > 25]
        if cand:
            chosen.append(int(rng.choice(cand)))
    return chosen


def plot_fly(cfg, train: FlyDataset, val: FlyDataset, success: dict[str, float], out: Path) -> None:
    s = cfg.sensors
    idx = pick_sequences(val if len(val) > 500 else train,
                         [Action.FORWARD, Action.LEFT, Action.SHARP_RIGHT, Action.SHARP_LEFT])
    src = val if len(val) > 500 else train
    fig = plt.figure(figsize=(16, 9.5))
    grid = fig.add_gridspec(len(idx), SEQ + 2, width_ratios=[1] * SEQ + [0.9, 3.0], hspace=0.45, wspace=0.08)
    for row, i in enumerate(idx):
        for k in range(SEQ):
            j = i - SEQ + 1 + k
            ax = fig.add_subplot(grid[row, k])
            draw_eye_frame(ax, src.patches[j], s.patch_forward, s.patch_back, s.patch_half_width,
                           goal=src.goals[j] if k == SEQ - 1 else None, goal_scale=GOAL_SCALE)
            if row == 0:
                ax.set_title(f"t{k - SEQ + 1:+d}" if k < SEQ - 1 else "t (ahora)", fontsize=9)
            if k == 0:
                ax.set_ylabel(FAMILY_LABELS_ES[list(FAMILIES)[src.family[i]]], fontsize=7)
            if k == SEQ - 1:
                ax.set_xlabel(f"experto → {ACTION_LABELS_ES[Action(src.actions[i])]}", fontsize=9,
                              fontweight="bold", color="#c0392b")
    right = grid[:, SEQ + 1].subgridspec(2, 1, hspace=0.55)
    ax_dist = fig.add_subplot(right[0])
    draw_action_distribution(ax_dist, {f"entrenamiento ({len(train):,})": train.action_counts(),
                                       f"validación ({len(val):,})": val.action_counts()})
    ax_dist.set_title("Distribución de acciones del experto\n(desbalanceada → se reporta F1 macro)")
    ax_exp = fig.add_subplot(right[1])
    names = list(success)
    ax_exp.barh([FAMILY_LABELS_ES[n] for n in names], [success[n] for n in names], color="#30a14e")
    ax_exp.set_xlim(0, 1.05)
    ax_exp.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax_exp.set_title("Éxito del experto (maestro) por condición")
    ax_exp.invert_yaxis()
    fig.suptitle("v7 — Datos de la mosca: secuencias del ojo (flecha verde = objetivo) y la acción que enseña el experto",
                 fontsize=12, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_bee(cfg, bee: BeeDataset, out: Path) -> None:
    sample = collect_bee_samples(cfg, "val", 1, per_scenario=150, seed=1)
    world = ScenarioGenerator(cfg).generate("nominal", seeds_for("val", 1)[0]).world
    # reconstruir posiciones: base + distancia * (-dirección de retorno)
    dist = np.expm1(sample.target[:, 2])
    pos = world.base - sample.target[:, :2] * dist[:, None]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.2, 1]})
    draw_world(axes[0], world, title="Vistas de entrenamiento alrededor de la base\n(flecha = dirección real de regreso)")
    sc = axes[0].scatter(pos[:, 0], pos[:, 1], c=dist, cmap="viridis", s=14, zorder=6)
    axes[0].quiver(pos[::3, 0], pos[::3, 1], sample.target[::3, 0], sample.target[::3, 1], color="#c0392b",
                   scale=30, width=0.004, zorder=7)
    fig.colorbar(sc, ax=axes[0], label="distancia a la base (m)", shrink=0.8)
    all_dist = np.expm1(bee.target[:, 2])
    axes[1].hist(all_dist, bins=30, color="#f1c40f", edgecolor="#b8860b")
    axes[1].set_xlabel("distancia a la base (m)")
    axes[1].set_ylabel("muestras")
    axes[1].set_title(f"Conjunto de la abeja: {len(bee):,} pares (vista actual, vista memorizada)")
    fig.suptitle("v7 — Datos de la abeja: aprender a volver a casa sólo con el panorama",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "generated"))
    parser.add_argument("--expert-episodes", type=int, default=12)
    parser.add_argument("--quick", action="store_true", help="no leer data/generated (muestra pequeña)")
    parser.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = parser.parse_args(argv)

    apply_style()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, val, bee = load_or_make(cfg, Path(args.data), args.quick)
    success = expert_success(cfg, args.expert_episodes)
    with open(out.parent / "v7_experto.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condicion", "episodios", "exito_experto"])
        w.writerows([[f, args.expert_episodes, f"{v:.3f}"] for f, v in success.items()])
    plot_fly(cfg, train, val, success, out / "v7_dataset_mosca.png")
    plot_bee(cfg, bee, out / "v7_dataset_abeja.png")
    print(f"Figuras guardadas en {out}")


if __name__ == "__main__":
    main()
