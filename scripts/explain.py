"""Interpretabilidad y análisis de errores (v15).

Siempre (sin PyTorch):
    results/figures/v15_analisis_errores.png   causas de fallo por controlador y condición
    results/v15_errores.csv
Con PyTorch y pesos en checkpoints/:
    results/figures/v15_gradcam_mosca.png      qué zona del ojo provocó cada maniobra
    results/figures/v15_ig_abeja.png           qué direcciones del horizonte usó la abeja + fidelidad
    results/v15_fidelidad_abeja.csv

Uso:
    python scripts/explain.py [--episodes 30]
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
from neuroflybee.control import classical  # noqa: E402,F401
from neuroflybee.control import registry  # noqa: E402
from neuroflybee.control.neural_registry import register_neural_controllers  # noqa: E402
from neuroflybee.evaluation.benchmark import FAMILY_SHORT_ES  # noqa: E402
from neuroflybee.evaluation.error_analysis import CATEGORIES, CATEGORY_COLORS, category_shares, diagnose  # noqa: E402
from neuroflybee.sim.scenarios import FAMILIES, seeds_for  # noqa: E402
from neuroflybee.viz.export import CONTROLLER_STYLE  # noqa: E402
from neuroflybee.viz.style import apply_style  # noqa: E402


# ------------------------------------------------------------------------- errores
def error_analysis(cfg, controllers: list[str], episodes: int, out_png: Path, out_csv: Path) -> None:
    seeds = seeds_for("test", episodes)
    shares: dict[str, dict[str, dict[str, float]]] = {}
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["controlador", "condicion", "semilla", "categoria", "x", "y", "paso"])
        for name in controllers:
            diags = diagnose(cfg, lambda n=name: registry.create(n), list(FAMILIES), seeds)
            for d in diags:
                w.writerow([name, d.family, d.seed, d.category, f"{d.position[0]:.2f}", f"{d.position[1]:.2f}", d.step])
            shares[name] = {f: category_shares([d for d in diags if d.family == f]) for f in FAMILIES}
            shares[name]["total"] = category_shares(diags)
            print(f"  {name:20s} " + ", ".join(f"{c}: {v:.0%}" for c, v in shares[name]["total"].items() if v > 0))

    cols = ["total", *FAMILIES]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols) + 2, 5.2), sharey=True)
    for ax, col in zip(axes, cols):
        bottom = np.zeros(len(controllers))
        for cat, color in zip(CATEGORIES, CATEGORY_COLORS):
            vals = np.array([shares[n][col][cat] for n in controllers])
            ax.barh(range(len(controllers)), vals, left=bottom, color=color,
                    label=cat if col == "total" else None)
            bottom += vals
        ax.set_title("TOTAL" if col == "total" else FAMILY_SHORT_ES[col], fontsize=10,
                     fontweight="bold" if col == "total" else "normal")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xticks([0, 0.5, 1])
    axes[0].set_yticks(range(len(controllers)))
    axes[0].set_yticklabels([CONTROLLER_STYLE.get(n, (n,))[0] for n in controllers])
    axes[0].invert_yaxis()
    fig.legend(loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"v15 — ¿Por qué falla cada controlador? ({episodes} escenarios de prueba por condición)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out_png)
    plt.close(fig)


# ------------------------------------------------------------------------- Grad-CAM
def gradcam_figure(cfg, checkpoints: Path, out: Path, n_examples: int = 4) -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    from neuroflybee.control.neural_registry import find_fly_checkpoint
    from neuroflybee.data.generate import collect_fly_episodes
    from neuroflybee.domain.drone import ACTION_LABELS_ES, Action
    from neuroflybee.evaluation.metrics import sequence_windows
    from neuroflybee.interpret.gradcam import grad_cam
    from neuroflybee.training.checkpoints import load_fly_checkpoint

    path = find_fly_checkpoint(checkpoints, 0)
    if path is None:
        return False
    model, _ = load_fly_checkpoint(path)
    data = collect_fly_episodes(cfg, "val", 8, families=("dynamic", "unknown_obstacles"), noise_eps=0.0)
    windows = sequence_windows(data.episode, model.cfg.seq_len)
    rng = np.random.default_rng(0)
    turning = np.where((data.actions != Action.FORWARD) & (data.patches[:, :20].sum(axis=(1, 2)) > 25))[0]
    picks = rng.choice(turning, size=min(n_examples, len(turning)), replace=False)
    T = model.cfg.seq_len
    fig, axes = plt.subplots(len(picks), T + 1, figsize=(2.3 * (T + 1), 2.5 * len(picks)), squeeze=False)
    e = cfg.sensors
    extent = (-e.patch_half_width, e.patch_half_width, -e.patch_back, e.patch_forward)
    for row, i in enumerate(picks):
        frames = torch.from_numpy(data.patches[windows[i]].astype(np.float32))
        cam, action, probs = grad_cam(model, frames, torch.from_numpy(data.goals[i]))
        for k in range(T):
            ax = axes[row, k]
            ax.imshow(frames[k], cmap="Greys", vmin=0, vmax=1, extent=extent)
            ax.imshow(cam[k], cmap="jet", alpha=0.45, vmin=0, vmax=1, extent=extent)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t{k - T + 1:+d}" if k < T - 1 else "t (ahora)", fontsize=9)
        ax = axes[row, T]
        ax.barh(range(len(probs)), probs.numpy(), color=["#c0392b" if j == action else "#95a5a6" for j in range(len(probs))])
        ax.set_yticks(range(len(probs)))
        ax.set_yticklabels([ACTION_LABELS_ES[a] for a in Action], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_title(f"red: {ACTION_LABELS_ES[Action(action)]}\nexperto: {ACTION_LABELS_ES[Action(int(data.actions[i]))]}",
                     fontsize=8)
    fig.suptitle(f"v15 — Grad-CAM de la mosca ({path.name}): rojo = zona que más pesó en la maniobra elegida",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


# ------------------------------------------------------------------------- Integrated Gradients
def ig_figure(cfg, checkpoints: Path, out_png: Path, out_csv: Path, n_views: int = 200) -> bool:
    try:
        import torch
    except ImportError:
        return False
    from neuroflybee.control.homing import panorama_angles
    from neuroflybee.control.neural_registry import find_bee_checkpoint
    from neuroflybee.data.generate import collect_bee_samples
    from neuroflybee.interpret.integrated_gradients import deletion_test, integrated_gradients
    from neuroflybee.training.bee import load_bee_checkpoint

    path = find_bee_checkpoint(checkpoints, 0)
    if path is None:
        return False
    model, _ = load_bee_checkpoint(path)
    data = collect_bee_samples(cfg, "val", max(1, n_views // 20), per_scenario=20, seed=3)
    fractions = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
    rng = np.random.default_rng(0)
    tops, rands, completeness = [], [], []
    for j in range(len(data)):
        attr, fx, fb = integrated_gradients(model, torch.from_numpy(data.current[j]), torch.from_numpy(data.home[j]))
        completeness.append(abs(attr.sum() - (fx - fb)) / max(abs(fx - fb), 1e-6))
        t, r = deletion_test(model, data.current[j], data.home[j], attr, fractions, rng)
        tops.append(t); rands.append(r)
    tops, rands = np.array(tops), np.array(rands)

    deg = np.degrees(panorama_angles(data.current.shape[1]))
    attr0, _, _ = integrated_gradients(model, torch.from_numpy(data.current[0]), torch.from_numpy(data.home[0]))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.fill_between(deg, 0, data.current[0] * 90, step="mid", color="#f1c40f", alpha=0.5, label="vista actual")
    ax.step(deg, data.home[0] * 90, where="mid", color="#1abc9c", lw=1.5, label="memorizada en la base")
    ax2 = ax.twinx()
    ax2.bar(deg, attr0, width=4.5, color=np.where(attr0 > 0, "#c0392b", "#2980b9"), alpha=0.7)
    ax2.set_ylabel("atribución IG")
    true = np.degrees(np.arctan2(data.target[0, 1], data.target[0, 0]))
    ax.axvline(true, color="black", ls="--", lw=1, label="dirección real a la base")
    ax.set_xlabel("dirección (°, brújula)")
    ax.set_ylabel("elevación (°)")
    ax.set_xlim(-180, 180)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("¿Qué direcciones del horizonte usó la abeja? (rojo: a favor, azul: en contra)")
    ax = axes[1]
    for vals, color, label in ((tops, "#c0392b", "borrar las más atribuidas (IG)"), (rands, "#7f8c8d", "borrar al azar")):
        ax.errorbar(fractions * 100, vals.mean(0), yerr=vals.std(0) / np.sqrt(len(vals)), marker="o", color=color,
                    capsize=3, label=label)
    ax.set_xlabel("% de direcciones reemplazadas por la vista de la base")
    ax.set_ylabel("cambio de la dirección predicha (°)")
    ax.set_title(f"Fidelidad por borrado ({len(data)} vistas de validación)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"v15 — Integrated Gradients de la abeja ({path.name})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fraccion_borrada", "cambio_ig_medio_grados", "cambio_azar_medio_grados", "vistas"])
        for k, f in enumerate(fractions):
            w.writerow([f, f"{tops[:, k].mean():.2f}", f"{rands[:, k].mean():.2f}", len(data)])
        w.writerow(["completitud_error_relativo_mediano", f"{np.median(completeness):.4f}", "", ""])
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    apply_style()
    cfg = load_config(args.config)
    results = Path(args.results)
    figs = results / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    checkpoints = Path(args.checkpoints)
    neural = [n for n in register_neural_controllers(checkpoints) if n in ("fly", "bee", "hybrid")]
    controllers = ["potential_field", "astar", "astar_alv", "astar_supervised", *neural]
    print("== Análisis de errores ==")
    error_analysis(cfg, controllers, args.episodes, figs / "v15_analisis_errores.png", results / "v15_errores.csv")
    print("== Grad-CAM (mosca) ==", "listo" if gradcam_figure(cfg, checkpoints, figs / "v15_gradcam_mosca.png")
          else "requiere PyTorch y pesos de la mosca")
    print("== Integrated Gradients (abeja) ==", "listo" if ig_figure(
        cfg, checkpoints, figs / "v15_ig_abeja.png", results / "v15_fidelidad_abeja.csv")
        else "requiere PyTorch y pesos de la abeja")


if __name__ == "__main__":
    main()
