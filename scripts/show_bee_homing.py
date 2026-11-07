"""Figura de la v11: ¿hacia dónde está la base? Dirección real vs ALV clásico vs red de la abeja.

Sin PyTorch o sin `checkpoints/bee_s*.pt` se dibuja sólo la dirección real y el ALV
(la red se agrega automáticamente cuando hay pesos entrenados).

Uso:
    python scripts/show_bee_homing.py
Genera:
    results/figures/v11_retorno_abeja.png
    results/v11_abeja_validacion.csv
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
from neuroflybee.control.homing import alv_directions  # noqa: E402
from neuroflybee.data.generate import BeeDataset, collect_bee_samples  # noqa: E402
from neuroflybee.evaluation.metrics import angular_errors_deg  # noqa: E402
from neuroflybee.sim.scenarios import ScenarioGenerator, seeds_for  # noqa: E402
from neuroflybee.sim.sensors import PanoramaSensor  # noqa: E402
from neuroflybee.viz.style import apply_style  # noqa: E402
from neuroflybee.viz.world_plot import draw_world  # noqa: E402

COLORS = {"real": "#555555", "alv": "#8e44ad", "red": "#d4ac0d"}
BINS = np.array([0, 2, 4, 6, 8, 10, 12.01])


# compatibilidad: el cálculo vive ahora en control/homing.py (v20)
alv_angles = alv_directions


def load_network(checkpoints: Path):
    """(predictor, n_semillas) o (None, 0) si no hay PyTorch o pesos."""
    paths = sorted(checkpoints.glob("bee_s*.pt"))
    if not paths:
        return None, 0
    try:
        import torch

        from neuroflybee.models.bee import BeeHomingNet
        from neuroflybee.training.bee import load_bee_checkpoint
    except ImportError:
        return None, 0
    models = [load_bee_checkpoint(p)[0] for p in paths]

    @torch.no_grad()
    def predict(current: np.ndarray, home: np.ndarray) -> np.ndarray:
        """Ángulo y distancia promediando los modelos de todas las semillas (por semilla en el CSV)."""
        c, h = torch.from_numpy(current).float(), torch.from_numpy(home).float()
        outs = [BeeHomingNet.decode(m(c, h)) for m in models]
        return [(a.numpy(), d.numpy()) for a, d in outs]

    return predict, len(models)


def field_positions(world, n: int = 13, radius: float = 10.0) -> np.ndarray:
    xs = np.linspace(-radius, radius, n)
    pts = np.array([world.base + [x, y] for x in xs for y in xs if 1.0 < np.hypot(x, y) <= radius])
    ok = (pts[:, 0] > 0) & (pts[:, 1] > 0) & (pts[:, 0] < world.width) & (pts[:, 1] < world.height)
    pts = pts[ok]
    return pts[world.clearance(pts) > 0.5]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "generated"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    apply_style()
    cfg = load_config(args.config)
    results = Path(args.results)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    val_path = Path(args.data) / "bee_val.npz"
    val = BeeDataset.load(val_path) if val_path.exists() else collect_bee_samples(cfg, "val", 20, per_scenario=32)
    predict, n_models = load_network(Path(args.checkpoints))

    true_angle = np.arctan2(val.target[:, 1], val.target[:, 0])
    dist = np.expm1(val.target[:, 2])
    errors = {"ALV (clásico)": angular_errors_deg(alv_angles(val.current, val.home), true_angle)}
    per_seed_net = []
    if predict is not None:
        outs = predict(val.current, val.home)
        per_seed_net = [angular_errors_deg(a, true_angle) for a, _ in outs]
        errors[f"Red de la abeja ({n_models} sem.)"] = np.mean(per_seed_net, axis=0)
        dist_mae = [float(np.abs(d - dist).mean()) for _, d in outs]

    # ---------- campo de flechas en un escenario de validación
    world = ScenarioGenerator(cfg).generate("nominal", seeds_for("val", 1)[0]).world
    pano = PanoramaSensor(cfg.sensors)
    objs = pano.objects(world)
    pts = field_positions(world)
    cur = np.array([pano.view_from(objs, p) for p in pts], dtype=np.float32)
    home = np.repeat(pano.view_from(objs, world.base)[None], len(pts), axis=0).astype(np.float32)
    true_dir = np.arctan2(world.base[1] - pts[:, 1], world.base[0] - pts[:, 0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    draw_world(ax, world, title="¿Hacia dónde está la base? (escenario de validación)")
    ax.set_xlim(world.base[0] - 12, world.base[0] + 12)
    ax.set_ylim(world.base[1] - 12, world.base[1] + 12)
    arrows = [("dirección real", true_dir, COLORS["real"]), ("ALV clásico", alv_angles(cur, home), COLORS["alv"])]
    if predict is not None:
        arrows.append(("red de la abeja", predict(cur, home)[0][0], COLORS["red"]))
    for label, ang, color in arrows:
        ax.quiver(pts[:, 0], pts[:, 1], np.cos(ang), np.sin(ang), color=color, scale=22, width=0.005,
                  label=label, zorder=7, alpha=0.9)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    centers = (BINS[:-1] + BINS[1:]) / 2
    for (label, err), color in zip(errors.items(), (COLORS["alv"], COLORS["red"])):
        med = [np.median(err[(dist >= a) & (dist < b)]) for a, b in zip(BINS[:-1], BINS[1:])]
        ax.plot(centers, med, marker="o", lw=2, color=color, label=label)
    ax.axhline(90, ls="--", color="grey", lw=1, label="azar (90°)")
    ax.set_xlabel("distancia a la base (m)")
    ax.set_ylabel("error angular mediano (°)")
    ax.set_ylim(0, 120)
    ax.set_title(f"Error angular en validación ({len(val):,} vistas)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    for (label, err), color in zip(errors.items(), (COLORS["alv"], COLORS["red"])):
        xs = np.sort(err)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), lw=2, color=color, label=label)
    ax.axvline(30, ls=":", color="grey")
    ax.set_xlabel("error angular (°)")
    ax.set_ylabel("fracción de vistas con error ≤ x")
    ax.set_xlim(0, 180)
    ax.set_title("Distribución acumulada del error")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    status = "" if predict is not None else "   (la red se agrega al entrenar: scripts/train_bee.py)"
    fig.suptitle(f"v11 — Retorno visual de la abeja: red siamesa vs ALV clásico{status}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(results / "figures" / "v11_retorno_abeja.png")
    plt.close(fig)

    with open(results / "v11_abeja_validacion.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metodo", "vistas", "error_angular_mediano_grados", "error_angular_medio_grados",
                    "fraccion_error_menor_30", "error_distancia_medio_m"])
        alv = errors["ALV (clásico)"]
        w.writerow(["alv", len(alv), f"{np.median(alv):.2f}", f"{alv.mean():.2f}", f"{(alv <= 30).mean():.3f}", ""])
        for s, e in enumerate(per_seed_net):
            w.writerow([f"red_abeja_s{s}", len(e), f"{np.median(e):.2f}", f"{e.mean():.2f}",
                        f"{(e <= 30).mean():.3f}", f"{dist_mae[s]:.3f}"])
    print(f"ALV: error angular mediano {np.median(errors['ALV (clásico)']):.1f}°"
          + ("" if predict is None else f" | red: {np.median(list(errors.values())[1]):.1f}°"))


if __name__ == "__main__":
    main()
