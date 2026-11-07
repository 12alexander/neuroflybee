"""Figuras de la v9 (no requiere PyTorch).

Siempre genera:
    results/figures/v9_pesos_clase.png        frecuencia de cada acción y peso en la pérdida
Si existen historiales de entrenamiento (scripts/train_fly.py), además:
    results/figures/v9_curvas_mosca.png       pérdida y F1 macro de validación por época (media ± d.e.)
    results/figures/v9_confusion_mosca.png    matriz de confusión normalizada del modelo propuesto
    results/v9_mosca_validacion.csv           F1 macro y accuracy por modelo (media ± d.e.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from neuroflybee.data.generate import FlyDataset  # noqa: E402
from neuroflybee.domain.drone import ACTION_LABELS_ES, Action  # noqa: E402
from neuroflybee.evaluation.metrics import inverse_frequency_weights  # noqa: E402
from neuroflybee.evaluation.reports import aggregate, best_rows, load_histories, write_summary_csv  # noqa: E402
from neuroflybee.models.spec import FLY_MODEL_LABELS_ES  # noqa: E402
from neuroflybee.training.config import load_train_config  # noqa: E402
from neuroflybee.viz.style import apply_style  # noqa: E402

MODEL_COLORS = {"fly_mlp": "#9e9e9e", "fly_cnn": "#1f4e94", "fly_cnn_gru": "#c0392b",
                "fly_cnn_transformer": "#8e44ad"}
LABELS = [ACTION_LABELS_ES[a] for a in Action]


def plot_class_weights(counts: np.ndarray, tcfg, out: Path) -> None:
    w = inverse_frequency_weights(counts, tcfg.class_weight_power, tcfg.max_class_weight)
    freq = counts / counts.sum()
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(LABELS))
    ax1.bar(x - 0.2, freq, 0.4, color="#1f4e94", label="frecuencia en entrenamiento")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax1.set_ylabel("frecuencia")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, w, 0.4, color="#e67e22", label="peso en la entropía cruzada")
    ax2.set_ylabel("peso de la clase")
    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS, rotation=20, ha="right")
    handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax1.legend(handles, ["frecuencia en entrenamiento", "peso en la entropía cruzada"], loc="upper center")
    ax1.set_title(f"v9 — Acciones desbalanceadas: pesos ∝ frecuencia^-{tcfg.class_weight_power} "
                  f"(máx. {tcfg.max_class_weight:g})")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _curve(histories: list[dict], model: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = [h for h in histories if h["model"] == model]
    n = max(len(h["history"]) for h in runs)
    values = np.full((len(runs), n), np.nan)
    for i, h in enumerate(runs):
        for j, e in enumerate(h["history"]):
            values[i, j] = e["metrics"][key] if key in e["metrics"] else e[key]
    return np.arange(1, n + 1), np.nanmean(values, axis=0), np.nanstd(values, axis=0)


def plot_curves(histories: list[dict], out: Path) -> None:
    models = [m for m in MODEL_COLORS if any(h["model"] == m for h in histories)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for m in models:
        for ax, key, title in zip(axes, ("train_loss", "val_loss", "f1_macro"),
                                  ("Pérdida de entrenamiento", "Pérdida de validación", "F1 macro de validación")):
            x, mean, std = _curve(histories, m, key)
            ax.plot(x, mean, color=MODEL_COLORS[m], lw=2, label=FLY_MODEL_LABELS_ES[m])
            ax.fill_between(x, mean - std, mean + std, color=MODEL_COLORS[m], alpha=0.15)
            ax.set_title(title)
            ax.set_xlabel("época")
            ax.grid(alpha=0.3)
    axes[2].legend(fontsize=8, loc="lower right")
    n_seeds = len({h["seed"] for h in histories})
    fig.suptitle(f"v9 — Entrenamiento por imitación de la mosca (media ± d.e., {n_seeds} semilla(s))",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_confusion(histories: list[dict], model: str, out: Path) -> None:
    runs = [h for h in histories if h["model"] == model] or histories
    cm = np.sum([np.array(h["confusion_val"]) for h in runs], axis=0).astype(float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("acción predicha por la red")
    ax.set_ylabel("acción del experto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="fracción de la fila")
    ax.set_title(f"v9 — Matriz de confusión en validación\n{FLY_MODEL_LABELS_ES.get(runs[0]['model'], model)}")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "data" / "generated"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--train-config", default=str(ROOT / "configs" / "train_fly.yaml"))
    parser.add_argument("--model", default="fly_cnn_gru", help="modelo para la matriz de confusión")
    args = parser.parse_args(argv)

    apply_style()
    results = Path(args.results)
    figs = results / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    tcfg = load_train_config(args.train_config)
    train_path = Path(args.data) / "fly_train.npz"
    if train_path.exists():
        plot_class_weights(FlyDataset.load(train_path).action_counts().astype(float), tcfg,
                           figs / "v9_pesos_clase.png")
    else:
        print("Sin data/generated: no se genera v9_pesos_clase.png (ejecute scripts/generate_data.py)")
    histories = load_histories(results, "v9")
    if not histories:
        print("Aún no hay historiales: entrene con  python scripts/train_fly.py  (requiere PyTorch)")
        return
    plot_curves(histories, figs / "v9_curvas_mosca.png")
    plot_confusion(histories, args.model, figs / "v9_confusion_mosca.png")
    summary = aggregate(best_rows(histories))
    write_summary_csv(results / "v9_mosca_validacion.csv", summary)
    for m, s in summary.items():
        print(f"{m:22s} F1 macro {s['f1_mean']:.3f} ± {s['f1_std']:.3f}  ({s['n_seeds']} semillas)")


if __name__ == "__main__":
    main()
