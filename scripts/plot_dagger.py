"""Figuras de la v10 (no requiere PyTorch).

Siempre genera:
    results/figures/v10_dagger_esquema.png   el ciclo de DAgger (para el artículo y las diapositivas)
Si existen resultados (scripts/train_dagger.py), además:
    results/figures/v10_dagger_mejora.png    éxito y choques en lazo cerrado por iteración (media ± d.e.)
    results/v10_dagger.csv                   la tabla correspondiente
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from neuroflybee.sim.scenarios import FAMILY_LABELS_ES  # noqa: E402
from neuroflybee.viz.style import apply_style  # noqa: E402


def plot_scheme(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    boxes = {
        "data": (0.4, 3.0, "Datos agregados\nD = D₀ ∪ D₁ ∪ …", "#a9cce3"),
        "train": (4.2, 3.0, "Entrenar la mosca\n(CNN + GRU)", "#f5b7b1"),
        "fly": (8.0, 3.0, "La mosca vuela\n(con prob. 1−β)", "#f9e79f"),
        "expert": (8.0, 0.4, "El experto etiqueta\nlos estados visitados", "#abebc6"),
        "agg": (4.2, 0.4, "Nuevas muestras Dₖ\n(estados de la mosca,\nacción del experto)", "#d7bde2"),
    }
    for x, y, text, color in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), 3.0, 1.5, boxstyle="round,pad=0.08", fc=color, ec="#555555"))
        ax.text(x + 1.5, y + 0.75, text, ha="center", va="center", fontsize=10, fontweight="bold")
    arrows = [((3.4, 3.75), (4.2, 3.75)), ((7.2, 3.75), (8.0, 3.75)), ((9.5, 3.0), (9.5, 1.9)),
              ((8.0, 1.15), (7.2, 1.15)), ((1.9, 1.15), (1.9, 3.0))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color="#333333", lw=1.6))
    ax.plot([4.2, 1.9], [1.15, 1.15], color="#333333", lw=1.6)     # tramo sin punta hasta el codo
    ax.text(6.0, 5.0, "β: 0.5 → 0.25 → 0   (cada vez vuela más la red y menos el experto)", ha="center",
            fontsize=10, style="italic")
    ax.text(6.0, -0.45, "Iteración 0 = imitación pura (v9). La prueba nunca se usa: se evalúa en validación.",
            ha="center", fontsize=9, color="#555555")
    ax.set_xlim(0, 11.6)
    ax.set_ylim(-0.8, 5.4)
    ax.axis("off")
    ax.set_title("v10 — DAgger: aprender a salir de los errores propios (Ross et al., 2011)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def load_reports(results: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(results.glob("v10_dagger_*.json"))]


def plot_improvement(reports: list[dict], out_png: Path, out_csv: Path) -> None:
    n_it = min(len(r["iterations"]) for r in reports)
    its = np.arange(n_it)
    get = lambda key: np.array([[r["iterations"][i]["closed_loop"]["all"][key] for i in its] for r in reports])  # noqa: E731
    success, collision = get("success"), get("collision")
    families = list(reports[0]["iterations"][0]["closed_loop"]["by_family"])
    fam = {f: np.array([[r["iterations"][i]["closed_loop"]["by_family"][f]["success"] for i in its]
                        for r in reports]) for f in families}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for values, color, label in ((success, "#27ae60", "éxito de misión"), (collision, "#c0392b", "choques")):
        axes[0].errorbar(its, values.mean(0), yerr=values.std(0), color=color, marker="o", capsize=4, lw=2,
                         label=label)
    axes[0].set_title("Todas las condiciones de validación")
    for f, values in fam.items():
        axes[1].plot(its, values.mean(0), marker="o", lw=1.8, label=FAMILY_LABELS_ES.get(f, f))
    axes[1].set_title("Éxito por condición")
    for ax in axes:
        ax.set_xticks(its)
        ax.set_xticklabels(["0\n(imitación)"] + [str(i) for i in its[1:]])
        ax.set_xlabel("iteración de DAgger")
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"v10 — DAgger en lazo cerrado ({len(reports)} semilla(s), media ± d.e.)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["iteracion", "beta", "muestras_media", "exito_media", "exito_de", "choques_media",
                    "choques_de", "f1_macro_val_media"])
        for i in its:
            w.writerow([i, reports[0]["iterations"][i]["beta"],
                        int(np.mean([r["iterations"][i]["samples"] for r in reports])),
                        f"{success[:, i].mean():.3f}", f"{success[:, i].std():.3f}",
                        f"{collision[:, i].mean():.3f}", f"{collision[:, i].std():.3f}",
                        f"{np.mean([r['iterations'][i]['val_f1_macro'] for r in reports]):.4f}"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args(argv)
    apply_style()
    results = Path(args.results)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    plot_scheme(results / "figures" / "v10_dagger_esquema.png")
    reports = load_reports(results)
    if not reports:
        print("Aún no hay resultados de DAgger: ejecute  python scripts/train_dagger.py  (requiere PyTorch)")
        return
    plot_improvement(reports, results / "figures" / "v10_dagger_mejora.png", results / "v10_dagger.csv")
    print(f"Figuras en {results / 'figures'}")


if __name__ == "__main__":
    main()
