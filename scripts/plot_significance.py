"""Figura de la v22: comparaciones pareadas con su IC 95 % (no requiere PyTorch).

Lee results/v22_significancia.csv (lo escribe scripts/significance.py) y genera
    results/figures/v22_significancia.png
Cada fila es una comparación «A frente a B» en una condición: el punto es la diferencia
de éxito en puntos porcentuales y la barra su IC 95 % por bootstrap pareado. En verde las
que siguen siendo significativas tras la corrección de Holm; en gris las demás.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from neuroflybee.viz.style import apply_style  # noqa: E402

CONDITION_LABELS = {
    "total": "total",
    "nominal": "nominal",
    "unknown_obstacles": "no previstos",
    "dynamic": "móviles",
    "wind": "viento",
    "gps_loss": "sin GPS",
    "combined": "combinado",
}
SIGNIFICANT = "#30a14e"
NOT_SIGNIFICANT = "#8a9399"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def short_name(name: str) -> str:
    return name.replace(" (clásico)", "")


def plot_significance(rows: list[dict], out: Path) -> None:
    apply_style()
    rows = list(reversed(rows))          # la primera comparación queda arriba
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.55 * len(rows)))
    for y, r in enumerate(rows):
        diff, lo, hi = (100 * float(r[k]) for k in ("diferencia", "ic_inf", "ic_sup"))
        color = SIGNIFICANT if r["significativa"] == "1" else NOT_SIGNIFICANT
        ax.plot([lo, hi], [y, y], color=color, lw=2.5, solid_capstyle="round")
        ax.plot(diff, y, "o", color=color, ms=8, mec="white", mew=1.5)
        ax.text(hi + 1.5, y, f"{diff:+.0f} pp  ·  p Holm {float(r['p_holm']):.3f}  ·  h {float(r['h_cohen']):.2f}",
                va="center", fontsize=8, color="#333333")
    labels = [f"{short_name(r['a'])} vs {short_name(r['b'])} — "
              f"{CONDITION_LABELS.get(r['condicion'], r['condicion'])}" for r in rows]
    ax.set_yticks(range(len(rows)), labels)
    ax.axvline(0, color="#444444", lw=1, ls="--")
    right = max(100 * float(r["ic_sup"]) for r in rows)
    ax.set_xlim(min(-5, min(100 * float(r["ic_inf"]) for r in rows) - 3), right + 38)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Diferencia de éxito A − B (puntos porcentuales), IC 95 % bootstrap pareado")
    ax.grid(axis="x", color="#dddddd", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.plot([], [], "o-", color=SIGNIFICANT, label="significativa tras Holm")
    ax.plot([], [], "o-", color=NOT_SIGNIFICANT, label="no significativa")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(f"v22 — Comparaciones pareadas en los mismos escenarios de prueba\n"
                 f"(corrección de Holm sobre {len(rows)} pruebas)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(ROOT / "results" / "v22_significancia.csv"))
    parser.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = parser.parse_args(argv)

    path = Path(args.csv)
    rows = load_rows(path) if path.exists() else []
    if not rows:
        print(f"No hay comparaciones en {path}: ejecute primero scripts/significance.py")
        return
    out = Path(args.out) / "v22_significancia.png"
    plot_significance(rows, out)
    print(f"Figura: {out}")


if __name__ == "__main__":
    main()
