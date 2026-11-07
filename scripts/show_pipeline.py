"""Diagrama del pipeline de resultados (v19), generado desde las etapas definidas en el código.

    python scripts/show_pipeline.py   → results/figures/v19_pipeline.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from neuroflybee.pipeline.stages import default_stages  # noqa: E402

PRODUCES = {"pruebas": "0 pruebas\nsaltadas", "datos": "datos\n*.npz", "mosca": "fly_*.pt\nF1 macro",
            "dagger": "*_dagger*.pt\néxito por\niteración", "abeja": "bee_*.pt\nerror angular", "benchmark": "Tabla principal\n+ ablaciones",
            "figuras": "results/figures/\nvisor", "interpretabilidad": "Grad-CAM, IG,\ncausas de error",
            "video": "v16_demo.mp4", "entorno": "requirements-\nlock.txt", "pesos": "MANIFEST.json\n+ .zip (enlace)",
            "numeros": "paper/generado/\nnumeros.tex"}


def main() -> None:
    stages = default_stages()
    fig, ax = plt.subplots(figsize=(19, 5.4))
    ax.set_xlim(0, len(stages)); ax.set_ylim(0, 5.4); ax.axis("off")
    for i, s in enumerate(stages):
        color = "#e74c3c" if s.uses_gpu else ("#9e9e9e" if s.optional else "#3498db")
        ax.add_patch(FancyBboxPatch((i + 0.08, 2.9), 0.84, 1.3, boxstyle="round,pad=0.02", fc=color, ec="none", alpha=0.9))
        ax.text(i + 0.5, 3.75, f"{i + 1}. {s.name}" if len(s.name) < 11 else f"{i + 1}. {s.name[:9]}-\n{s.name[9:]}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(i + 0.5, 3.12, "GPU" if s.uses_gpu else ("opcional" if s.optional else "CPU"), ha="center", color="white", fontsize=8)
        ax.annotate("", (i + 0.5, 2.2), (i + 0.5, 2.85), arrowprops=dict(arrowstyle="->", color="#555"))
        ax.text(i + 0.5, 1.75, PRODUCES.get(s.name, ""), ha="center", va="center", fontsize=7.6, color="#333")
        if i < len(stages) - 1:
            ax.annotate("", (i + 1.08, 3.55), (i + 0.92, 3.55), arrowprops=dict(arrowstyle="->", color="#555"))
    ax.add_patch(FancyBboxPatch((0.1, 0.25), len(stages) - 0.2, 0.75, boxstyle="round,pad=0.02", fc="#2ecc71", ec="none", alpha=0.25))
    ax.text(len(stages) / 2, 0.62, "results/v19_costo.csv: minutos y horas de GPU por etapa (R6)  ·  "
            "se reanuda si Colab se desconecta  ·  cada número del artículo sale de results/ (collect_results.py)",
            ha="center", va="center", fontsize=9.5)
    ax.set_title("v19 — Un comando en Colab: python scripts/run_pipeline.py  (rojo = usa GPU)", fontsize=13, fontweight="bold", loc="left")
    out = ROOT / "results" / "figures" / "v19_pipeline.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); print(out)


if __name__ == "__main__":
    main()
