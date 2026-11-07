"""Figura de la ablación de la abeja (v20): precisión frente a tamaño de cada variante."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..evaluation.bee_ablation import ALV_NAME, VariantSummary
from .style import apply_style

VARIANT_COLORS = {ALV_NAME: "#8e44ad", "completa": "#d4ac0d", "vista_unica": "#7f8c8d",
                  "sin_diferencia": "#e67e22", "compacta": "#27ae60"}
RANDOM_GUESS_DEG = 90.0          # error mediano esperado de una dirección al azar
# Tamaños de referencia de Bee-Nav (Ou et al., 2026), contados desde su código fuente (MIT):
# CompactCNN_rgb y InceptAttentionCNN_rgb (models/model_rgb.py). Sólo como escala: su
# entrada es una imagen RGB de 192×1800, no nuestro panorama de 72 sectores.
BEE_NAV_REFERENCES = {"Bee-Nav compacta (868)": 868, "Bee-Nav Inception (10 820)": 10_820}


def plot_bee_ablation(summary: Sequence[VariantSummary], out_png: str | Path, views: int) -> Path:
    """Dos paneles: (a) error angular mediano con su d.e. entre semillas; (b) parámetros (log).

    Cada barra lleva su valor; la línea punteada marca el azar (90°), que es lo que obtendría
    una red que no sabe hacia dónde está la base."""
    apply_style()
    labels = [s.label for s in summary]
    colors = [VARIANT_COLORS.get(s.variant, "#34495e") for s in summary]
    x = np.arange(len(summary))
    fig, (ax_err, ax_par) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})

    medians = np.array([s.median_mean for s in summary])
    sds = np.array([s.median_sd for s in summary])
    bars = ax_err.bar(x, medians, yerr=sds, capsize=4, color=colors, edgecolor="#333333", linewidth=0.6)
    for bar, value, sd in zip(bars, medians, sds):
        ax_err.annotate(f"{value:.1f}°", (bar.get_x() + bar.get_width() / 2, bar.get_height() + sd),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax_err.axhline(RANDOM_GUESS_DEG, ls="--", color="grey", lw=1, label="dirección al azar (90°)")
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(labels, rotation=15, ha="right")
    ax_err.set_ylabel("error angular mediano en prueba (°)")
    ax_err.set_xlabel("variante")
    ax_err.set_ylim(0, max(100.0, float((medians + sds).max()) * 1.15))
    ax_err.set_title(f"(a) Precisión de la dirección a la base ({views:,} vistas de prueba)")
    ax_err.grid(axis="y", alpha=0.3)
    ax_err.legend(loc="upper right", fontsize=8)

    nets = [i for i, s in enumerate(summary) if s.parameters > 0]
    params = np.array([summary[i].parameters for i in nets])
    pbars = ax_par.bar(np.arange(len(nets)), params, color=[colors[i] for i in nets],
                       edgecolor="#333333", linewidth=0.6)
    for bar, value in zip(pbars, params):
        ax_par.annotate(f"{value:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    for (label, value), style in zip(BEE_NAV_REFERENCES.items(), (":", "-.")):
        ax_par.axhline(value, ls=style, color="#2c3e50", lw=1, label=label)
    ax_par.set_yscale("log")
    ax_par.set_xticks(np.arange(len(nets)))
    ax_par.set_xticklabels([labels[i] for i in nets], rotation=15, ha="right")
    ax_par.set_ylabel("parámetros entrenables (escala log)")
    ax_par.set_xlabel("variante")
    ax_par.set_title("(b) Tamaño de cada red")
    ax_par.grid(axis="y", alpha=0.3, which="both")
    lowest = min([*params.tolist(), *BEE_NAV_REFERENCES.values()])
    ax_par.set_ylim(max(1, lowest / 3), max([*params.tolist(), 1]) * 12)
    ax_par.legend(loc="upper right", fontsize=7, title="referencia publicada", title_fontsize=7)

    fig.suptitle("v20 — Ablación de la red de la abeja: ¿qué parte del diseño importa?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
