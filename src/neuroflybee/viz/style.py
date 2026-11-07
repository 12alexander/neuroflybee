"""Paleta y estilo comunes para todas las figuras del proyecto."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")   # sin ventana: funciona en Colab, servidores y Windows
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "background": "#f4f1ea",
    "known": "#5b6770",      # obstáculo del mapa previo
    "unknown": "#d1495b",    # obstáculo no previsto
    "moving": "#edae49",     # obstáculo móvil
    "base": "#00798c",
    "waypoint": "#30a14e",
    "route": "#1f4e94",
    "wind": "#6c5ce7",
    "text": "#222222",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#888888",
        "axes.labelcolor": COLORS["text"],
    })
