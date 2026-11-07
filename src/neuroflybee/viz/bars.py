"""Gráficos de barras agrupadas (controlador × condición)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.axes import Axes

CONTROLLER_COLORS = {
    "direct": "#9e9e9e",
    "potential_field": "#e67e22",
    "astar": "#1f4e94",
    "astar_alv": "#8e44ad",
    "fly": "#c0392b",
    "bee": "#f1c40f",
    "hybrid": "#27ae60",
}
CONTROLLER_LABELS_ES = {
    "direct": "Directo (sin evasión)",
    "potential_field": "Campos de potencial",
    "astar": "A* + replanificación",
    "astar_alv": "A* + retorno ALV",
    "fly": "Mosca",
    "bee": "Abeja",
    "hybrid": "Híbrido",
}


def grouped_bars(ax: Axes, groups: Sequence[str], series: dict[str, np.ndarray],
                 errors: Optional[dict[str, np.ndarray]] = None, ylabel: str = "",
                 percent: bool = True) -> None:
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (name, values) in enumerate(series.items()):
        err = None if errors is None else errors.get(name)
        ax.bar(x + (i - (n - 1) / 2) * width, values, width, yerr=err, capsize=2,
               color=CONTROLLER_COLORS.get(name, None), label=CONTROLLER_LABELS_ES.get(name, name))
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if percent:
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", alpha=0.3)
