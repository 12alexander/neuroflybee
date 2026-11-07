"""Diagramas de arquitectura dibujados con matplotlib a partir de `models.spec`."""
from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..models.spec import ModelDiagram

KIND_COLORS = {
    "input": "#dfe6e9",
    "cnn": "#a9cce3",
    "temporal": "#f5b7b1",
    "goal": "#abebc6",
    "head": "#f9e79f",
    "output": "#d7bde2",
}
BOX_W, BOX_H, GAP = 2.3, 1.35, 0.55


def _box(ax: Axes, x: float, y: float, block, color: str) -> None:
    ax.add_patch(FancyBboxPatch((x, y), BOX_W, BOX_H, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=color, ec="#555555", lw=1))
    ax.text(x + BOX_W / 2, y + BOX_H - 0.12, block.title, ha="center", va="top", fontsize=7.2, fontweight="bold")
    ax.text(x + BOX_W / 2, y + 0.08, block.shape + (f"\n{block.params:,} parám." if block.params else ""),
            ha="center", va="bottom", fontsize=6.3, style="italic", color="#333333")


def _arrow(ax: Axes, start, end) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color="#444444", lw=1))


def draw_model(ax: Axes, diagram: ModelDiagram, y: float, label: str, total: int) -> float:
    """Dibuja una red en la fila `y`; devuelve el ancho usado."""
    chain = [b for b in diagram.blocks if b.kind != "goal"]
    goal = next(b for b in diagram.blocks if b.kind == "goal")
    ax.text(-0.3, y + BOX_H / 2, f"{label}\n{total:,} parámetros", ha="right", va="center",
            fontsize=8.5, fontweight="bold")
    x = 0.0
    for i, block in enumerate(chain):
        if block.kind == "head":
            cx = x + 0.25
            ax.text(cx, y + BOX_H / 2, "⊕", ha="center", va="center", fontsize=16)
            _box(ax, cx - BOX_W / 2, y - BOX_H - 0.35, goal, KIND_COLORS["goal"])
            _arrow(ax, (cx, y - 0.35), (cx, y + BOX_H / 2 - 0.25))
            _arrow(ax, (cx + 0.25, y + BOX_H / 2), (cx + 0.25 + GAP, y + BOX_H / 2))
            x = cx + 0.25 + GAP
        _box(ax, x, y, block, KIND_COLORS[block.kind])
        if i < len(chain) - 1:
            nxt = x + BOX_W + (0.05 if chain[i + 1].kind == "head" else GAP)
            if chain[i + 1].kind != "head":
                _arrow(ax, (x + BOX_W, y + BOX_H / 2), (nxt, y + BOX_H / 2))
            else:
                _arrow(ax, (x + BOX_W, y + BOX_H / 2), (x + BOX_W + 0.2, y + BOX_H / 2))
            x = nxt
    return x + BOX_W
