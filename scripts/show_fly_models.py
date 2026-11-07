"""Figura de la v8: las cuatro redes de la mosca con dimensiones y parámetros.

No necesita PyTorch: usa la especificación de `neuroflybee.models.spec` (la prueba
`tests/test_fly_models.py` verifica que los conteos coinciden con PyTorch).

Uso:
    python scripts/show_fly_models.py
Genera:
    results/figures/v8_redes_mosca.png
    results/v8_parametros.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from neuroflybee.models.spec import (FLY_MODEL_LABELS_ES, FLY_MODEL_NAMES, count_parameters,  # noqa: E402
                                     describe, load_fly_config)
from neuroflybee.viz.architecture_plot import BOX_H, KIND_COLORS, draw_model  # noqa: E402
from neuroflybee.viz.style import apply_style  # noqa: E402

KIND_LABELS = {"input": "entrada", "cnn": "extracción visual", "temporal": "memoria temporal",
               "goal": "objetivo", "head": "decisión", "output": "salida"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "fly_models.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = parser.parse_args(argv)

    apply_style()
    cfg = load_fly_config(args.model_config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    row_h = 2 * BOX_H + 1.2
    fig, ax = plt.subplots(figsize=(17, 2.6 * len(FLY_MODEL_NAMES) + 0.8))
    width = 0.0
    for i, name in enumerate(FLY_MODEL_NAMES):
        y = (len(FLY_MODEL_NAMES) - 1 - i) * row_h + BOX_H + 0.4
        width = max(width, draw_model(ax, describe(name, cfg), y, FLY_MODEL_LABELS_ES[name],
                                      count_parameters(name, cfg)))
    for j, (kind, color) in enumerate(KIND_COLORS.items()):
        ax.add_patch(plt.Rectangle((j * 2.4, -0.9), 0.35, 0.35, fc=color, ec="#555555"))
        ax.text(j * 2.4 + 0.5, -0.72, KIND_LABELS[kind], va="center", fontsize=8)
    ax.set_xlim(-5.2, width + 0.3)
    ax.set_ylim(-1.2, len(FLY_MODEL_NAMES) * row_h + 0.2)
    ax.axis("off")
    ax.set_title(f"v8 — Redes de la mosca: misma entrada (ojo {cfg.patch_size}×{cfg.patch_size}, "
                 f"{cfg.seq_len} cuadros) y misma salida ({cfg.n_actions} acciones); cambia la memoria temporal",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "v8_redes_mosca.png")
    plt.close(fig)

    with open(out.parent / "v8_parametros.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo", "descripcion", "parametros_entrenables", "usa_secuencia"])
        for name in FLY_MODEL_NAMES:
            w.writerow([name, FLY_MODEL_LABELS_ES[name], count_parameters(name, cfg),
                        name in ("fly_cnn_gru", "fly_cnn_transformer")])
    print(f"Figura guardada en {out / 'v8_redes_mosca.png'}")


if __name__ == "__main__":
    main()
