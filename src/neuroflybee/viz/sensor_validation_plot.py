"""Figura de la validación del sensor panorámico (v21)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..config import ProjectConfig
from ..evaluation.metrics import angular_errors_deg
from ..evaluation.sensor_validation import (ScenarioValidation, SensorValidationConfig, ValidationSummary,
                                            alv_field, idf_curve, ring_points, rotational_idf)
from ..sim.scenarios import ScenarioGenerator
from ..sim.sensors import PanoramaSensor
from .style import apply_style
from .world_plot import draw_world

ROTATION_COLORS = ("#00798c", "#edae49", "#d1495b", "#5b6770")


def _panel_idf(ax, rows: Sequence[ScenarioValidation], summary: ValidationSummary, arrive_rms: float) -> None:
    curve = idf_curve(rows)
    r = curve["radii"]
    ax.plot(r, curve["clean_mean"], marker="o", color="#1f4e94", lw=2, label="sin ruido (geometría)")
    ax.fill_between(r, curve["clean_mean"] - curve["clean_sd"], curve["clean_mean"] + curve["clean_sd"],
                    color="#1f4e94", alpha=0.15)
    ax.plot(r, curve["noisy_mean"], marker="s", color="#c0392b", lw=1.5, ls="--", label="con el ruido del sensor")
    ax.axhline(arrive_rms, color="grey", ls=":", lw=1.2, label=f"umbral de llegada del ALV ({arrive_rms})")
    if np.isfinite(summary.arrival_radius_m):
        ax.axvline(summary.arrival_radius_m, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("distancia a la base (m)")
    ax.set_ylabel("diferencia RMS con la vista de casa")
    ax.set_title(f"(a) IDF traslacional: crece con la distancia\n"
                 f"(ρ de Spearman medio = {summary.spearman_mean:.2f}, {summary.scenarios} escenarios de prueba)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")


def _panel_rotation(ax, cfg: ProjectConfig, vcfg: SensorValidationConfig, family: str, seed: int,
                    summary: ValidationSummary) -> None:
    world = ScenarioGenerator(cfg).generate(family, seed).world
    sensor = PanoramaSensor(cfg.sensors)
    objects = sensor.objects(world)
    home = sensor.view_from(objects, world.base)
    bins = len(home)
    shifts = (np.arange(bins) + bins // 2) % bins - bins // 2          # −180°…+175°
    order = np.argsort(shifts)
    for r, color in zip((0.0, *vcfg.rotation_radii), ROTATION_COLORS):
        pts = ring_points(world, world.base, r, 8, vcfg.min_clearance)
        if not len(pts):
            continue
        curve = rotational_idf(sensor.view_from(objects, pts[0]), home)
        ax.plot(shifts[order] * 360 / bins, curve[order], color=color, lw=1.8,
                label=f"a {r:g} m de la base" if r else "en la base")
    errs = ", ".join(f"{k} m: {v:.0f}°" for k, v in summary.heading_error.items())
    ax.axvline(0, color="grey", ls=":", lw=1)
    ax.set_xlabel("giro aplicado al panorama (°)")
    ax.set_ylabel("diferencia RMS con la vista de casa")
    ax.set_title(f"(b) IDF rotacional: mínimo en el rumbo correcto\n(error mediano al recuperar el rumbo — {errs})")
    ax.set_xlim(-180, 180)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def _panel_alv(fig, ax, cfg: ProjectConfig, vcfg: SensorValidationConfig, family: str, seed: int,
               summary: ValidationSummary) -> None:
    world = ScenarioGenerator(cfg).generate(family, seed).world
    sensor = PanoramaSensor(cfg.sensors)
    pts, est, true = alv_field(sensor, sensor.objects(world), world, vcfg.field_step, vcfg.field_radius,
                               vcfg.min_clearance)
    err = angular_errors_deg(est, true)
    draw_world(ax, world, title="")
    q = ax.quiver(pts[:, 0], pts[:, 1], np.cos(est), np.sin(est), err, cmap="RdYlGn_r", clim=(0, 180),
                  scale=24, width=0.006, zorder=8)
    fig.colorbar(q, ax=ax, fraction=0.046, pad=0.02, label="error angular del ALV (°)")
    ax.set_xlim(world.base[0] - vcfg.field_radius - 1, world.base[0] + vcfg.field_radius + 1)
    ax.set_ylim(world.base[1] - vcfg.field_radius - 1, world.base[1] + vcfg.field_radius + 1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"(c) Campo del ALV en un escenario de prueba\n(área de captura media: "
                 f"{summary.alv_catchment_mean:.0%} de los puntos con error < 90°)")


def plot_sensor_validation(cfg: ProjectConfig, vcfg: SensorValidationConfig, rows: Sequence[ScenarioValidation],
                           summary: ValidationSummary, out_png: str | Path) -> Path:
    """Tres paneles: IDF traslacional, IDF rotacional y campo del ALV (ejes y leyendas)."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    example = rows[0]
    _panel_idf(axes[0], rows, summary, vcfg.arrive_rms)
    _panel_rotation(axes[1], cfg, vcfg, example.family, example.seed, summary)
    _panel_alv(fig, axes[2], cfg, vcfg, example.family, example.seed, summary)
    fig.suptitle("v21 — Validación del sensor panorámico 2,5D (silueta del horizonte, 72 sectores)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
