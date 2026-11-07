"""Ablación de la red de la abeja (v20) — parte sin PyTorch.

Responde a dos preguntas que el jurado hará sobre la red siamesa:

1. **¿Hace falta comparar con la vista de casa?**  Se entrena la misma red viendo sólo la
   vista actual (`vista_unica`) y quitando la resta explícita (`sin_diferencia`).
2. **¿Por qué 172 547 parámetros si Bee-Nav vuela con 868?**  Se reemplaza la cabeza plana
   (148 000 parámetros) por una lectura vectorial de ~150 (`compacta`).

Cada variante cambia UNA decisión, usa los mismos datos, semillas e hiperparámetros, y se
evalúa en la partición de PRUEBA (semillas 20000+), que nunca intervino en elegir épocas. El
ALV clásico se evalúa sobre las mismas vistas como referencia.

Este módulo sólo define las variantes, puntúa predicciones y arma tablas: el entrenamiento
(PyTorch) lo orquesta `scripts/run_bee_ablation.py`.
"""
from __future__ import annotations

import csv
import statistics
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import yaml

from ..control.homing import alv_directions
from ..evaluation.metrics import homing_report
from ..models.bee_spec import BeeNetConfig

ALV_NAME = "alv"


# ----------------------------------------------------------------------------- variantes
@dataclass(frozen=True)
class AblationVariant:
    """Una variante de la red: nombre corto, etiqueta para tablas y cambios respecto de v11.

    `reuse_v11=True` significa que la variante ES el modelo principal: se reutilizan sus
    pesos (`bee_s<semilla>.pt`) en lugar de entrenar otra vez la misma red."""

    name: str
    label: str
    overrides: dict = field(default_factory=dict)
    reuse_v11: bool = False

    def config(self, base: BeeNetConfig) -> BeeNetConfig:
        """Configuración de la red para esta variante (valida los nombres de fusión/cabeza)."""
        return replace(base, **self.overrides)

    def checkpoint_name(self, seed: int) -> str:
        """Nombre del archivo de pesos; las variantes nuevas nunca pisan los de v11."""
        return f"bee_s{seed}.pt" if self.reuse_v11 else f"bee_ablacion_{self.name}_s{seed}.pt"

    def run_name(self, seed: int) -> str:
        return Path(self.checkpoint_name(seed)).stem


@dataclass(frozen=True)
class BeeAblationConfig:
    """Contenido de `configs/bee_ablation.yaml`."""

    test_scenarios: int = 100
    per_scenario: int = 64
    variants: tuple[AblationVariant, ...] = ()

    def variant(self, name: str) -> AblationVariant:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"Variante desconocida: {name}. Hay: {[v.name for v in self.variants]}")


def load_bee_ablation_config(path: str | Path, base: Optional[BeeNetConfig] = None) -> BeeAblationConfig:
    """Lee el YAML y valida cada variante construyendo su configuración de red."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    variants = tuple(AblationVariant(v["name"], v.get("label", v["name"]), dict(v.get("overrides") or {}),
                                     bool(v.get("reuse_v11", False))) for v in raw.get("variants", []))
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"Nombres de variante repetidos: {names}")
    if ALV_NAME in names:
        raise ValueError(f"'{ALV_NAME}' está reservado para la línea base clásica")
    for v in variants:
        v.config(base or BeeNetConfig())                 # falla temprano si hay un error de tipeo
    return BeeAblationConfig(int(raw.get("test_scenarios", 100)), int(raw.get("per_scenario", 64)), variants)


# ----------------------------------------------------------------------------- puntajes
@dataclass(frozen=True)
class VariantScore:
    """Resultado de una variante con una semilla de entrenamiento sobre el conjunto de prueba."""

    variant: str
    label: str
    seed: Optional[int]           # None para el ALV (no se entrena)
    parameters: int
    views: int
    angle_error_median: float
    angle_error_mean: float
    within_30deg: float
    distance_mae: Optional[float]  # None si el método no estima distancia


def score_predictions(variant: AblationVariant, seed: int, parameters: int, pred: np.ndarray,
                      target: np.ndarray) -> VariantScore:
    """Puntúa las salidas crudas (N, 3) de una red contra el objetivo (N, 3)."""
    rep = homing_report(np.asarray(pred), np.asarray(target))
    return VariantScore(variant.name, variant.label, seed, parameters, len(target), rep["angle_error_median"],
                        rep["angle_error_mean"], rep["within_30deg"], rep["distance_mae"])


def score_alv(current: np.ndarray, home: np.ndarray, target: np.ndarray) -> VariantScore:
    """Puntúa el ALV clásico sobre las mismas vistas (sólo dirección, 0 parámetros)."""
    angle = alv_directions(current, home)
    pred = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1)
    rep = homing_report(pred, np.asarray(target))
    return VariantScore(ALV_NAME, "ALV clásico", None, 0, len(target), rep["angle_error_median"],
                        rep["angle_error_mean"], rep["within_30deg"], None)


# ----------------------------------------------------------------------------- resumen
@dataclass(frozen=True)
class VariantSummary:
    """Media y desviación estándar entre semillas de una variante."""

    variant: str
    label: str
    seeds: int
    parameters: int
    median_mean: float
    median_sd: float
    within30_mean: float
    within30_sd: float
    distance_mae_mean: Optional[float]


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    return (float(statistics.mean(values)), float(statistics.stdev(values)) if len(values) > 1 else 0.0)


def summarize(scores: Iterable[VariantScore]) -> list[VariantSummary]:
    """Agrupa por variante conservando el orden de aparición (ALV primero si está)."""
    groups: dict[str, list[VariantScore]] = {}
    for s in scores:
        groups.setdefault(s.variant, []).append(s)
    out = []
    for name, rows in groups.items():
        med, med_sd = _mean_sd([r.angle_error_median for r in rows])
        w30, w30_sd = _mean_sd([r.within_30deg for r in rows])
        dist = [r.distance_mae for r in rows if r.distance_mae is not None]
        out.append(VariantSummary(name, rows[0].label, len(rows) if rows[0].seed is not None else 0,
                                  rows[0].parameters, med, med_sd, w30, w30_sd,
                                  float(statistics.mean(dist)) if dist else None))
    return out


# ----------------------------------------------------------------------------- salida
SCORE_HEADER = ["variante", "etiqueta", "semilla", "parametros", "vistas", "error_angular_mediano",
                "error_angular_medio", "fraccion_menor_30", "error_distancia_m"]
SUMMARY_HEADER = ["variante", "etiqueta", "semillas", "parametros", "error_mediano_media", "error_mediano_de",
                  "fraccion_menor_30_media", "fraccion_menor_30_de", "error_distancia_m"]


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_scores_csv(path: str | Path, scores: Iterable[VariantScore]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SCORE_HEADER)
        for s in scores:
            w.writerow([s.variant, s.label, "" if s.seed is None else s.seed, s.parameters, s.views,
                        _fmt(s.angle_error_median, 2), _fmt(s.angle_error_mean, 2), _fmt(s.within_30deg),
                        _fmt(s.distance_mae)])


def write_summary_csv(path: str | Path, summary: Iterable[VariantSummary]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SUMMARY_HEADER)
        for s in summary:
            w.writerow([s.variant, s.label, s.seeds, s.parameters, _fmt(s.median_mean, 2), _fmt(s.median_sd, 2),
                        _fmt(s.within30_mean), _fmt(s.within30_sd), _fmt(s.distance_mae_mean)])


def read_summary_csv(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def latex_table(summary: Sequence[VariantSummary], views: int) -> str:
    """Tabla booktabs para el artículo (`paper/tablas/v20_ablacion_abeja.tex`)."""
    def pm(mean: float, sd: float, seeds: int, fmt: str) -> str:
        return fmt.format(mean) + (f" $\\pm$ {fmt.format(sd)}" if seeds > 1 else "")

    lines = ["\\begin{table}[t]",
             f"\\caption{{Ablación de la red de la abeja en la partición de prueba ({views:,} vistas; "
             "media $\\pm$ d.e. entre semillas)}".replace(",", "\\,"),
             "\\label{tab:ablacion_abeja}", "\\centering", "\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{lrrr}", "\\toprule",
             "Variante & Parámetros & Error mediano (\\textdegree) & Error $\\le$30\\textdegree{} \\\\",
             "\\midrule"]
    for s in summary:
        params = "--" if s.parameters == 0 else f"{s.parameters:,}".replace(",", "\\,")
        lines.append(f"{s.label} & {params} & {pm(s.median_mean, s.median_sd, s.seeds, '{:.1f}')} & "
                     f"{pm(100 * s.within30_mean, 100 * s.within30_sd, s.seeds, '{:.0f}')}\\,\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def as_rows(summary: Iterable[VariantSummary]) -> list[dict]:
    """Resumen como diccionarios (útil para pruebas y para el lector de números)."""
    return [asdict(s) for s in summary]
