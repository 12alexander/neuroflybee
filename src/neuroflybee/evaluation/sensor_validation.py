"""Validación del sensor panorámico de la abeja (v21) — sin PyTorch.

¿Es creíble un simulador 2,5D? El movimiento es plano (altura fija, como en Firlefyn et al.,
2024), pero la abeja ve una **silueta de horizonte**: la elevación angular de los objetos en
72 sectores alineados con la brújula. Antes de entrenar redes con ese sensor comprobamos que
reproduce tres propiedades conocidas de los panoramas reales:

1. **Función de diferencia de imagen traslacional (IDF)** — Zeil et al. (2003): la diferencia
   RMS entre la vista actual y la de casa crece de forma suave y monótona al alejarse de
   casa. Si no creciera, ningún método de retorno visual podría funcionar.
2. **IDF rotacional** — si se gira el panorama, la diferencia es mínima en la orientación
   correcta: el horizonte contiene suficiente estructura para recuperar el rumbo.
3. **Área de captura del ALV** — el método analítico clásico apunta a casa (error < 90°)
   en la mayor parte del entorno; moverse con error < 90° siempre acerca a la meta.

Todo se mide con el mismo `PanoramaSensor`, ruido y escenarios que usan las redes.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

from ..config import ProjectConfig, from_dict
from ..control.homing import alv_directions
from ..domain.world import World
from ..evaluation.metrics import angular_errors_deg
from ..sim.scenarios import ScenarioGenerator, seeds_for
from ..sim.sensors import PanoramaSensor


# ----------------------------------------------------------------------------- configuración
@dataclass(frozen=True)
class SensorValidationConfig:
    """Parámetros de la validación (ver `configs/sensor_validation.yaml`)."""

    scenarios: int = 30                                   # escenarios de PRUEBA evaluados
    families: tuple[str, ...] = ("nominal", "unknown_obstacles")
    radii: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0)
    directions: int = 16                                  # puntos por anillo alrededor de casa
    noise_draws: int = 8                                  # realizaciones del ruido por punto
    rotation_radii: tuple[float, ...] = (1.0, 2.0, 4.0)   # distancias para recuperar el rumbo
    field_step: float = 1.5                               # m entre puntos del campo del ALV
    field_radius: float = 10.0
    min_clearance: float = 0.5                            # m libres alrededor de cada punto
    arrive_rms: float = 0.03                              # umbral de llegada del ALV (homing.py)
    seed: int = 0


def load_sensor_validation_config(path: Optional[str | Path]) -> SensorValidationConfig:
    if path is None:
        return SensorValidationConfig()
    with open(path, encoding="utf-8") as fh:
        return from_dict(SensorValidationConfig, yaml.safe_load(fh) or {})


# ----------------------------------------------------------------------------- primitivas
def rms_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Diferencia RMS entre panoramas (sobre el último eje)."""
    return np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2, axis=-1))


def noisy(view: np.ndarray, sd: float, rng: np.random.Generator, draws: int) -> np.ndarray:
    """`draws` lecturas ruidosas de una vista, recortadas a [0, 1] como en `NavigationEnv`."""
    return np.clip(view[None] + rng.normal(0.0, sd, (draws, view.shape[-1])), 0.0, 1.0)


def ring_points(world: World, center: np.ndarray, radius: float, n: int, min_clearance: float) -> np.ndarray:
    """Puntos de un anillo alrededor de `center` que están dentro del mapa y en espacio libre."""
    if radius == 0:
        return np.atleast_2d(center).astype(float)
    ang = np.linspace(-np.pi, np.pi, n, endpoint=False)
    pts = center + radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    inside = (pts[:, 0] > 0) & (pts[:, 1] > 0) & (pts[:, 0] < world.width) & (pts[:, 1] < world.height)
    pts = pts[inside]
    return pts[world.clearance(pts) >= min_clearance] if len(pts) else pts


def rotational_idf(view: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """RMS entre `view` girada k sectores y `reference`, para k = 0..bins-1."""
    bins = len(view)
    rolled = np.stack([np.roll(view, k) for k in range(bins)])
    return rms_difference(rolled, reference[None])


def heading_error_deg(view: np.ndarray, reference: np.ndarray, true_shift: int) -> float:
    """Error (°) al recuperar el giro aplicado a la vista buscando el mínimo de la IDF rotacional."""
    bins = len(view)
    best = int(np.argmin(rotational_idf(view, reference)))
    # `best` deshace el giro: debe ser -true_shift (módulo bins)
    diff = ((best + true_shift + bins // 2) % bins) - bins // 2
    return abs(diff) * 360.0 / bins


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Correlación de rangos de Spearman (empates con rango promedio); NaN si no hay variación."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for value in np.unique(v):                       # promedia los empates
            tie = v == value
            r[tie] = r[tie].mean()
        return r

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def alv_field(sensor: PanoramaSensor, objects: np.ndarray, world: World, step: float, radius: float,
              min_clearance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Campo del ALV alrededor de casa: (puntos, dirección estimada, dirección real) en rad."""
    xs = np.arange(-radius, radius + 1e-9, step)
    pts = np.array([world.base + [x, y] for x in xs for y in xs if 0.75 <= np.hypot(x, y) <= radius])
    inside = (pts[:, 0] > 0) & (pts[:, 1] > 0) & (pts[:, 0] < world.width) & (pts[:, 1] < world.height)
    pts = pts[inside]
    pts = pts[world.clearance(pts) >= min_clearance]
    home = sensor.view_from(objects, world.base)
    views = np.array([sensor.view_from(objects, p) for p in pts])
    est = alv_directions(views, np.repeat(home[None], len(pts), axis=0))
    true = np.arctan2(world.base[1] - pts[:, 1], world.base[0] - pts[:, 0])
    return pts, est, true


# ----------------------------------------------------------------------------- un escenario
@dataclass
class ScenarioValidation:
    """Resultado de la validación en un escenario de prueba."""

    family: str
    seed: int
    radii: list[float]
    idf_clean: list[float]                 # IDF sin ruido (geometría pura)
    idf_noisy: list[float]                 # IDF con el ruido del sensor (lo que ve el dron)
    spearman: float                        # monotonía de la IDF (sin ruido) frente a la distancia
    heading_error: dict[str, float] = field(default_factory=dict)   # radio → error mediano (°)
    alv_error_median: float = float("nan")
    alv_catchment: float = float("nan")    # fracción del campo con error < 90°


def validate_scenario(cfg: ProjectConfig, vcfg: SensorValidationConfig, family: str, seed: int,
                      rng: np.random.Generator) -> ScenarioValidation:
    world = ScenarioGenerator(cfg).generate(family, seed).world
    sensor = PanoramaSensor(cfg.sensors)
    objects = sensor.objects(world)
    home = sensor.view_from(objects, world.base)

    clean, noisy_mean = [], []
    for r in vcfg.radii:
        pts = ring_points(world, world.base, r, vcfg.directions, vcfg.min_clearance)
        if not len(pts):
            clean.append(float("nan")); noisy_mean.append(float("nan"))
            continue
        views = np.array([sensor.view_from(objects, p) for p in pts])
        clean.append(float(rms_difference(views, home[None]).mean()))
        noisy_mean.append(float(np.mean([rms_difference(noisy(v, sensor.noise, rng, vcfg.noise_draws), home[None])
                                         for v in views])))

    heading: dict[str, float] = {}
    for r in vcfg.rotation_radii:
        pts = ring_points(world, world.base, r, vcfg.directions, vcfg.min_clearance)
        errs = []
        for p in pts:
            shift = int(rng.integers(len(home)))
            view = noisy(sensor.view_from(objects, p), sensor.noise, rng, 1)[0]
            errs.append(heading_error_deg(np.roll(view, shift), home, shift))
        heading[f"{r:g}"] = float(np.median(errs)) if errs else float("nan")

    pts, est, true = alv_field(sensor, objects, world, vcfg.field_step, vcfg.field_radius, vcfg.min_clearance)
    err = angular_errors_deg(est, true)
    return ScenarioValidation(family, seed, list(vcfg.radii), clean, noisy_mean,
                              spearman_rho(vcfg.radii, clean), heading,
                              float(np.median(err)) if len(err) else float("nan"),
                              float((err < 90).mean()) if len(err) else float("nan"))


def run_validation(cfg: ProjectConfig, vcfg: SensorValidationConfig) -> list[ScenarioValidation]:
    """Valida el sensor en `vcfg.scenarios` escenarios de PRUEBA, alternando familias."""
    rng = np.random.default_rng(vcfg.seed)
    seeds = seeds_for("test", vcfg.scenarios)
    return [validate_scenario(cfg, vcfg, vcfg.families[i % len(vcfg.families)], s, rng)
            for i, s in enumerate(seeds)]


# ----------------------------------------------------------------------------- resumen
@dataclass(frozen=True)
class ValidationSummary:
    scenarios: int
    spearman_mean: float
    monotonic_fraction: float              # escenarios con ρ de Spearman > 0.9
    noise_floor: float                     # IDF con ruido en la propia casa
    arrival_radius_m: float                # distancia a la que la IDF media supera el umbral de llegada
    heading_error: dict[str, float]        # radio → mediana entre escenarios (°)
    alv_error_median: float
    alv_catchment_mean: float

    def as_dict(self) -> dict:
        return asdict(self)


def idf_curve(rows: Sequence[ScenarioValidation]) -> dict[str, np.ndarray]:
    """Media y d.e. entre escenarios de la IDF limpia y con ruido, por radio."""
    clean = np.array([r.idf_clean for r in rows], dtype=float)
    noisy_ = np.array([r.idf_noisy for r in rows], dtype=float)
    return {"radii": np.asarray(rows[0].radii, dtype=float),
            "clean_mean": np.nanmean(clean, axis=0), "clean_sd": np.nanstd(clean, axis=0),
            "noisy_mean": np.nanmean(noisy_, axis=0), "noisy_sd": np.nanstd(noisy_, axis=0)}


def arrival_radius(radii: np.ndarray, curve: np.ndarray, threshold: float) -> float:
    """Primer radio (interpolado) en que la curva supera el umbral; inf si nunca lo hace."""
    above = np.flatnonzero(curve >= threshold)
    if not len(above):
        return float("inf")
    i = int(above[0])
    if i == 0:
        return 0.0
    r0, r1, c0, c1 = radii[i - 1], radii[i], curve[i - 1], curve[i]
    return float(r0 + (threshold - c0) * (r1 - r0) / (c1 - c0)) if c1 != c0 else float(r1)


def summarize_validation(rows: Sequence[ScenarioValidation], arrive_rms: float) -> ValidationSummary:
    curve = idf_curve(rows)
    rho = np.array([r.spearman for r in rows], dtype=float)
    radii_keys = rows[0].heading_error.keys()
    return ValidationSummary(
        scenarios=len(rows),
        spearman_mean=float(np.nanmean(rho)),
        monotonic_fraction=float(np.mean(rho > 0.9)),
        noise_floor=float(curve["noisy_mean"][0]) if curve["radii"][0] == 0 else float("nan"),
        arrival_radius_m=arrival_radius(curve["radii"], curve["noisy_mean"], arrive_rms),
        heading_error={k: float(np.nanmedian([r.heading_error[k] for r in rows])) for k in radii_keys},
        alv_error_median=float(np.nanmedian([r.alv_error_median for r in rows])),
        alv_catchment_mean=float(np.nanmean([r.alv_catchment for r in rows])),
    )


# ----------------------------------------------------------------------------- archivos
def write_outputs(rows: Sequence[ScenarioValidation], summary: ValidationSummary, results: str | Path) -> None:
    """`v21_validacion_sensor.csv` (por escenario), `v21_idf.csv` (curva) y `v21_validacion_resumen.json`."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].heading_error)
    with open(results / "v21_validacion_sensor.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["familia", "semilla", "spearman_idf", *[f"error_rumbo_{k}m" for k in keys],
                    "error_alv_mediano", "captura_alv"])
        for r in rows:
            w.writerow([r.family, r.seed, f"{r.spearman:.4f}", *[f"{r.heading_error[k]:.2f}" for k in keys],
                        f"{r.alv_error_median:.2f}", f"{r.alv_catchment:.4f}"])
    curve = idf_curve(rows)
    with open(results / "v21_idf.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["distancia_m", "idf_media", "idf_de", "idf_ruido_media", "idf_ruido_de"])
        for i, r in enumerate(curve["radii"]):
            w.writerow([f"{r:g}", f"{curve['clean_mean'][i]:.5f}", f"{curve['clean_sd'][i]:.5f}",
                        f"{curve['noisy_mean'][i]:.5f}", f"{curve['noisy_sd'][i]:.5f}"])
    (results / "v21_validacion_resumen.json").write_text(
        json.dumps(summary.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
