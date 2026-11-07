"""Significancia estadística de las comparaciones del benchmark (v22) — sin PyTorch.

Cada comparación enfrenta dos controladores (A y B) sobre los MISMOS escenarios de prueba,
así que se analiza de forma **pareada**: para cada escenario (condición, semilla) se toma el
éxito de A y el de B (en los controladores neuronales, la media entre sus semillas de
entrenamiento) y se estudia la diferencia d_i = éxito_A − éxito_B.

* **Δ e IC 95 %** — media de d_i con intervalo bootstrap percentil pareado (se remuestrean
  escenarios, no episodios sueltos).
* **p** — prueba de permutación por cambio de signo (bajo H0 A y B son intercambiables en cada
  escenario). Si todas las d_i valen −1, 0 o 1 se calcula EXACTA (equivale a la prueba exacta
  de McNemar sobre los pares discordantes); si no, por Monte Carlo.
* **Holm** — corrección por comparaciones múltiples sobre todas las filas del archivo.
* **h de Cohen** — tamaño de efecto entre proporciones: 2·asen√p_A − 2·asen√p_B
  (≈0,2 pequeño, 0,5 mediano, 0,8 grande).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import yaml

TOTAL = "total"
CONDITION_ES = {"total": "todas", "nominal": "nominal", "unknown_obstacles": "no previstos",
                "dynamic": "móviles", "wind": "viento", "gps_loss": "sin GPS", "combined": "combinada"}


# ----------------------------------------------------------------------------- configuración
@dataclass(frozen=True)
class Comparison:
    """A frente a B dentro de un grupo del benchmark (`grupo` y `etiqueta` de v14_episodios.csv)."""

    group: str
    a: str
    b: str
    conditions: tuple[str, ...] = (TOTAL,)
    macro: str = ""                      # prefijo opcional para las macros del artículo


@dataclass(frozen=True)
class SignificanceConfig:
    n_bootstrap: int = 10_000
    n_permutations: int = 20_000
    alpha: float = 0.05
    seed: int = 0
    comparisons: tuple[Comparison, ...] = ()


def load_significance_config(path: str | Path) -> SignificanceConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    comps = tuple(Comparison(c["grupo"], c["a"], c["b"], tuple(c.get("condiciones", [TOTAL])), c.get("macro", ""))
                  for c in raw.get("comparaciones", []))
    if any(c.macro and not c.macro.isalpha() for c in comps):
        raise ValueError("Los prefijos de macro sólo pueden tener letras")
    return SignificanceConfig(int(raw.get("n_bootstrap", 10_000)), int(raw.get("n_permutaciones", 20_000)),
                              float(raw.get("alfa", 0.05)), int(raw.get("semilla", 0)), comps)


# ----------------------------------------------------------------------------- datos
Outcomes = dict[tuple[str, int], float]          # (condición, semilla de escenario) → éxito medio


def load_episodes(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def scenario_outcomes(rows: Iterable[dict], group: str, label: str) -> Outcomes:
    """Éxito por escenario de un brazo; promedia las semillas de entrenamiento si hay varias."""
    acc: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        if r["grupo"] == group and r["etiqueta"] == label:
            acc.setdefault((r["condicion"], int(r["semilla_escenario"])), []).append(float(r["exito"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def paired_differences(a: Outcomes, b: Outcomes, condition: str) -> tuple[np.ndarray, np.ndarray]:
    """Éxitos pareados (A, B) sobre los escenarios comunes de una condición (o de todas)."""
    keys = sorted(k for k in a.keys() & b.keys() if condition == TOTAL or k[0] == condition)
    return np.array([a[k] for k in keys], dtype=float), np.array([b[k] for k in keys], dtype=float)


# ----------------------------------------------------------------------------- estadística
def bootstrap_ci(diffs: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05) -> tuple[float, float]:
    """IC percentil de la media de `diffs` remuestreando escenarios con reemplazo."""
    if len(diffs) == 0:
        return float("nan"), float("nan")
    means = diffs[rng.integers(0, len(diffs), (n_boot, len(diffs)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _exact_sign_test(diffs: np.ndarray) -> float:
    """p bilateral exacto cuando d_i ∈ {−1, 0, 1} (McNemar exacta sobre los discordantes)."""
    n = int(np.count_nonzero(diffs))
    if n == 0:
        return 1.0
    k = int((diffs > 0).sum())
    extreme = abs(2 * k - n)
    p = sum(math.comb(n, j) for j in range(n + 1) if abs(2 * j - n) >= extreme) / 2 ** n
    return min(1.0, p)


def sign_flip_pvalue(diffs: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """p bilateral de la prueba de permutación pareada (cambio de signo)."""
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0 or not np.any(diffs):
        return 1.0
    if np.all(np.isin(diffs, (-1.0, 0.0, 1.0))):
        return _exact_sign_test(diffs)
    observed = abs(diffs.mean())
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + n_perm))


def cohen_h(p_a: float, p_b: float) -> float:
    return float(2 * math.asin(math.sqrt(p_a)) - 2 * math.asin(math.sqrt(p_b)))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Valores p ajustados por Holm–Bonferroni (monótonos, recortados a 1)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


# ----------------------------------------------------------------------------- comparaciones
@dataclass(frozen=True)
class ComparisonResult:
    group: str
    a: str
    b: str
    condition: str
    n: int
    rate_a: float
    rate_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    cohen_h: float
    macro: str = ""
    p_holm: float = float("nan")
    significant: bool = False


def compare(a: Outcomes, b: Outcomes, comp: Comparison, condition: str, cfg: SignificanceConfig,
            rng: np.random.Generator) -> Optional[ComparisonResult]:
    sa, sb = paired_differences(a, b, condition)
    if len(sa) == 0:
        return None
    d = sa - sb
    lo, hi = bootstrap_ci(d, cfg.n_bootstrap, rng, cfg.alpha)
    return ComparisonResult(comp.group, comp.a, comp.b, condition, len(d), float(sa.mean()), float(sb.mean()),
                            float(d.mean()), lo, hi, sign_flip_pvalue(d, cfg.n_permutations, rng),
                            cohen_h(float(sa.mean()), float(sb.mean())), comp.macro)


@dataclass
class SignificanceReport:
    results: list[ComparisonResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)     # comparaciones sin datos (p. ej. sin pesos)


def run_significance(rows: Sequence[dict], cfg: SignificanceConfig) -> SignificanceReport:
    """Todas las comparaciones del archivo de configuración, con corrección de Holm conjunta."""
    rng = np.random.default_rng(cfg.seed)
    report = SignificanceReport()
    for comp in cfg.comparisons:
        a, b = scenario_outcomes(rows, comp.group, comp.a), scenario_outcomes(rows, comp.group, comp.b)
        for cond in comp.conditions:
            res = compare(a, b, comp, cond, cfg, rng) if a and b else None
            if res is None:
                report.skipped.append(f"{comp.group}: {comp.a} vs {comp.b} ({cond})")
            else:
                report.results.append(res)
    adjusted = holm([r.p_value for r in report.results])
    report.results = [replace(r, p_holm=p, significant=p < cfg.alpha) for r, p in zip(report.results, adjusted)]
    return report


# ----------------------------------------------------------------------------- salida
HEADER = ["grupo", "a", "b", "condicion", "escenarios", "exito_a", "exito_b", "diferencia", "ic_inf", "ic_sup",
          "p", "p_holm", "h_cohen", "significativa", "macro"]


def write_csv(path: str | Path, report: SignificanceReport) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for r in report.results:
            w.writerow([r.group, r.a, r.b, r.condition, r.n, f"{r.rate_a:.4f}", f"{r.rate_b:.4f}", f"{r.diff:.4f}",
                        f"{r.ci_low:.4f}", f"{r.ci_high:.4f}", f"{r.p_value:.5f}", f"{r.p_holm:.5f}",
                        f"{r.cohen_h:.3f}", int(r.significant), r.macro])


def _p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def latex_table(report: SignificanceReport, alpha: float = 0.05) -> str:
    """Tabla a dos columnas para el artículo (`paper/tablas/v22_significancia.tex`)."""
    lines = ["\\begin{table*}[t]",
             "\\caption{Comparaciones pareadas sobre los mismos escenarios de prueba: diferencia de éxito "
             "$\\Delta$ con IC 95\\,\\% bootstrap, valor $p$ de la permutación pareada corregido por Holm y "
             f"$h$ de Cohen. En negrita, $p_{{\\text{{Holm}}}}<{alpha}$.}}",
             "\\label{tab:significancia}", "\\centering", "\\resizebox{\\textwidth}{!}{%",
             "\\begin{tabular}{llllrrrrrr}", "\\toprule",
             "Grupo & A & B & Condición & $n$ & Éxito A & Éxito B & $\\Delta$ [IC 95\\,\\%] & "
             "$p_{\\text{Holm}}$ & $h$ \\\\", "\\midrule"]
    for r in report.results:
        p = _p(r.p_holm)
        p = f"\\textbf{{{p}}}" if r.significant else p
        group = "Principal" if r.group == "principal" else r.group
        lines.append(f"{group} & {r.a} & {r.b} & {CONDITION_ES.get(r.condition, r.condition)} & {r.n} & "
                     f"{100 * r.rate_a:.0f}\\,\\% & {100 * r.rate_b:.0f}\\,\\% & "
                     f"{100 * r.diff:+.0f} [{100 * r.ci_low:+.0f}, {100 * r.ci_high:+.0f}] & {p} & {r.cohen_h:+.2f} \\\\")
    if not report.results:
        lines.append("\\multicolumn{10}{c}{\\pendiente{correr el benchmark y scripts/significance.py}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(lines).replace("A*", "A$^*$") + "\n"
