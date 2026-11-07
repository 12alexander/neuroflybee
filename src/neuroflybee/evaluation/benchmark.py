"""Benchmark en lazo cerrado y ablaciones (sin PyTorch; los controladores neuronales se
construyen con una fábrica inyectada).

Unidad de medida: un episodio en un escenario de PRUEBA. Para los controladores
neuronales se repite con cada semilla de entrenamiento y se reporta media ± d.e. entre
semillas; para los clásicos (deterministas) se reporta además un intervalo de confianza
del 95 % (Wilson) sobre los episodios.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import yaml

from ..config import ProjectConfig
from ..control.base import Controller
from ..sim.scenarios import FAMILIES, ScenarioGenerator, seeds_for
from .closed_loop import EpisodeResult, run_episode


FAMILY_SHORT_ES = {"nominal": "Nominal", "unknown_obstacles": "No previstos", "dynamic": "Móviles",
                   "wind": "Viento", "gps_loss": "Sin GPS", "combined": "Combinado"}


# ------------------------------------------------------------------------- especificación
@dataclass(frozen=True)
class Variant:
    label: str
    controller: str
    fly_model: str = "fly_cnn_gru"
    dagger: bool = True


@dataclass(frozen=True)
class Ablation:
    name: str
    families: tuple[str, ...]
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class BenchmarkSpec:
    families: tuple[str, ...] = tuple(FAMILIES)
    episodes_per_family: int = 30
    classical: tuple[str, ...] = ("direct", "potential_field", "astar", "astar_alv", "astar_supervised")
    neural: tuple[str, ...] = ("fly", "bee", "hybrid")
    neural_seeds: tuple[int, ...] = (0, 1, 2)
    ablations: tuple[Ablation, ...] = field(default_factory=tuple)


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    abls = tuple(Ablation(a["name"], tuple(a["families"]), tuple(Variant(**v) for v in a["variants"]))
                 for a in raw.pop("ablations", []))
    return BenchmarkSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}, ablations=abls)


# ------------------------------------------------------------------------- ejecución
NeuralFactory = Callable[[Variant, int], Optional[Controller]]   # (variante, semilla) → controlador o None
ClassicalFactory = Callable[[str], Controller]


@dataclass(frozen=True)
class Record:
    label: str
    controller: str
    train_seed: Optional[int]
    result: EpisodeResult


def run_variant(cfg: ProjectConfig, variant: Variant, families: Sequence[str], episodes: int,
                classical: ClassicalFactory, neural: Optional[NeuralFactory], neural_names: Sequence[str],
                neural_seeds: Sequence[int], log: Callable[[str], None] = print) -> list[Record]:
    gen, seeds = ScenarioGenerator(cfg), seeds_for("test", episodes)
    records: list[Record] = []
    is_neural = variant.controller in neural_names
    if is_neural and neural is None:
        log(f"  · {variant.label}: requiere PyTorch y pesos entrenados, se omite")
        return records
    for train_seed in (neural_seeds if is_neural else [None]):
        if is_neural:
            if neural(variant, train_seed) is None:
                log(f"  · {variant.label}: sin pesos para la semilla {train_seed}, se omite")
                continue
            make = lambda: neural(variant, train_seed)  # noqa: E731
        else:
            make = lambda: classical(variant.controller)  # noqa: E731
        for family in families:
            for seed in seeds:
                records.append(Record(variant.label, variant.controller, train_seed,
                                      run_episode(cfg, make(), family, seed, gen)))
    return records


# ------------------------------------------------------------------------- estadística
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass(frozen=True)
class Cell:
    n_episodes: int
    n_seeds: int
    success_mean: float
    success_std: float
    success_ci: tuple[float, float]
    collision_mean: float
    home_error_median: float
    latency_ms: float

    def fmt(self, key: str = "success") -> str:
        mean = self.success_mean if key == "success" else self.collision_mean
        if key == "success" and self.n_seeds > 1:
            return f"{mean:.0%} ± {self.success_std:.0%}"
        return f"{mean:.0%}"


def summarize_cell(records: Sequence[Record]) -> Cell:
    by_seed: dict = {}
    for r in records:
        by_seed.setdefault(r.train_seed, []).append(r.result)
    seed_success = [np.mean([e.success for e in eps]) for eps in by_seed.values()]
    all_eps = [r.result for r in records]
    k = int(sum(e.success for e in all_eps))
    return Cell(len(all_eps), len(by_seed), float(np.mean(seed_success)), float(np.std(seed_success)),
                wilson_interval(k, len(all_eps)), float(np.mean([e.collided for e in all_eps])),
                float(np.median([e.home_error for e in all_eps])), float(np.mean([e.mean_latency_ms for e in all_eps])))


def table(records: Sequence[Record], families: Sequence[str]) -> dict[str, dict[str, Cell]]:
    """etiqueta → {condición | 'total' → Cell}, en el orden de aparición."""
    labels = list(dict.fromkeys(r.label for r in records))
    out: dict[str, dict[str, Cell]] = {}
    for label in labels:
        sel = [r for r in records if r.label == label]
        row = {f: summarize_cell([r for r in sel if r.result.family == f]) for f in families
               if any(r.result.family == f for r in sel)}
        row["total"] = summarize_cell(sel)
        out[label] = row
    return out


# ------------------------------------------------------------------------- salida
def write_episodes_csv(path: str | Path, records: Sequence[Record], group: str = "") -> None:
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["grupo", "etiqueta", "controlador", "semilla_entrenamiento", "condicion", "semilla_escenario",
                        "exito", "choque", "aterrizo", "cobertura", "error_retorno_m", "pasos", "longitud_m",
                        "latencia_ms"])
        for r in records:
            e = r.result
            w.writerow([group, r.label, r.controller, "" if r.train_seed is None else r.train_seed, e.family, e.seed,
                        int(e.success), int(e.collided), int(e.landed), f"{e.coverage:.3f}", f"{e.home_error:.3f}",
                        e.steps, f"{e.path_length:.2f}", f"{e.mean_latency_ms:.3f}"])


def write_table_csv(path: str | Path, tab: dict[str, dict[str, Cell]], families: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["controlador", *[f"exito_{f}" for f in families], "exito_total", "exito_total_de",
                    "ic95_inf", "ic95_sup", "choques_total", "error_retorno_mediano_m", "latencia_ms",
                    "semillas", "episodios"])
        for label, row in tab.items():
            t = row["total"]
            w.writerow([label, *[f"{row[f].success_mean:.3f}" if f in row else "" for f in families],
                        f"{t.success_mean:.3f}", f"{t.success_std:.3f}", f"{t.success_ci[0]:.3f}",
                        f"{t.success_ci[1]:.3f}", f"{t.collision_mean:.3f}", f"{t.home_error_median:.2f}",
                        f"{t.latency_ms:.2f}", t.n_seeds, t.n_episodes])


def latex_table(tab: dict[str, dict[str, Cell]], families: Sequence[str], caption: str, label: str) -> str:
    """Tabla IEEE: éxito por condición (%), éxito total y choques. Resalta el mejor total."""
    best = max(row["total"].success_mean for row in tab.values())
    head = " & ".join(FAMILY_SHORT_ES.get(f, f) for f in families)
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "c" * (len(families) + 2) + "}", r"\toprule",
             rf"Controlador & {head} & Total & Choques \\", r"\midrule"]
    esc = lambda text: text.replace("%", r"\%")  # noqa: E731
    for name, row in tab.items():
        cells = [esc(row[f].fmt()) if f in row else "--" for f in families]
        total = esc(row["total"].fmt())
        if abs(row["total"].success_mean - best) < 1e-9:
            total = rf"\textbf{{{total}}}"
        lines.append(rf"{name} & {' & '.join(cells)} & {total} & {esc(row['total'].fmt('collision'))} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"
