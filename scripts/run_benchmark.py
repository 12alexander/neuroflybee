"""Benchmark final y ablaciones (v14).

Corre todos los controladores en las 6 condiciones con semillas de PRUEBA. Los
controladores neuronales se incluyen si hay PyTorch y pesos (cada semilla de entrenamiento
cuenta como una repetición → media ± d.e.); si no, se reportan sólo los clásicos.

Uso:
    python scripts/run_benchmark.py                    # configs/benchmark.yaml (30 episodios por condición)
    python scripts/run_benchmark.py --episodes 5       # prueba rápida
Genera:
    results/v14_episodios.csv              cada episodio (para reanalizar)
    results/v14_tabla_principal.csv        éxito por condición, IC 95 %, choques, error de retorno, latencia
    results/v14_ablaciones.csv
    paper/tablas/v14_tabla_principal.tex   tabla lista para el artículo (y una por ablación)
    results/figures/v14_benchmark.png
    results/figures/v14_ablaciones.png     (si hay ablaciones con al menos dos variantes)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from neuroflybee.config import load_config  # noqa: E402
from neuroflybee.control import classical  # noqa: E402,F401
from neuroflybee.control import registry  # noqa: E402
from neuroflybee.control.neural_registry import NEURAL_NAMES, build_neural_controller, neural_checkpoints  # noqa: E402
from neuroflybee.evaluation.benchmark import (Record, Variant, latex_table, load_benchmark_spec,  # noqa: E402
                                              run_variant, table, write_episodes_csv, write_table_csv)
from neuroflybee.sim.scenarios import FAMILY_LABELS_ES  # noqa: E402
from neuroflybee.viz.export import CONTROLLER_STYLE  # noqa: E402
from neuroflybee.viz.style import apply_style  # noqa: E402


def neural_factory(checkpoints: Path):
    try:
        import torch  # noqa: F401
    except ImportError:
        return None

    def make(variant: Variant, seed: int):
        if neural_checkpoints(variant.controller, checkpoints, seed, variant.fly_model, variant.dagger) is None:
            return None
        return build_neural_controller(variant.controller, checkpoints, seed, variant.fly_model, variant.dagger)
    return make


class CachedRunner:
    """Evita repetir episodios idénticos entre la tabla principal y las ablaciones."""

    def __init__(self, cfg, spec, neural):
        self.cfg, self.spec, self.neural, self.cache = cfg, spec, neural, {}

    def __call__(self, variant: Variant, families) -> list[Record]:
        out = []
        for family in families:
            key = (variant.controller, variant.fly_model, variant.dagger, family)
            if key not in self.cache:
                t0 = time.time()
                self.cache[key] = run_variant(self.cfg, variant, [family], self.spec.episodes_per_family,
                                              registry.create, self.neural, NEURAL_NAMES, self.spec.neural_seeds)
                print(f"  {variant.label:32s} {FAMILY_LABELS_ES[family]:28s} {time.time() - t0:5.1f} s")
            out += [Record(variant.label, r.controller, r.train_seed, r.result) for r in self.cache[key]]
        return out


def plot_main(tab, families, out: Path, episodes: int) -> None:
    labels = list(tab)
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(families))
    width = 0.8 / len(labels)
    for i, label in enumerate(labels):
        row, ctrl = tab[label], next((k for k, v in CONTROLLER_STYLE.items() if v[0] == label), label)
        color = CONTROLLER_STYLE.get(ctrl, (label, None))[1]
        means = [row[f].success_mean if f in row else np.nan for f in families]
        lo = [row[f].success_mean - row[f].success_ci[0] if f in row else 0 for f in families]
        hi = [row[f].success_ci[1] - row[f].success_mean if f in row else 0 for f in families]
        axes[0].bar(x + (i - (len(labels) - 1) / 2) * width, means, width, yerr=[lo, hi], capsize=2,
                    color=color, label=label)
        axes[1].barh(i, row["total"].success_mean, color=color,
                     xerr=[[row["total"].success_mean - row["total"].success_ci[0]],
                           [row["total"].success_ci[1] - row["total"].success_mean]], capsize=3)
        axes[1].text(row["total"].success_ci[1] + 0.02, i, f"{row['total'].success_mean:.0%}", va="center", fontsize=9,
                     fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([FAMILY_LABELS_ES[f] for f in families], rotation=12)
    axes[0].set_ylabel("misiones exitosas")
    axes[0].set_title(f"Éxito por condición ({episodes} escenarios de prueba; barras = IC 95 %)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_title("Éxito total")
    for ax, fmt in ((axes[0], "y"), (axes[1], "x")):
        ax.set_ylim(0, 1.05) if fmt == "y" else ax.set_xlim(0, 1.1)
        (ax.yaxis if fmt == "y" else ax.xaxis).set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis=fmt, alpha=0.3)
    fig.suptitle("v14 — Benchmark en lazo cerrado (partición de PRUEBA)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_ablations(results: list[tuple[str, dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(max(8.0, 5.2 * len(results)), 4.8), squeeze=False)
    for ax, (name, tab) in zip(axes[0], results):
        labels = list(tab)
        cells = [tab[lb]["total"] for lb in labels]
        means = [c.success_mean for c in cells]
        err = [[c.success_mean - c.success_ci[0] for c in cells], [c.success_ci[1] - c.success_mean for c in cells]]
        ax.bar(range(len(labels)), means, yerr=err, capsize=4, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(labels))))
        for i, c in enumerate(cells):
            sd = f" ± {c.success_std:.0%}" if c.n_seeds > 1 else ""
            ax.text(i, c.success_ci[1] + 0.03, f"{c.success_mean:.0%}{sd}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("v14 — Ablaciones: éxito de misión (barras = IC 95 %)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", default=str(ROOT / "configs" / "benchmark.yaml"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--paper", default=str(ROOT / "paper" / "tablas"))
    args = parser.parse_args(argv)

    apply_style()
    cfg, spec = load_config(args.config), load_benchmark_spec(args.spec)
    if args.episodes:
        from dataclasses import replace
        spec = replace(spec, episodes_per_family=args.episodes)
    results, paper = Path(args.results), Path(args.paper)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)
    episodes_csv = results / "v14_episodios.csv"
    episodes_csv.unlink(missing_ok=True)
    neural = neural_factory(Path(args.checkpoints))
    runner = CachedRunner(cfg, spec, neural)
    t0 = time.time()

    print("== Tabla principal ==")
    main_variants = [Variant(CONTROLLER_STYLE.get(n, (n,))[0], n) for n in (*spec.classical, *spec.neural)]
    records = [r for v in main_variants for r in runner(v, spec.families)]
    write_episodes_csv(episodes_csv, records, "principal")
    tab = table(records, spec.families)
    write_table_csv(results / "v14_tabla_principal.csv", tab, spec.families)
    (paper / "v14_tabla_principal.tex").write_text(latex_table(
        tab, spec.families, "Éxito de misión en la partición de prueba (media $\\pm$ d.e. entre semillas de "
        "entrenamiento para los controladores neuronales) y tasa total de choques.", "tab:principal"), encoding="utf-8")
    plot_main(tab, spec.families, results / "figures" / "v14_benchmark.png", spec.episodes_per_family)

    print("== Ablaciones ==")
    done = []
    for k, ablation in enumerate(spec.ablations, start=1):
        recs = [r for v in ablation.variants for r in runner(v, ablation.families)]
        tab_a = table(recs, ablation.families)
        if len(tab_a) < 2:
            print(f"  · '{ablation.name}': menos de dos variantes con pesos, se omite")
            continue
        write_episodes_csv(episodes_csv, recs, ablation.name)
        write_table_csv(results / f"v14_ablacion_{k}.csv", tab_a, ablation.families)
        (paper / f"v14_ablacion_{k}.tex").write_text(latex_table(tab_a, ablation.families, ablation.name,
                                                                 f"tab:ablacion{k}"), encoding="utf-8")
        done.append((ablation.name, tab_a))
    if done:
        plot_ablations(done, results / "figures" / "v14_ablaciones.png")
    print(f"Listo en {(time.time() - t0) / 60:.1f} min. Ablaciones con datos: {len(done)}/{len(spec.ablations)}")
    for label, row in tab.items():
        t = row["total"]
        print(f"  {label:28s} éxito {t.success_mean:5.0%} (IC95 {t.success_ci[0]:.0%}–{t.success_ci[1]:.0%})"
              f"  choques {t.collision_mean:4.0%}  latencia {t.latency_ms:5.2f} ms")


if __name__ == "__main__":
    main()
