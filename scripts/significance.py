"""Significancia estadística de las comparaciones del benchmark (v22). No necesita PyTorch.

Lee results/v14_episodios.csv (un renglón por episodio) y, para cada comparación de
configs/significance.yaml, calcula la diferencia de éxito pareada por escenario, su IC 95 %
bootstrap, el valor p de la permutación pareada (exacto cuando los éxitos son 0/1), la
corrección de Holm y la h de Cohen.

Uso:
    python scripts/significance.py
Genera:
    results/v22_significancia.csv
    paper/tablas/v22_significancia.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroflybee.evaluation.significance import (latex_table, load_episodes,  # noqa: E402
                                                 load_significance_config, run_significance, write_csv)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", default=str(ROOT / "results" / "v14_episodios.csv"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "significance.yaml"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--paper", default=str(ROOT / "paper"))
    args = parser.parse_args(argv)

    cfg = load_significance_config(args.config)
    episodes = Path(args.episodes)
    rows = load_episodes(episodes) if episodes.exists() else []
    if not rows:
        print(f"No hay episodios en {episodes}: ejecute primero scripts/run_benchmark.py")
    report = run_significance(rows, cfg)
    write_csv(Path(args.results) / "v22_significancia.csv", report)
    tables = Path(args.paper) / "tablas"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "v22_significancia.tex").write_text(latex_table(report, cfg.alpha), encoding="utf-8")
    for r in report.results:
        mark = "*" if r.significant else " "
        print(f"{mark} {r.a[:26]:26s} vs {r.b[:22]:22s} [{r.condition:17s}] Δ {100 * r.diff:+5.1f} pp "
              f"[{100 * r.ci_low:+5.1f}, {100 * r.ci_high:+5.1f}]  p_Holm {r.p_holm:.4f}  h {r.cohen_h:+.2f}")
    if report.skipped:
        print(f"Sin datos ({len(report.skipped)}): " + "; ".join(report.skipped[:4])
              + ("…" if len(report.skipped) > 4 else ""))
    n_sig = sum(r.significant for r in report.results)
    print(f"{n_sig} de {len(report.results)} comparaciones significativas (Holm, α = {cfg.alpha})")


if __name__ == "__main__":
    main()
