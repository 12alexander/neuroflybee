"""Regenera los números y tablas del artículo desde results/ (v19).

    python scripts/collect_results.py           # luego: make paper

Escribe paper/generado/numeros.tex (macros \\nfb...), paper/generado/tabla_costo.tex y
results/v19_numeros.csv (de qué archivo sale cada número). Lo que falta se marca en rojo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroflybee.reporting.numbers import collect, cost_table  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--paper", default=str(ROOT / "paper"))
    args = parser.parse_args(argv)
    results, out = Path(args.results), Path(args.paper) / "generado"
    ns, cost = collect(results)
    ns.write(out / "numeros.tex", results / "v19_numeros.csv")
    (out / "tabla_costo.tex").write_text(cost_table(cost), encoding="utf-8")
    pending = [n for n in ns.values if ns.is_pending(n)]
    print(f"{len(ns.values)} números → {out / 'numeros.tex'}; pendientes: {len(pending)}"
          + (f" ({', '.join(sorted(pending)[:6])}{'…' if len(pending) > 6 else ''})" if pending else ""))


if __name__ == "__main__":
    main()
