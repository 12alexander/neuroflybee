"""Versión Word (.docx) del artículo, con formato IEEE, para revisar y comentar en equipo (v19).

    python scripts/collect_results.py && python scripts/paper_to_word.py
    → paper/NeuroFlyBee_Paper.docx

Requiere pandoc. El fuente oficial que se entrega es LaTeX (paper/main.tex); los cambios
acordados en Word se pasan a su archivo en paper/secciones/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroflybee.reporting.word_export import export  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", default=str(ROOT / "paper"))
    parser.add_argument("--figures", default=str(ROOT / "results" / "figures"))
    parser.add_argument("--out", default=str(ROOT / "paper" / "NeuroFlyBee_Paper.docx"))
    args = parser.parse_args(argv)
    info = export(Path(args.paper), Path(args.figures), Path(args.out))
    print(f"Word listo: {args.out}  ({info['autores']} autores, {info['figuras']} figuras, "
          f"{info['referencias_cruzadas']} referencias cruzadas)")


if __name__ == "__main__":
    main()
