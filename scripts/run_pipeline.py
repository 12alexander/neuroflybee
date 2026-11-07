"""Pipeline de resultados reales (v19): UN comando en Colab produce todo lo del artículo.

    python scripts/run_pipeline.py              # completo (≈ 1–2 h en una T4, estimado)
    python scripts/run_pipeline.py --quick      # prueba de punta a punta (1 semilla, 10 episodios)
    python scripts/run_pipeline.py --dry-run    # sólo muestra qué correría
    python scripts/run_pipeline.py --only benchmark,numeros
    python scripts/run_pipeline.py --video      # incluye el video de respaldo

Registra el tiempo y las horas de GPU de cada etapa en results/v19_costo.csv (R6) y guarda
el avance: si Colab se desconecta, vuelva a ejecutar y continúa donde quedó.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroflybee.pipeline.runner import PipelineRunner  # noqa: E402
from neuroflybee.pipeline.stages import default_stages, select  # noqa: E402


def _names(text: str | None) -> list[str]:
    return [t for t in (text or "").split(",") if t]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="repite etapas ya completadas")
    parser.add_argument("--only", help="etapas separadas por comas")
    parser.add_argument("--skip", help="etapas a omitir")
    parser.add_argument("--video", action="store_true", help="incluye la grabación del video")
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args(argv)
    stages = select(default_stages(), _names(args.only), _names(args.skip), ["video"] if args.video else [])
    runner = PipelineRunner(ROOT, Path(args.results), quick=args.quick, force=args.force)
    return 0 if runner.run(stages, dry_run=args.dry_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
