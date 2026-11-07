"""Validación del sensor panorámico 2,5D de la abeja (v21). No necesita PyTorch.

Comprueba en escenarios de PRUEBA que la silueta de horizonte simulada se comporta como los
panoramas reales: (a) la diferencia con la vista de casa crece con la distancia (IDF de
Zeil et al., 2003), (b) girar el panorama permite recuperar el rumbo y (c) el ALV clásico
apunta a casa en la mayor parte del entorno (área de captura).

Uso:
    python scripts/validate_sensor.py                 # 30 escenarios (configs/sensor_validation.yaml)
    python scripts/validate_sensor.py --scenarios 4   # rápido
Genera:
    results/v21_validacion_sensor.csv   (por escenario)
    results/v21_idf.csv                 (curva media ± d.e.)
    results/v21_validacion_resumen.json
    results/figures/v21_validacion_sensor.png
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroflybee.config import load_config  # noqa: E402
from neuroflybee.evaluation.sensor_validation import (load_sensor_validation_config, run_validation,  # noqa: E402
                                                      summarize_validation, write_outputs)
from neuroflybee.viz.sensor_validation_plot import plot_sensor_validation  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--validation-config", default=str(ROOT / "configs" / "sensor_validation.yaml"))
    parser.add_argument("--scenarios", type=int, default=None, help="reemplaza el número de escenarios")
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    vcfg = load_sensor_validation_config(args.validation_config)
    if args.scenarios is not None:
        vcfg = dataclasses.replace(vcfg, scenarios=args.scenarios)
    t0 = time.time()
    rows = run_validation(cfg, vcfg)
    summary = summarize_validation(rows, vcfg.arrive_rms)
    results = Path(args.results)
    write_outputs(rows, summary, results)
    plot_sensor_validation(cfg, vcfg, rows, summary, results / "figures" / "v21_validacion_sensor.png")
    print(f"{summary.scenarios} escenarios en {time.time() - t0:.0f} s")
    print(f"IDF: ρ de Spearman medio {summary.spearman_mean:.3f}; monótona (ρ > 0.9) en "
          f"{summary.monotonic_fraction:.0%} de los escenarios; piso de ruido {summary.noise_floor:.4f}; "
          f"supera el umbral de llegada a {summary.arrival_radius_m:.2f} m")
    print("Rumbo recuperado (error mediano): " + ", ".join(f"{k} m → {v:.1f}°" for k, v in summary.heading_error.items()))
    print(f"ALV: error mediano {summary.alv_error_median:.1f}°, área de captura {summary.alv_catchment_mean:.0%}")


if __name__ == "__main__":
    main()
