"""Construye el visor HTML de la simulación (v13).

Hace volar los controladores en los escenarios elegidos y guarda cada paso en
`results/visor/datos.js`; `results/visor/index.html` lo reproduce (doble clic, sin
servidor ni internet). Si hay PyTorch y pesos en `checkpoints/`, agrega automáticamente
los controladores neuronales (mosca, abeja e híbrido).

Uso:
    python scripts/build_viewer.py
    python scripts/build_viewer.py --families dynamic,gps_loss --seed 20003
    python scripts/build_viewer.py --controllers astar,fly,bee,hybrid
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroflybee.config import load_config  # noqa: E402
from neuroflybee.control import classical  # noqa: E402,F401
from neuroflybee.control import registry  # noqa: E402
from neuroflybee.control.neural_registry import register_neural_controllers  # noqa: E402
from neuroflybee.sim.scenarios import FAMILIES, ScenarioGenerator  # noqa: E402
from neuroflybee.app.core_routes import plugins_bundle  # noqa: E402
from neuroflybee.viz.export import build_viewer_data, write_viewer_data  # noqa: E402

TEMPLATE = ROOT / "src" / "neuroflybee" / "viz" / "viewer" / "index.html"
CLASSICAL = ["astar", "potential_field", "astar_alv", "direct_supervised"]
NEURAL = ["fly", "bee", "hybrid"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--seed", type=int, default=20_000, help="semilla de PRUEBA")
    parser.add_argument("--controllers", default=None, help="por defecto: A* + neuronales si hay pesos")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--out", default=str(ROOT / "results" / "visor"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    neural = register_neural_controllers(args.checkpoints)
    if args.controllers:
        controllers = args.controllers.split(",")
    else:
        controllers = (["astar"] + [n for n in NEURAL if n in neural]) if neural else CLASSICAL
    missing = [c for c in controllers if c not in registry.available()]
    if missing:
        raise SystemExit(f"Controladores no disponibles: {missing}. Hay: {registry.available()}")
    gen = ScenarioGenerator(cfg)
    scenarios = [gen.generate(f, args.seed) for f in args.families.split(",")]
    print(f"Simulando {len(controllers)} controladores × {len(scenarios)} escenarios: {controllers}")
    data = build_viewer_data(cfg, scenarios, controllers, registry.create)
    out = Path(args.out)
    write_viewer_data(data, out / "datos.js")
    shutil.copyfile(TEMPLATE, out / "index.html")
    (out / "plugins.js").write_text(plugins_bundle(), encoding="utf-8")   # v18
    size = (out / "datos.js").stat().st_size / 1e6
    print(f"Visor listo: abra {out / 'index.html'}  (datos {size:.1f} MB)")
    for sc in data["scenarios"]:
        print(f"  {sc['label']:28s} " + " | ".join(f"{r['label']}: {r['outcome']['text']}" for r in sc["runs"]))


if __name__ == "__main__":
    main()
