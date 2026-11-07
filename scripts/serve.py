"""Simulación EN VIVO en el navegador (v17).

Levanta un servidor local: en el visor aparece un panel para elegir condición, semilla de
prueba y hasta 4 controladores; al pulsar «Simular» el dron vuela al momento. Si hay
PyTorch y pesos, también se pueden elegir mosca, abeja e híbrido.

Uso:
    python scripts/serve.py                 # http://127.0.0.1:8000
    python scripts/serve.py --port 8080 --no-browser
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroflybee.app.http_server import build_routes, create_server  # noqa: E402
from neuroflybee.app.service import SimulationService  # noqa: E402
from neuroflybee.config import load_config  # noqa: E402
from neuroflybee.control import classical  # noqa: E402,F401
from neuroflybee.control.neural_registry import register_neural_controllers  # noqa: E402


def default_controllers(available: list[str]) -> list[str]:
    neural = [n for n in ("fly", "bee", "hybrid") if n in available]
    return ["astar", *neural] if neural else ["astar", "potential_field", "astar_alv", "direct_supervised"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    neural = register_neural_controllers(args.checkpoints)
    service = SimulationService(load_config(args.config))
    print("Controladores disponibles:", ", ".join(service.available()))
    print("Preparando la simulación inicial…")
    initial = service.simulate(service.parse("dynamic", 20_000, default_controllers(list(service.available()))))
    routes = build_routes()
    print("Rutas:", ", ".join(routes.paths()))
    server = create_server(service, args.host, args.port, initial, routes)
    url = f"http://{args.host}:{args.port}/"
    print(f"Simulación en vivo en {url}  (Ctrl+C para detener)"
          + ("" if neural else "\n  Sin pesos neuronales: sólo controladores clásicos (entrene en Colab para agregar mosca/abeja/híbrido)."))
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
