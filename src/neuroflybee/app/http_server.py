"""Adaptador HTTP (biblioteca estándar, sin dependencias) para `SimulationService`.

Desde v18 el servidor sólo despacha: las rutas viven en `core_routes.py` y en los módulos
de `app/extensions/` (cada uno con su `register(routes)`), de modo que agregar una
función nueva no requiere editar este archivo.

Rutas base:
    GET /                         visor HTML en modo en vivo
    GET /datos.js                 simulación inicial para que el visor arranque
    GET /plugins.js               plugins del visor (viz/viewer/plugins/*.js)
    GET /api/catalogo             condiciones, controladores disponibles y rango de semillas
    GET /api/simular?familia=…&semilla=…&controladores=a,b   simula y devuelve los datos del visor
"""
from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import core_routes
from .core_routes import VIEWER_HTML  # noqa: F401  (compatibilidad con v17)
from .routes import RequestContext, Response, RouteTable, load_extensions
from .service import SimulationService


def build_routes(with_extensions: bool = True) -> RouteTable:
    routes = RouteTable()
    core_routes.register(routes)
    if with_extensions:
        load_extensions(routes)
    return routes


def make_handler(service: SimulationService, initial: dict, routes: RouteTable | None = None):
    routes = routes or build_routes()
    shared = {"initial_js": "window.NFB_DATA = " + json.dumps(initial, ensure_ascii=False, separators=(",", ":"))
                            + ";\nwindow.NFB_LIVE = true;\n"}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "NeuroFlyBee/1.1"

        def log_message(self, fmt, *args):          # registro breve en consola
            print(f"[{self.log_date_time_string()}] {fmt % args}")

        def _send(self, response: Response) -> None:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(response.body)

        def do_GET(self) -> None:  # noqa: N802 (nombre impuesto por http.server)
            url = urlparse(self.path)
            handler = routes.get(url.path)
            if handler is None:
                self._send(Response.json({"error": f"Ruta desconocida: {url.path}"}, HTTPStatus.NOT_FOUND))
                return
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            self._send(handler(RequestContext(service, query, shared, lock)))

    return Handler


def create_server(service: SimulationService, host: str = "127.0.0.1", port: int = 8000,
                  initial: dict | None = None, routes: RouteTable | None = None) -> ThreadingHTTPServer:
    initial = initial or {"meta": {}, "scenarios": []}
    return ThreadingHTTPServer((host, port), make_handler(service, initial, routes))
