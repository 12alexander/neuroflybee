"""Puntos de extensión de v18: rutas HTTP y plugins del visor."""
import json
import threading
import urllib.request

import pytest

from neuroflybee.app.core_routes import plugins_bundle
from neuroflybee.app.http_server import build_routes, create_server
from neuroflybee.app.routes import RouteTable, load_extensions
from neuroflybee.app.service import SimulationService
from neuroflybee.config import ProjectConfig
from neuroflybee.control import classical  # noqa: F401


def test_route_table_rejects_duplicates():
    routes = RouteTable()
    routes.add("/x", lambda ctx: None)
    with pytest.raises(ValueError):
        routes.add("/x", lambda ctx: None)


def test_extensions_are_discovered_but_templates_are_not():
    routes = RouteTable()
    loaded = load_extensions(routes)
    assert "estado" in loaded and "_ejemplo" not in loaded
    assert "/api/estado" in routes.paths() and "/api/ejemplo" not in routes.paths()


def test_core_routes_kept():
    assert {"/", "/datos.js", "/plugins.js", "/api/catalogo", "/api/simular"} <= set(build_routes().paths())


def test_plugins_bundle_contains_example_plugin():
    js = plugins_bundle()
    assert "NFB.use(" in js and "coordenadas" in js


def test_server_serves_extensions_and_plugins():
    srv = create_server(SimulationService(ProjectConfig()), "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        estado = json.load(urllib.request.urlopen(base + "/api/estado", timeout=30))
        assert "/api/estado" in estado["rutas"] and "escenario" in estado["condiciones"]
        assert b"coordenadas" in urllib.request.urlopen(base + "/plugins.js", timeout=30).read()
    finally:
        srv.shutdown(); srv.server_close()
