"""Simulación en vivo (v17): servicio de aplicación y servidor HTTP."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from neuroflybee.app.http_server import create_server
from neuroflybee.app.service import SimulationService, ValidationError
from neuroflybee.config import load_config
from neuroflybee.control import classical  # noqa: F401  (registra los controladores clásicos)


@pytest.fixture(scope="module")
def service():
    return SimulationService(load_config("configs/default.yaml"))


def test_catalog_lists_families_controllers_and_test_seeds(service):
    cat = service.catalog()
    assert {f["id"] for f in cat["families"]} >= {"nominal", "dynamic", "gps_loss", "combined"}
    assert "astar" in {c["id"] for c in cat["controllers"]}
    assert (cat["seed_min"], cat["seed_max"]) == (20000, 20999)


@pytest.mark.parametrize("family, seed, ctrls", [
    ("lluvia", 20000, "astar"),            # condición inexistente
    ("dynamic", "abc", "astar"),           # semilla no numérica
    ("dynamic", 5, "astar"),               # semilla de ENTRENAMIENTO: prohibida
    ("dynamic", 20000, ""),                # ningún controlador
    ("dynamic", 20000, "astar,direct,potential_field,astar_alv,direct_supervised"),  # más de 4
    ("dynamic", 20000, "astar,teletransporte"),  # desconocido
])
def test_parse_rejects_invalid_requests(service, family, seed, ctrls):
    with pytest.raises(ValidationError):
        service.parse(family, seed, ctrls)


def test_parse_deduplicates_controllers(service):
    req = service.parse("nominal", "20001", "astar,astar,direct")
    assert req.controllers == ("astar", "direct") and req.seed == 20001


def test_simulate_returns_viewer_data(service):
    data = service.simulate(service.parse("nominal", 20000, ["direct"]))
    sc = data["scenarios"][0]
    assert sc["seed"] == 20000 and len(sc["runs"]) == 1 and sc["runs"][0]["steps"]
    assert data["meta"]["elapsed_s"] >= 0


@pytest.fixture(scope="module")
def server(service):
    initial = service.simulate(service.parse("nominal", 20000, ["direct"]))
    srv = create_server(service, "127.0.0.1", 0, initial)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); srv.server_close()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_http_routes(server):
    status, body = _get(server + "/")
    assert status == 200 and b"initLive" in body
    status, body = _get(server + "/datos.js")
    assert status == 200 and b"window.NFB_LIVE = true" in body
    status, body = _get(server + "/api/catalogo")
    assert status == 200 and json.loads(body)["seed_min"] == 20000
    status, body = _get(server + "/api/simular?familia=dynamic&semilla=20003&controladores=astar")
    assert status == 200 and json.loads(body)["scenarios"][0]["family"] == "dynamic"
    status, body = _get(server + "/api/simular?familia=dynamic&semilla=7&controladores=astar")
    assert status == 400 and "PRUEBA" in json.loads(body)["error"]
    assert _get(server + "/no-existe")[0] == 404
