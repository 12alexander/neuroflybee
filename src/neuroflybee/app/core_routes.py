"""Rutas base del servidor (v17): visor, datos iniciales, plugins, catálogo y simulación."""
from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

from .routes import RequestContext, Response, RouteTable
from .service import ValidationError

VIEWER_DIR = Path(__file__).resolve().parents[1] / "viz" / "viewer"
VIEWER_HTML = VIEWER_DIR / "index.html"


def plugins_bundle(directory: Path = VIEWER_DIR / "plugins") -> str:
    """Une los plugins del visor (`viz/viewer/plugins/*.js`) en un solo archivo."""
    parts = [f"// ---- {p.name}\n{p.read_text(encoding='utf-8')}" for p in sorted(directory.glob("*.js"))]
    return "\n".join(parts) + "\n"


def index(ctx: RequestContext) -> Response:
    return Response.text(VIEWER_HTML.read_bytes(), "text/html; charset=utf-8")


def initial_data(ctx: RequestContext) -> Response:
    return Response.text(ctx.state.get("initial_js", "window.NFB_DATA = null;\n"), "application/javascript; charset=utf-8")


def plugins(ctx: RequestContext) -> Response:
    return Response.text(plugins_bundle(), "application/javascript; charset=utf-8")


def catalog(ctx: RequestContext) -> Response:
    return Response.json(ctx.service.catalog())


def simulate(ctx: RequestContext) -> Response:
    q = ctx.query
    try:
        request = ctx.service.parse(q.get("familia", ""), q.get("semilla", ""), q.get("controladores", ""))
    except ValidationError as exc:
        return Response.json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    with ctx.lock:                       # la simulación es CPU: se atiende una a la vez
        return Response.json(ctx.service.simulate(request))


def register(routes: RouteTable) -> None:
    routes.add("/", index)
    routes.add("/index.html", index)
    routes.add("/datos.js", initial_data)
    routes.add("/plugins.js", plugins)
    routes.add("/api/catalogo", catalog)
    routes.add("/api/simular", simulate)
