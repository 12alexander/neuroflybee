"""Plantilla de extensión (no se carga porque el nombre empieza con «_»).

Copie este archivo como `mi_modulo.py` en esta carpeta; su ruta aparecerá sola en el servidor.
"""
from __future__ import annotations

from ..routes import RequestContext, Response, RouteTable


def hello(ctx: RequestContext) -> Response:
    return Response.json({"mensaje": "hola", "parametros": ctx.query})


def register(routes: RouteTable) -> None:
    routes.add("/api/ejemplo", hello)
