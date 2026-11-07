"""Extensión mínima (v18): `/api/estado` informa versión, rutas y condiciones registradas.

Sirve de ejemplo real del mecanismo de extensiones y de comprobación rápida del servidor.
"""
from __future__ import annotations

from ... import __version__
from ...contracts import conditions_registry
from ..routes import RequestContext, Response, RouteTable

_ROUTES: list[RouteTable] = []


def status(ctx: RequestContext) -> Response:
    return Response.json({"version": __version__, "rutas": _ROUTES[-1].paths() if _ROUTES else [],
                          "condiciones": conditions_registry.describe(),
                          "controladores": list(ctx.service.available())})


def register(routes: RouteTable) -> None:
    _ROUTES.append(routes)
    routes.add("/api/estado", status)
