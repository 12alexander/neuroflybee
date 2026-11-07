"""Tabla de rutas HTTP (Abierto/Cerrado, v18).

El servidor no conoce las rutas: cada módulo las agrega con `register(routes)`. Así el
sabotaje (v20) o el piloto humano (v30) se suman sin editar `http_server.py`.
"""
from __future__ import annotations

import importlib
import json
import pkgutil
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Callable, Optional

from .service import SimulationService


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    content_type: str

    @classmethod
    def json(cls, payload: dict | list, status: int = HTTPStatus.OK) -> "Response":
        return cls(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    @classmethod
    def text(cls, text: str | bytes, content_type: str, status: int = HTTPStatus.OK) -> "Response":
        return cls(status, text if isinstance(text, bytes) else text.encode("utf-8"), content_type)


@dataclass
class RequestContext:
    """Lo que recibe cada ruta: el servicio, los parámetros y recursos compartidos."""

    service: SimulationService
    query: dict[str, str]
    state: dict = field(default_factory=dict)        # p. ej. datos iniciales del visor
    lock: threading.Lock = field(default_factory=threading.Lock)


Handler = Callable[[RequestContext], Response]


class RouteTable:
    def __init__(self) -> None:
        self._routes: dict[str, Handler] = {}

    def add(self, path: str, handler: Handler) -> None:
        if path in self._routes:
            raise ValueError(f"La ruta {path} ya está registrada")
        self._routes[path] = handler

    def get(self, path: str) -> Optional[Handler]:
        return self._routes.get(path)

    def paths(self) -> list[str]:
        return sorted(self._routes)


def load_extensions(routes: RouteTable, package: str = "neuroflybee.app.extensions") -> list[str]:
    """Importa cada módulo de `app/extensions/` y llama a su `register(routes)`."""
    pkg = importlib.import_module(package)
    loaded = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package}.{info.name}")
        register = getattr(module, "register", None)
        if callable(register):
            register(routes)
            loaded.append(info.name)
    return loaded
