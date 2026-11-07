"""Registro de los controladores neuronales (requieren PyTorch y pesos entrenados).

| nombre                 | navegador (abeja)            | control local      | supervisor |
|------------------------|------------------------------|--------------------|------------|
| fly                    | ruta + estimación (sin red)  | red de la mosca    | no         |
| bee                    | ruta + RED de retorno visual | A* (clásico)       | no         |
| hybrid                 | ruta + RED de retorno visual | red de la mosca    | sí         |
| hybrid_no_supervisor   | ruta + RED de retorno visual | red de la mosca    | no         |

`bee` usa el mismo control local que `astar` y `astar_alv`: así la comparación aísla el
aporte del retorno visual neuronal frente al ALV clásico.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from .registry import register

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def default_checkpoint_dir() -> Path:
    return Path(os.environ.get("NEUROFLYBEE_CHECKPOINTS", PROJECT_ROOT / "checkpoints"))


def find_fly_checkpoint(directory: Path, seed: int = 0, model: str = "fly_cnn_gru") -> Optional[Path]:
    """Prefiere la última iteración de DAgger; si no hay, el modelo de imitación (v9)."""
    dagger = sorted(directory.glob(f"{model}_dagger*_s{seed}.pt"),
                    key=lambda p: int(re.search(r"dagger(\d+)_", p.name).group(1)))
    if dagger:
        return dagger[-1]
    plain = directory / f"{model}_s{seed}.pt"
    return plain if plain.exists() else None


def find_bee_checkpoint(directory: Path, seed: int = 0) -> Optional[Path]:
    path = directory / f"bee_s{seed}.pt"
    return path if path.exists() else None


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _load(kind: str, path: Path, device: str):
    """Carga (y guarda en caché) un modelo: evita releer el archivo en cada episodio."""
    key = (kind, str(path))
    if key not in _MODEL_CACHE:
        if kind == "fly":
            from ..training.checkpoints import load_fly_checkpoint
            _MODEL_CACHE[key] = load_fly_checkpoint(path, device)[0]
        else:
            from ..training.bee import load_bee_checkpoint
            _MODEL_CACHE[key] = load_bee_checkpoint(path, device)[0]
    return _MODEL_CACHE[key]


def neural_checkpoints(name: str, directory: Path, seed: int, fly_model: str = "fly_cnn_gru",
                       dagger: bool = True) -> Optional[dict[str, Path]]:
    """Pesos que necesita un controlador neuronal, o None si falta alguno."""
    need_fly = name in ("fly", "hybrid", "hybrid_no_supervisor")
    need_bee = name in ("bee", "hybrid", "hybrid_no_supervisor")
    paths: dict[str, Path] = {}
    if need_fly:
        fly = find_fly_checkpoint(directory, seed, fly_model) if dagger else directory / f"{fly_model}_s{seed}.pt"
        if fly is None or not fly.exists():
            return None
        paths["fly"] = fly
    if need_bee:
        bee = find_bee_checkpoint(directory, seed)
        if bee is None:
            return None
        paths["bee"] = bee
    return paths


def build_neural_controller(name: str, checkpoint_dir: Optional[str | Path] = None, seed: int = 0,
                            fly_model: str = "fly_cnn_gru", dagger: bool = True, device: str = "cpu"):
    """Construye un controlador neuronal concreto (variante de modelo, semilla, con/sin DAgger)."""
    if not _torch_available():
        raise RuntimeError("Se necesita PyTorch para los controladores neuronales")
    from ..inference.bee_homing import BeeNetHoming
    from ..inference.fly_runner import FlyPolicyRunner
    from .composite import NavigationController
    from .navigation import MissionNavigator
    from .neural import FlySteering
    from .safety import SafetySupervisor
    from .steering import AStarSteering

    directory = Path(checkpoint_dir) if checkpoint_dir else default_checkpoint_dir()
    paths = neural_checkpoints(name, directory, seed, fly_model, dagger)
    if paths is None:
        raise FileNotFoundError(f"Faltan pesos para '{name}' (semilla {seed}) en {directory}")
    fly = lambda: FlySteering(FlyPolicyRunner(_load("fly", paths["fly"], device), device))  # noqa: E731
    homing = lambda: BeeNetHoming(_load("bee", paths["bee"], device), device)  # noqa: E731
    if name == "fly":
        return NavigationController("fly", MissionNavigator(), fly())
    if name == "bee":
        return NavigationController("bee", MissionNavigator(homing()), AStarSteering())
    if name == "hybrid_no_supervisor":
        return NavigationController("hybrid_no_supervisor", MissionNavigator(homing()), fly())
    if name == "hybrid":
        return SafetySupervisor(NavigationController("hybrid", MissionNavigator(homing()), fly()), name="hybrid")
    raise KeyError(f"Controlador neuronal desconocido: {name}")


NEURAL_NAMES = ("fly", "bee", "hybrid", "hybrid_no_supervisor")
NEURAL_DESCRIPTIONS = {
    "fly": "Mosca: CNN+GRU", "bee": "Abeja: retorno visual neuronal + A*",
    "hybrid": "Híbrido: abeja + mosca + supervisor", "hybrid_no_supervisor": "Híbrido sin supervisor (ablación)",
}


def register_neural_controllers(checkpoint_dir: Optional[str | Path] = None, seed: int = 0,
                                device: str = "cpu") -> list[str]:
    """Registra en el catálogo los controladores para los que hay pesos. Devuelve sus nombres."""
    if not _torch_available():
        return []
    directory = Path(checkpoint_dir) if checkpoint_dir else default_checkpoint_dir()
    names = [n for n in NEURAL_NAMES if neural_checkpoints(n, directory, seed) is not None]
    for n in names:
        register(n, NEURAL_DESCRIPTIONS[n])(
            lambda n=n: build_neural_controller(n, directory, seed, device=device))
    return names
