"""Especificación de la red de la abeja, independiente de PyTorch (conteo de parámetros
y diagrama). La prueba `tests/test_bee_model.py` verifica que coincide con PyTorch.

Desde v20 la especificación describe también las **variantes de la ablación**:

* `fusion` — cómo se combinan las dos ramas del codificador compartido:
    - ``"siamesa"``       [f_actual, f_base, f_actual − f_base]   (diseño de v11)
    - ``"sin_diferencia"`` [f_actual, f_base]                     (¿aporta la resta explícita?)
    - ``"vista_unica"``    [f_actual]                             (¿hace falta la vista de casa?)
* `head` — cómo se lee la dirección:
    - ``"plana"``      Flatten → Linear → ReLU → Dropout → Linear(3)   (diseño de v11)
    - ``"vectorial"``  cada sector "vota" con un peso y la dirección es la suma vectorial de
      los votos (vector de población, como el ALV), con muy pocos parámetros.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ..config import from_dict

FUSIONS = ("siamesa", "sin_diferencia", "vista_unica")
HEADS = ("plana", "vectorial")
_FUSION_BRANCHES = {"siamesa": 3, "sin_diferencia": 2, "vista_unica": 1}


@dataclass(frozen=True)
class BeeNetConfig:
    """Hiperparámetros de la red de la abeja (ver `configs/bee_model.yaml`)."""

    bins: int = 72                     # direcciones del panorama (5° cada una)
    c1: int = 16
    c2: int = 32
    c3: int = 32
    fusion_channels: int = 64
    hidden: int = 128
    dropout: float = 0.1
    land_distance: float = 0.7         # m: por debajo, la abeja considera que llegó
    fusion: str = "siamesa"            # variante de fusión (ablación v20)
    head: str = "plana"                # variante de cabeza (ablación v20)

    def __post_init__(self) -> None:
        if self.fusion not in FUSIONS:
            raise ValueError(f"fusion='{self.fusion}' no válida; opciones: {FUSIONS}")
        if self.head not in HEADS:
            raise ValueError(f"head='{self.head}' no válida; opciones: {HEADS}")
        if self.bins % 4:
            raise ValueError("bins debe ser múltiplo de 4 (la red aplica dos MaxPool de 2)")

    @property
    def pooled_bins(self) -> int:
        """Sectores tras los dos MaxPool de 2 (72 → 18, es decir, 20° por sector)."""
        return self.bins // 4

    @property
    def fusion_in_channels(self) -> int:
        """Canales que entran a la convolución de fusión según la variante."""
        return _FUSION_BRANCHES[self.fusion] * self.c3


def load_bee_config(path: Optional[str | Path]) -> BeeNetConfig:
    """Lee el YAML de la red; sin ruta devuelve la configuración por defecto (v11)."""
    if path is None:
        return BeeNetConfig()
    with open(path, encoding="utf-8") as fh:
        return from_dict(BeeNetConfig, yaml.safe_load(fh) or {})


def _conv1d(i: int, o: int, k: int) -> int:
    """Parámetros de una Conv1d con sesgo."""
    return i * o * k + o


def count_bee_encoder(c: BeeNetConfig) -> int:
    """Codificador 1D: tres convoluciones con BatchNorm (2 parámetros por canal)."""
    return (_conv1d(1, c.c1, 5) + 2 * c.c1 + _conv1d(c.c1, c.c2, 5) + 2 * c.c2
            + _conv1d(c.c2, c.c3, 3) + 2 * c.c3)


def count_bee_fusion(c: BeeNetConfig) -> int:
    """Convolución de fusión (k=3) con BatchNorm."""
    return _conv1d(c.fusion_in_channels, c.fusion_channels, 3) + 2 * c.fusion_channels


def count_bee_readout(c: BeeNetConfig) -> int:
    """Cabeza de salida: plana (MLP sobre el mapa aplanado) o vectorial (votos por sector)."""
    if c.head == "plana":
        return (c.fusion_channels * c.pooled_bins * c.hidden + c.hidden) + (c.hidden * 3 + 3)
    # vectorial: Conv1d(F → 2, k=1) [voto y rasgo de distancia] + Linear(sectores → 1)
    return _conv1d(c.fusion_channels, 2, 1) + (c.pooled_bins + 1)


def count_bee_head(c: BeeNetConfig) -> int:
    """Todo lo que no es el codificador compartido: fusión + cabeza."""
    return count_bee_fusion(c) + count_bee_readout(c)


def count_bee_parameters(c: BeeNetConfig) -> int:
    """El codificador se cuenta una vez: es COMPARTIDO por las dos ramas (siamesa)."""
    return count_bee_encoder(c) + count_bee_head(c)
