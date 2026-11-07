"""Red de la abeja (PyTorch): retorno visual con una red siamesa 1D.

    vista actual  ─┐  codificador 1D compartido (relleno CIRCULAR: el panorama da la vuelta)
    vista en base ─┘        ↓                       ↓
                     f_actual,  f_base,  f_actual − f_base   (concatenados por canal)
                                  ↓
            conv 1D + cabeza  →  (cos θ, sin θ, log(1 + d))   dirección y distancia a la base

No se usa *pooling* global al final: el panorama está alineado con la brújula, así que
la POSICIÓN de cada rasgo dentro del vector codifica su dirección.

Desde v20 la fusión y la cabeza son intercambiables (ver `bee_spec.py`) para la ablación;
la configuración por defecto es idéntica a la de v11 y sus checkpoints siguen cargando.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .bee_spec import BeeNetConfig


def _conv(i: int, o: int, k: int) -> list[nn.Module]:
    """Conv1d circular + BatchNorm + ReLU (bloque básico de la red)."""
    return [nn.Conv1d(i, o, k, padding=k // 2, padding_mode="circular"), nn.BatchNorm1d(o), nn.ReLU()]


class PanoramaEncoder(nn.Module):
    """Codificador compartido: (B, bins) → (B, c3, bins/4)."""

    def __init__(self, cfg: BeeNetConfig):
        super().__init__()
        self.net = nn.Sequential(*_conv(1, cfg.c1, 5), *_conv(cfg.c1, cfg.c2, 5), nn.MaxPool1d(2),
                                 *_conv(cfg.c2, cfg.c3, 3), nn.MaxPool1d(2))

    def forward(self, view: torch.Tensor) -> torch.Tensor:
        return self.net(view.unsqueeze(1))


def fuse(mode: str, current: torch.Tensor, home: torch.Tensor | None) -> torch.Tensor:
    """Combina los rasgos de las dos vistas según la variante de fusión.

    * siamesa:        [f_actual, f_base, f_actual − f_base]
    * sin_diferencia: [f_actual, f_base]
    * vista_unica:    [f_actual]   (la vista de casa no se usa)
    """
    if mode == "vista_unica":
        return current
    if home is None:
        raise ValueError(f"La fusión '{mode}' necesita los rasgos de la vista de casa")
    if mode == "sin_diferencia":
        return torch.cat([current, home], dim=1)
    return torch.cat([current, home, current - home], dim=1)


class FlatHead(nn.Sequential):
    """Cabeza de v11: aplana el mapa (canales × sectores) y aplica un MLP.

    Hereda de `nn.Sequential` para conservar los nombres de los pesos de v11
    (`head.1.weight`, `head.4.weight`) y poder cargar sus checkpoints."""

    def __init__(self, cfg: BeeNetConfig):
        super().__init__(nn.Flatten(), nn.Linear(cfg.fusion_channels * cfg.pooled_bins, cfg.hidden),
                         nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, 3))


class VectorReadoutHead(nn.Module):
    """Lectura vectorial (vector de población), variante compacta de la ablación v20.

    Una convolución 1×1 produce, para cada sector k (centro φ_k), un voto w_k y un rasgo de
    distancia r_k. La dirección es la suma vectorial Σ_k w_k (cos φ_k, sin φ_k) —la misma idea
    que el ALV, pero con pesos aprendidos— y la distancia es una combinación lineal de r_k.
    Conserva la geometría del panorama con ~150 parámetros en lugar de ~148 000."""

    def __init__(self, cfg: BeeNetConfig):
        super().__init__()
        self.score = nn.Conv1d(cfg.fusion_channels, 2, kernel_size=1)
        self.distance = nn.Linear(cfg.pooled_bins, 1)
        n = cfg.pooled_bins
        centers = torch.tensor([-math.pi + (k + 0.5) * 2 * math.pi / n for k in range(n)])
        # (2, sectores): vectores unitarios de cada sector; no se entrena ni se guarda
        self.register_buffer("unit", torch.stack([centers.cos(), centers.sin()]), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:      # (B, F, sectores) → (B, 3)
        s = self.score(features)
        direction = s[:, 0] @ self.unit.T                              # (B, 2)
        distance = self.distance(s[:, 1])                              # (B, 1)
        return torch.cat([direction, distance], dim=1)


def build_head(cfg: BeeNetConfig) -> nn.Module:
    """Fábrica de cabezas (abierta a nuevas variantes sin tocar `BeeHomingNet`)."""
    return FlatHead(cfg) if cfg.head == "plana" else VectorReadoutHead(cfg)


class BeeHomingNet(nn.Module):
    """Red de retorno: (vista actual, vista en la base) → (cos θ, sin θ, log(1 + d))."""

    def __init__(self, cfg: BeeNetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or BeeNetConfig()
        self.encoder = PanoramaEncoder(cfg)
        self.fusion = nn.Sequential(*_conv(cfg.fusion_in_channels, cfg.fusion_channels, 3))
        self.head = build_head(cfg)

    def forward(self, current: torch.Tensor, home: torch.Tensor) -> torch.Tensor:
        fc = self.encoder(current.float())
        # la variante de vista única no necesita (ni calcula) los rasgos de la base
        fh = None if self.cfg.fusion == "vista_unica" else self.encoder(home.float())
        return self.head(self.fusion(fuse(self.cfg.fusion, fc, fh)))

    @staticmethod
    def decode(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Salida cruda → (ángulo en rad, distancia en m)."""
        angle = torch.atan2(output[:, 1], output[:, 0])
        distance = torch.expm1(output[:, 2].clamp(min=0.0))
        return angle, distance


class BeeHomingLoss(nn.Module):
    """L = (1 − cos(θ̂, θ)) + λ · SmoothL1(log(1+d̂), log(1+d))."""

    def __init__(self, distance_weight: float = 0.5):
        super().__init__()
        self.distance_weight = distance_weight

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        direction = 1.0 - F.cosine_similarity(output[:, :2], target[:, :2], dim=1, eps=1e-6)
        distance = F.smooth_l1_loss(output[:, 2], target[:, 2], reduction="none")
        return (direction + self.distance_weight * distance).mean()
