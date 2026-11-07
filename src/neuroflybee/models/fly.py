"""Redes de la mosca (PyTorch).

Todas comparten la misma firma — `forward(frames, goal) -> logits` — para que el
entrenador, la evaluación y el controlador las usen sin saber cuál es (sustitución de
Liskov). La clase base fija el esquema (método plantilla):

    contexto visual (depende del modelo)  ⊕  objetivo codificado  →  cabeza MLP  →  6 logits

Formas: `frames` (B, T, S, S) en {0, 1}; `goal` (B, 3); salida (B, n_acciones).
Los modelos de un solo cuadro usan `frames[:, -1]`.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .spec import FLY_MODEL_NAMES, FlyNetConfig


class ConvEncoder(nn.Module):
    """CNN tipo LeNet/VGG pequeña: 3 bloques conv3×3 + BatchNorm + ReLU + MaxPool."""

    def __init__(self, cfg: FlyNetConfig):
        super().__init__()
        layers: list[nn.Module] = []
        cin = 1
        for cout in cfg.channels:
            layers += [nn.Conv2d(cin, cout, kernel_size=3, padding=1), nn.BatchNorm2d(cout),
                       nn.ReLU(), nn.MaxPool2d(2)]
            cin = cout
        self.features = nn.Sequential(*layers)
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(cin * cfg.encoded_side ** 2, cfg.feature_dim),
                                     nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (N, 1, S, S) → (N, F)
        return self.project(self.features(x))

    @property
    def last_conv(self) -> nn.Conv2d:
        """Última convolución: la capa que usa Grad-CAM (v15)."""
        return [m for m in self.features if isinstance(m, nn.Conv2d)][-1]


class PolicyHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_actions: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlyPolicyNet(nn.Module):
    """Base: subclases implementan `visual_context` y declaran `context_dim`."""

    uses_sequence: bool = True

    def __init__(self, cfg: FlyNetConfig, context_dim: int):
        super().__init__()
        self.cfg = cfg
        self.goal_encoder = nn.Sequential(nn.Linear(cfg.goal_dim, cfg.goal_embed), nn.ReLU())
        self.head = PolicyHead(context_dim + cfg.goal_embed, cfg.head_hidden, cfg.n_actions, cfg.dropout)

    def visual_context(self, frames: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstracto
        raise NotImplementedError

    def forward(self, frames: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 4:
            raise ValueError(f"frames debe ser (B, T, S, S); llegó {tuple(frames.shape)}")
        context = self.visual_context(frames.float())
        return self.head(torch.cat([context, self.goal_encoder(goal.float())], dim=-1))

    @torch.no_grad()
    def predict_proba(self, frames: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(frames, goal), dim=-1)


class _SequenceEncoderMixin:
    encoder: ConvEncoder

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:   # (B, T, S, S) → (B, T, F)
        b, t, s1, s2 = frames.shape
        features = self.encoder(frames.reshape(b * t, 1, s1, s2))
        return features.reshape(b, t, -1)


class FlyMLP(FlyPolicyNet):
    """Línea base neuronal sin convolución: perceptrón multicapa sobre el último cuadro."""

    uses_sequence = False

    def __init__(self, cfg: FlyNetConfig):
        super().__init__(cfg, cfg.feature_dim)
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(cfg.patch_size ** 2, cfg.mlp_hidden), nn.ReLU(),
                                  nn.Linear(cfg.mlp_hidden, cfg.feature_dim), nn.ReLU())

    def visual_context(self, frames: torch.Tensor) -> torch.Tensor:
        return self.body(frames[:, -1])


class FlyCNN(FlyPolicyNet):
    """CNN sobre el último cuadro (sin memoria temporal)."""

    uses_sequence = False

    def __init__(self, cfg: FlyNetConfig):
        super().__init__(cfg, cfg.feature_dim)
        self.encoder = ConvEncoder(cfg)

    def visual_context(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames[:, -1:])      # (B, 1, S, S): el cuadro actual como canal


class FlyCNNGRU(_SequenceEncoderMixin, FlyPolicyNet):
    """Propuesta: CNN compartida por cuadro + GRU que integra el movimiento en el tiempo."""

    def __init__(self, cfg: FlyNetConfig):
        super().__init__(cfg, cfg.gru_hidden)
        self.encoder = ConvEncoder(cfg)
        self.gru = nn.GRU(cfg.feature_dim, cfg.gru_hidden, num_layers=1, batch_first=True)

    def visual_context(self, frames: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(self.encode_frames(frames))
        return hidden[-1]                        # (B, H): estado tras el último cuadro


class FlyCNNTransformer(_SequenceEncoderMixin, FlyPolicyNet):
    """Variante contemporánea: CNN compartida + codificador Transformer sobre los cuadros."""

    def __init__(self, cfg: FlyNetConfig):
        super().__init__(cfg, cfg.feature_dim)
        self.encoder = ConvEncoder(cfg)
        self.positions = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.feature_dim))
        nn.init.normal_(self.positions, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=cfg.feature_dim, nhead=cfg.transformer_heads,
                                           dim_feedforward=cfg.transformer_ff, dropout=cfg.dropout,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.transformer_layers,
                                                 enable_nested_tensor=False)

    def visual_context(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.encode_frames(frames)
        t = x.shape[1]
        if t > self.cfg.max_seq_len:
            raise ValueError(f"Secuencia de {t} cuadros > max_seq_len={self.cfg.max_seq_len}")
        z = self.transformer(x + self.positions[:, :t])
        return z[:, -1]                          # salida del token del cuadro actual


_FACTORIES: dict[str, Callable[[FlyNetConfig], FlyPolicyNet]] = {
    "fly_mlp": FlyMLP,
    "fly_cnn": FlyCNN,
    "fly_cnn_gru": FlyCNNGRU,
    "fly_cnn_transformer": FlyCNNTransformer,
}
assert set(_FACTORIES) == set(FLY_MODEL_NAMES)


def build_fly_model(name: str, cfg: FlyNetConfig | None = None) -> FlyPolicyNet:
    if name not in _FACTORIES:
        raise KeyError(f"Modelo desconocido: {name}. Opciones: {list(_FACTORIES)}")
    return _FACTORIES[name](cfg or FlyNetConfig())


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
