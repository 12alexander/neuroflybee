"""Especificación de las redes de la mosca, independiente de PyTorch.

Sirve para tres cosas sin necesitar PyTorch instalado: documentar las dimensiones de
cada bloque, contar parámetros de forma analítica (la prueba `test_fly_models.py`
verifica que coincide con PyTorch) y dibujar el diagrama de la v8.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from ..config import from_dict
from ..domain.drone import N_ACTIONS

FLY_MODEL_NAMES = ("fly_mlp", "fly_cnn", "fly_cnn_gru", "fly_cnn_transformer")
FLY_MODEL_LABELS_ES = {
    "fly_mlp": "MLP (sin convolución, 1 cuadro)",
    "fly_cnn": "CNN (1 cuadro)",
    "fly_cnn_gru": "CNN + GRU (secuencia) — propuesta",
    "fly_cnn_transformer": "CNN + Transformer (secuencia)",
}


@dataclass(frozen=True)
class FlyNetConfig:
    patch_size: int = 32
    seq_len: int = 4                     # cuadros que ve la red (t-3 … t)
    max_seq_len: int = 8                 # tamaño de la tabla de posiciones del Transformer
    channels: tuple[int, ...] = (16, 32, 64)
    feature_dim: int = 128               # vector por cuadro que produce la CNN
    goal_dim: int = 3                    # (cos, sin, distancia) del objetivo relativo
    goal_embed: int = 32
    gru_hidden: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff: int = 256
    mlp_hidden: int = 256
    head_hidden: int = 64
    n_actions: int = N_ACTIONS
    dropout: float = 0.1

    @property
    def encoded_side(self) -> int:
        return self.patch_size // (2 ** len(self.channels))


def load_fly_config(path: Optional[str | Path]) -> FlyNetConfig:
    if path is None:
        return FlyNetConfig()
    with open(path, encoding="utf-8") as fh:
        return from_dict(FlyNetConfig, yaml.safe_load(fh) or {})


# ----------------------------------------------------------------- conteo analítico
def _linear(i: int, o: int) -> int:
    return i * o + o


def _conv3x3(i: int, o: int) -> int:
    return i * o * 9 + o


def count_encoder(c: FlyNetConfig) -> int:
    total, cin = 0, 1
    for cout in c.channels:
        total += _conv3x3(cin, cout) + 2 * cout          # conv + BatchNorm (peso y sesgo)
        cin = cout
    return total + _linear(cin * c.encoded_side ** 2, c.feature_dim)


def count_goal(c: FlyNetConfig) -> int:
    return _linear(c.goal_dim, c.goal_embed)


def count_head(c: FlyNetConfig, context_dim: int) -> int:
    return _linear(context_dim + c.goal_embed, c.head_hidden) + _linear(c.head_hidden, c.n_actions)


def count_gru(c: FlyNetConfig) -> int:
    h, i = c.gru_hidden, c.feature_dim
    return 3 * h * (i + h + 2)                            # pesos ih, hh y dos sesgos por compuerta


def count_transformer(c: FlyNetConfig) -> int:
    d, ff = c.feature_dim, c.transformer_ff
    per_layer = 4 * d * d + 4 * d + 2 * d * ff + ff + d + 4 * d   # atención + FFN + 2 LayerNorm
    return c.max_seq_len * d + c.transformer_layers * per_layer


def count_mlp_body(c: FlyNetConfig) -> int:
    return _linear(c.patch_size ** 2, c.mlp_hidden) + _linear(c.mlp_hidden, c.feature_dim)


def count_parameters(name: str, c: FlyNetConfig) -> int:
    common = count_goal(c)
    if name == "fly_mlp":
        return count_mlp_body(c) + common + count_head(c, c.feature_dim)
    if name == "fly_cnn":
        return count_encoder(c) + common + count_head(c, c.feature_dim)
    if name == "fly_cnn_gru":
        return count_encoder(c) + count_gru(c) + common + count_head(c, c.gru_hidden)
    if name == "fly_cnn_transformer":
        return count_encoder(c) + count_transformer(c) + common + count_head(c, c.feature_dim)
    raise KeyError(f"Modelo desconocido: {name}. Opciones: {FLY_MODEL_NAMES}")


# ----------------------------------------------------------------- bloques para el diagrama
@dataclass(frozen=True)
class Block:
    title: str
    shape: str          # forma del tensor de salida
    params: int
    kind: str = "layer" # input | cnn | temporal | goal | head | output


@dataclass(frozen=True)
class ModelDiagram:
    name: str
    blocks: list[Block] = field(default_factory=list)


def describe(name: str, c: FlyNetConfig) -> ModelDiagram:
    """Bloques de cada red con la forma de sus tensores (B = lote, T = cuadros)."""
    s, f, t = c.patch_size, c.feature_dim, c.seq_len
    side = c.encoded_side
    conv_txt = "→".join(str(ch) for ch in c.channels)
    head = Block(f"Cabeza MLP\n[{c.head_hidden}]+Dropout", f"B×{c.n_actions} logits", 0, "head")
    out = Block("Acción\n(softmax)", f"B×{c.n_actions}", 0, "output")
    goal = Block(f"Objetivo\nLinear {c.goal_dim}→{c.goal_embed}", f"B×{c.goal_embed}", count_goal(c), "goal")
    if name == "fly_mlp":
        blocks = [Block("Entrada\núltimo cuadro", f"B×{s}×{s}", 0, "input"),
                  Block(f"MLP\n{s * s}→{c.mlp_hidden}→{f}", f"B×{f}", count_mlp_body(c), "cnn"),
                  goal, Block(head.title, head.shape, count_head(c, f), "head"), out]
    elif name == "fly_cnn":
        blocks = [Block("Entrada\núltimo cuadro", f"B×1×{s}×{s}", 0, "input"),
                  Block(f"CNN 3 bloques\nconv3×3 {conv_txt}\n+BN+ReLU+MaxPool", f"B×{c.channels[-1]}×{side}×{side}→B×{f}",
                        count_encoder(c), "cnn"),
                  goal, Block(head.title, head.shape, count_head(c, f), "head"), out]
    elif name == "fly_cnn_gru":
        blocks = [Block(f"Entrada\n{t} cuadros", f"B×{t}×1×{s}×{s}", 0, "input"),
                  Block(f"CNN compartida\npor cuadro ({conv_txt})", f"B×{t}×{f}", count_encoder(c), "cnn"),
                  Block(f"GRU\nocultas {c.gru_hidden}", f"B×{c.gru_hidden} (último estado)", count_gru(c), "temporal"),
                  goal, Block(head.title, head.shape, count_head(c, c.gru_hidden), "head"), out]
    elif name == "fly_cnn_transformer":
        blocks = [Block(f"Entrada\n{t} cuadros", f"B×{t}×1×{s}×{s}", 0, "input"),
                  Block(f"CNN compartida\npor cuadro ({conv_txt})", f"B×{t}×{f}", count_encoder(c), "cnn"),
                  Block(f"Transformer\n{c.transformer_layers} capas, {c.transformer_heads} cabezas\n+ posición aprendida",
                        f"B×{f} (último token)", count_transformer(c), "temporal"),
                  goal, Block(head.title, head.shape, count_head(c, f), "head"), out]
    else:
        raise KeyError(name)
    return ModelDiagram(name, blocks)
