"""Integrated Gradients (Sundararajan et al., 2017) para la red de la abeja.

Responde: ¿qué direcciones del horizonte usó la abeja para decidir hacia dónde está la
base? La línea de referencia es la vista MEMORIZADA en la base (si la vista actual fuera
igual, la red estaría "en casa"), así la atribución explica qué DIFERENCIAS importaron.
Incluye una prueba de fidelidad por borrado: si se reemplazan por la referencia las
direcciones más atribuidas, la predicción debería cambiar más que borrando direcciones al azar.
"""
from __future__ import annotations

import numpy as np
import torch

from ..evaluation.metrics import angular_errors_deg
from ..models.bee import BeeHomingNet


def _direction_score(model: BeeHomingNet, current: torch.Tensor, home: torch.Tensor,
                     direction: torch.Tensor) -> torch.Tensor:
    """Proyección de la salida sobre la dirección predicha original (escalar a explicar)."""
    out = model(current, home)
    return out[:, 0] * direction[0] + out[:, 1] * direction[1]


def integrated_gradients(model: BeeHomingNet, current: torch.Tensor, home: torch.Tensor,
                         steps: int = 64) -> tuple[np.ndarray, float, float]:
    """current, home: (bins,). Devuelve (atribución por bin, f(x), f(referencia))."""
    model.eval()
    cur, hom = current.float()[None], home.float()[None]
    with torch.no_grad():
        out = model(cur, hom)[0]
        direction = out[:2] / out[:2].norm().clamp(min=1e-8)
    alphas = torch.linspace(0.0, 1.0, steps + 1)[1:, None]           # regla de Riemann (derecha)
    path = (hom + alphas * (cur - hom)).requires_grad_(True)
    score = _direction_score(model, path, hom.expand_as(path), direction)
    grads, = torch.autograd.grad(score.sum(), path)
    attribution = ((cur - hom)[0] * grads.mean(dim=0)).detach().numpy()
    with torch.no_grad():
        fx = float(_direction_score(model, cur, hom, direction)[0])
        fb = float(_direction_score(model, hom, hom, direction)[0])
    return attribution, fx, fb


@torch.no_grad()
def deletion_test(model: BeeHomingNet, current: np.ndarray, home: np.ndarray, attribution: np.ndarray,
                  fractions: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Cambio angular (°) de la predicción al reemplazar por la referencia una fracción de
    bins: los más atribuidos (|IG|) frente a bins al azar."""
    model.eval()
    def angle(view: np.ndarray) -> float:
        out = model(torch.from_numpy(view[None]).float(), torch.from_numpy(home[None]).float())[0]
        return float(torch.atan2(out[1], out[0]))
    base = angle(current)
    order = np.argsort(-np.abs(attribution))
    top, rand = [], []
    for f in fractions:
        k = int(round(f * len(current)))
        for idx, acc in ((order[:k], top), (rng.permutation(len(current))[:k], rand)):
            view = current.copy()
            view[idx] = home[idx]
            acc.append(float(angular_errors_deg(angle(view), base)))
    return np.array(top), np.array(rand)
