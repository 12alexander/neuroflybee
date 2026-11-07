"""Retorno visual a la base (módulo "abeja").

Un `HomingProvider` estima, sólo a partir del panorama actual y del panorama
memorizado en la base, la dirección (y si puede, la distancia) hacia la base.
* `AverageLandmarkVectorHoming`: método clásico (Lambrinos et al., 2000) — línea base.
* La versión neuronal (`BeeNetHoming`) se agrega con el modelo en v11/v12.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from ..sim.observation import Observation


@dataclass
class HomingEstimate:
    direction: float                 # ángulo (rad, marco del mundo) hacia la base
    distance: Optional[float]        # m; None si el método no la estima
    at_home: bool                    # ¿el método considera que ya llegó?
    confidence: float = 1.0


class HomingProvider(Protocol):
    name: str

    def estimate(self, obs: Observation) -> Optional[HomingEstimate]: ...


def panorama_angles(bins: int) -> np.ndarray:
    """Centro (rad, marco del mundo) de cada sector del panorama alineado con la brújula."""
    return np.linspace(-np.pi, np.pi, bins, endpoint=False) + np.pi / bins


def alv_directions(current: np.ndarray, home: np.ndarray) -> np.ndarray:
    """ALV vectorizado: dirección (rad) a la base para N pares de vistas (N, bins).

    Es la misma regla que `AverageLandmarkVectorHoming`, aplicada a un conjunto completo
    (la usan la figura de v11 y la ablación de v20 como línea base clásica)."""
    current, home = np.atleast_2d(current), np.atleast_2d(home)
    ang = panorama_angles(current.shape[1])
    unit = np.stack([np.cos(ang), np.sin(ang)], axis=1)          # (bins, 2)
    diff = home @ unit - current @ unit                           # ALV(base) − ALV(actual)
    return np.arctan2(diff[:, 1], diff[:, 0])


class AverageLandmarkVectorHoming:
    """ALV: vector promedio de la silueta ponderado por la elevación angular. Al
    alejarse de la base, los objetos del lado hacia el que se movió se ven más altos;
    por eso ALV(base) − ALV(actual) apunta aproximadamente a la base. La llegada se decide con la función
    de diferencia de imágenes (RMS) respecto del panorama memorizado (Zeil et al., 2003)."""

    name = "alv"

    def __init__(self, arrive_rms: float = 0.03):
        self.arrive_rms = arrive_rms

    @staticmethod
    def alv(view: np.ndarray) -> np.ndarray:
        ang = panorama_angles(len(view))
        return np.array([(view * np.cos(ang)).sum(), (view * np.sin(ang)).sum()])

    def estimate(self, obs: Observation) -> Optional[HomingEstimate]:
        diff = self.alv(obs.home_snapshot) - self.alv(obs.panorama)
        rms = float(np.sqrt(np.mean((obs.panorama - obs.home_snapshot) ** 2)))
        return HomingEstimate(float(np.arctan2(diff[1], diff[0])), None, rms < self.arrive_rms)
