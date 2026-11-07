"""Utilidades geométricas 2D (vectorizadas con NumPy)."""
from __future__ import annotations

import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Lleva un ángulo (o arreglo de ángulos) al intervalo [-pi, pi)."""
    wrapped = (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi
    return wrapped if isinstance(angle, np.ndarray) else float(wrapped)


def unit(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)])


def rotate(points: np.ndarray, angle: float) -> np.ndarray:
    """Rota puntos (..., 2) un ángulo en radianes (sentido antihorario)."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T


def bearing(src: np.ndarray, dst: np.ndarray) -> float:
    d = np.asarray(dst, dtype=float) - np.asarray(src, dtype=float)
    return float(np.arctan2(d[1], d[0]))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def to_body_frame(vector_world: np.ndarray, heading: float) -> np.ndarray:
    """Expresa un vector del mundo en el marco del dron (x = adelante, y = izquierda)."""
    return rotate(np.asarray(vector_world, dtype=float), -heading)
