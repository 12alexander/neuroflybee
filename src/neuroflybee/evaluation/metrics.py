"""Métricas de clasificación sin dependencias de PyTorch (NumPy puro)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Filas = clase real, columnas = clase predicha."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return cm


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    denom = precision + recall
    return np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    f1_macro: float
    f1_per_class: tuple[float, ...]
    support: tuple[int, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> ClassificationReport:
    """F1 macro sólo sobre las clases presentes en `y_true` (las ausentes no se pueden evaluar)."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    support = cm.sum(axis=1)
    f1 = per_class_f1(cm)
    present = support > 0
    return ClassificationReport(
        accuracy=float(np.trace(cm) / max(cm.sum(), 1)),
        f1_macro=float(f1[present].mean()) if present.any() else 0.0,
        f1_per_class=tuple(float(v) for v in f1),
        support=tuple(int(v) for v in support),
    )


def inverse_frequency_weights(counts: np.ndarray, power: float = 0.5, max_weight: float = 10.0) -> np.ndarray:
    """Pesos por clase para la entropía cruzada: (frecuencia)^(-power), normalizados a
    media 1 y recortados para que una clase rarísima no domine el entrenamiento."""
    counts = np.asarray(counts, dtype=float)
    freq = counts / max(counts.sum(), 1.0)
    w = np.where(freq > 0, np.power(np.maximum(freq, 1e-12), -power), 0.0)
    w = w / w[w > 0].mean()
    return np.minimum(w, max_weight)


def sequence_windows(episode: np.ndarray, seq_len: int) -> np.ndarray:
    """Índices (N, seq_len) del cuadro actual y los anteriores del MISMO episodio.
    Al inicio de un episodio se repite el primer cuadro (relleno), nunca se mezclan
    episodios. Supone que las muestras de cada episodio están contiguas y en orden."""
    episode = np.asarray(episode)
    n = len(episode)
    idx = np.arange(n)
    starts = np.r_[0, np.flatnonzero(np.diff(episode)) + 1]
    first = np.repeat(starts, np.diff(np.r_[starts, n]))
    offsets = np.arange(seq_len - 1, -1, -1)          # [L-1, …, 0]
    return np.maximum(idx[:, None] - offsets[None, :], first[:, None])


def angular_errors_deg(pred_angle: np.ndarray, true_angle: np.ndarray) -> np.ndarray:
    """Error angular absoluto en grados, en [0, 180]."""
    diff = (np.asarray(pred_angle) - np.asarray(true_angle) + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(np.abs(diff))


def homing_report(pred: np.ndarray, target: np.ndarray) -> dict:
    """Métricas del retorno visual. `pred` y `target` son (N, 3) = (cos, sin, log(1+d))."""
    err = angular_errors_deg(np.arctan2(pred[:, 1], pred[:, 0]), np.arctan2(target[:, 1], target[:, 0]))
    d_pred, d_true = np.expm1(np.maximum(pred[:, 2], 0)), np.expm1(target[:, 2])
    return {"angle_error_median": float(np.median(err)), "angle_error_mean": float(err.mean()),
            "within_30deg": float((err <= 30).mean()), "distance_mae": float(np.abs(d_pred - d_true).mean())}


# ----------------------------------------------------------------------------- AUROC (v22)
def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rangos 1..N con los empates promediados (necesario para AUROC exacta con empates)."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    for end in np.r_[np.flatnonzero(np.diff(sorted_vals)) + 1, len(values)]:
        ranks[order[start:end]] = (start + end + 1) / 2.0      # promedio de start+1 … end
        start = end
    return ranks


def roc_auc_binary(positive: np.ndarray, scores: np.ndarray) -> float:
    """AUROC de un problema binario vía la U de Mann–Whitney; NaN si falta una de las clases.

    Es la probabilidad de que un positivo al azar reciba un puntaje mayor que un negativo al azar."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(scores, dtype=float))
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_auc_ovr_macro(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    """AUROC macro uno-contra-el-resto (Fawcett, 2006): promedio sobre las clases evaluables.

    Complementa al F1 macro: mide si la red ORDENA bien las acciones (independiente del umbral
    argmax), algo útil con clases desbalanceadas como «mantener posición»."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    aucs = [roc_auc_binary(y_true == c, probabilities[:, c]) for c in range(n_classes)]
    aucs = [a for a in aucs if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")
