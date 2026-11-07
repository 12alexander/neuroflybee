import numpy as np
import pytest

from neuroflybee.evaluation.metrics import (classification_report, confusion_matrix,
                                            inverse_frequency_weights, sequence_windows)


def test_confusion_and_report():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(y_true, y_pred, 4)
    assert cm.tolist()[:3] == [[1, 1, 0, 0], [0, 2, 0, 0], [1, 0, 1, 0]]
    rep = classification_report(y_true, y_pred, 4)
    assert rep.accuracy == pytest.approx(4 / 6)
    # F1: clase0 = 0.5, clase1 = 0.8, clase2 ≈ 0.667; la clase 3 no aparece y no cuenta
    assert rep.f1_macro == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert rep.support == (2, 2, 2, 0)


def test_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4, 5])
    assert classification_report(y, y, 6).f1_macro == 1.0


def test_inverse_frequency_weights():
    w = inverse_frequency_weights(np.array([900, 90, 9, 1]), power=0.5, max_weight=10)
    assert w[0] < w[1] < w[2] and w.max() <= 10
    assert inverse_frequency_weights(np.array([5, 0]))[1] == 0


def test_sequence_windows_never_cross_episodes():
    episode = np.array([0, 0, 0, 1, 1, 2])
    win = sequence_windows(episode, 3)
    assert win.tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 2], [3, 3, 3], [3, 3, 4], [5, 5, 5]]
    assert all(len(set(episode[w])) == 1 for w in win)
