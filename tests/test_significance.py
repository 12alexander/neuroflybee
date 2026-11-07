"""v22: significancia estadística pareada y AUROC macro de la mosca."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from neuroflybee.evaluation.metrics import roc_auc_binary, roc_auc_ovr_macro
from neuroflybee.evaluation.significance import (Comparison, SignificanceConfig, bootstrap_ci, cohen_h, holm,
                                                 latex_table, load_significance_config, paired_differences,
                                                 run_significance, scenario_outcomes, sign_flip_pvalue, write_csv)
from neuroflybee.reporting.numbers import NumberSet, best_epoch_metric, significance

ROOT = Path(__file__).resolve().parents[1]


def _episodes(label: str, successes: dict, group: str = "principal", seeds=("",)) -> list[dict]:
    """Filas con el formato de v14_episodios.csv; `successes[(cond, semilla)]` = éxito por semilla."""
    rows = []
    for (cond, scen), values in successes.items():
        values = values if isinstance(values, (list, tuple)) else [values] * len(seeds)
        for seed, ok in zip(seeds, values):
            rows.append({"grupo": group, "etiqueta": label, "semilla_entrenamiento": seed, "condicion": cond,
                         "semilla_escenario": str(scen), "exito": str(int(ok))})
    return rows


# ------------------------------------------------------------------ estadística
def test_holm_and_effect_size():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert cohen_h(0.5, 0.5) == 0 and cohen_h(1.0, 0.0) == pytest.approx(math.pi)


def test_exact_and_monte_carlo_pvalues():
    rng = np.random.default_rng(0)
    eight_wins = np.array([1.0] * 8 + [0.0] * 5)
    assert sign_flip_pvalue(eight_wins, 1000, rng) == pytest.approx(2 / 256)       # McNemar exacta
    assert sign_flip_pvalue(np.zeros(10), 1000, rng) == 1.0
    assert sign_flip_pvalue(np.array([1.0, -1.0] * 5), 1000, rng) == 1.0
    thirds = np.array([1 / 3] * 12)                                                 # 3 semillas: no binario
    assert sign_flip_pvalue(thirds, 5000, rng) < 0.01


def test_bootstrap_interval():
    rng = np.random.default_rng(0)
    lo, hi = bootstrap_ci(np.array([0.5] * 20), 500, rng)
    assert lo == hi == pytest.approx(0.5)
    d = np.r_[np.ones(30), np.zeros(30)]
    lo, hi = bootstrap_ci(d, 2000, rng)
    assert lo < 0.5 < hi
    assert all(np.isnan(bootstrap_ci(np.array([]), 10, rng)))


# ------------------------------------------------------------------ pareo por escenario
def test_outcomes_average_training_seeds_and_pair_by_scenario():
    rows = _episodes("Red", {("gps_loss", 1): [1, 0, 1], ("gps_loss", 2): [0, 0, 0], ("dynamic", 1): [1, 1, 1]},
                     seeds=("0", "1", "2"))
    out = scenario_outcomes(rows, "principal", "Red")
    assert out[("gps_loss", 1)] == pytest.approx(2 / 3) and out[("dynamic", 1)] == 1
    other = {("gps_loss", 1): 0.0, ("gps_loss", 3): 1.0}
    a, b = paired_differences(out, other, "gps_loss")
    assert list(a) == pytest.approx([2 / 3]) and list(b) == [0.0]                  # sólo escenarios comunes
    a, b = paired_differences(out, other, "total")
    assert len(a) == 1


def test_run_significance_tables_and_numbers(tmp_path):
    scen = {("gps_loss", s): 1 for s in range(30)}
    rows = _episodes("Híbrido abeja+mosca", scen)
    rows += _episodes("A* + ALV (clásico)", {k: int(k[1] % 3 == 0) for k in scen})
    rows += _episodes("A* (clásico)", {k: 1 for k in scen})
    cfg = SignificanceConfig(n_bootstrap=500, n_permutations=500, comparisons=(
        Comparison("principal", "Híbrido abeja+mosca", "A* + ALV (clásico)", ("total", "gps_loss"), "HibridoAlv"),
        Comparison("principal", "Híbrido abeja+mosca", "A* (clásico)", ("total",)),
        Comparison("principal", "Híbrido abeja+mosca", "Inexistente", ("total",)),
    ))
    report = run_significance(rows, cfg)
    assert len(report.results) == 3 and len(report.skipped) == 1
    first = report.results[0]
    assert first.diff == pytest.approx(2 / 3) and first.significant and first.p_holm >= first.p_value
    assert not report.results[2].significant and report.results[2].p_value == 1.0  # A = B

    write_csv(tmp_path / "v22_significancia.csv", report)
    tex = latex_table(report)
    assert "\\label{tab:significancia}" in tex and "\\textbf{" in tex and "A$^*$" in tex
    assert tex.count("{") == tex.count("}")                                        # LaTeX balanceado
    ns = NumberSet()
    significance(ns, tmp_path)
    assert ns.values["SigSignificativas"][0] == "2" and ns.values["SigComparaciones"][0] == "3"
    assert ns.values["SigHibridoAlvSinGPSDif"][0].startswith("+67")
    assert "Holm" in ns.values["SigHibridoAlvTotalP"][0]
    ns = NumberSet()
    significance(ns, tmp_path / "vacio")
    assert ns.is_pending("SigSignificativas")


def test_repository_config_and_script(tmp_path):
    cfg = load_significance_config(ROOT / "configs" / "significance.yaml")
    assert cfg.alpha == 0.05 and len(cfg.comparisons) >= 10
    assert all(c.conditions for c in cfg.comparisons)
    spec = importlib.util.spec_from_file_location("sig", ROOT / "scripts" / "significance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--episodes", str(tmp_path / "no_existe.csv"), "--results", str(tmp_path), "--paper", str(tmp_path)])
    assert "pendiente" in (tmp_path / "tablas" / "v22_significancia.tex").read_text(encoding="utf-8")


# ------------------------------------------------------------------ AUROC
def test_auroc_binary_cases():
    y = np.array([0, 0, 1, 1])
    assert roc_auc_binary(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc_binary(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc_binary(y, np.full(4, 0.5)) == 0.5                                # empates
    assert np.isnan(roc_auc_binary(np.ones(3), np.arange(3)))


def test_auroc_macro_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y = rng.integers(0, 6, 400)
    probs = rng.dirichlet(np.ones(6), 400)
    probs[np.arange(400), y] += 0.3                                                 # algo de señal
    probs = np.round(probs / probs.sum(1, keepdims=True), 2)                        # fuerza empates
    for c in range(6):                                                              # binario, con empates
        assert roc_auc_binary(y == c, probs[:, c]) == pytest.approx(
            sklearn_metrics.roc_auc_score(y == c, probs[:, c]), abs=1e-12)
    exact = probs / probs.sum(1, keepdims=True)                                     # sklearn exige suma 1
    assert roc_auc_ovr_macro(y, exact, 6) == pytest.approx(
        sklearn_metrics.roc_auc_score(y, exact, multi_class="ovr", average="macro"), abs=1e-12)


def test_auroc_ignores_absent_classes_and_reads_best_epoch():
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1, 0], [0.8, 0.2, 0], [0.3, 0.7, 0], [0.1, 0.9, 0]])
    assert roc_auc_ovr_macro(y, probs, 3) == 1.0                                    # la clase 2 no aparece
    history = {"best_epoch": 2, "history": [{"epoch": 1, "metrics": {"auroc_macro": 0.7}},
                                            {"epoch": 2, "metrics": {"auroc_macro": 0.9}}]}
    assert best_epoch_metric(history, "auroc_macro") == 0.9
    assert best_epoch_metric(history, "otra") is None
