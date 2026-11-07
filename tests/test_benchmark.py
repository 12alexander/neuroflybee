import importlib.util
from pathlib import Path

import pytest

from neuroflybee.evaluation.benchmark import (Cell, Record, Variant, latex_table, load_benchmark_spec, table,
                                              wilson_interval)
from neuroflybee.evaluation.closed_loop import EpisodeResult

ROOT = Path(__file__).resolve().parents[1]


def test_spec_loads_with_ablations():
    spec = load_benchmark_spec(ROOT / "configs" / "benchmark.yaml")
    assert len(spec.families) == 6 and spec.episodes_per_family == 30 and len(spec.ablations) == 4
    arch = spec.ablations[0]
    assert [v.fly_model for v in arch.variants] == ["fly_mlp", "fly_cnn", "fly_cnn_gru", "fly_cnn_transformer"]
    assert all(v.dagger is False for v in arch.variants)


def test_wilson_interval():
    lo, hi = wilson_interval(15, 30)
    assert lo < 0.5 < hi and 0.3 < lo and hi < 0.7
    assert wilson_interval(0, 30)[0] == 0.0 and wilson_interval(30, 30)[1] == pytest.approx(1.0)


def _rec(label, seed, family, success):
    return Record(label, "x", seed, EpisodeResult("x", family, 0, success, not success, success, 1.0, 1.0, 10,
                                                  5.0, 0.1))


def test_table_mean_and_std_across_training_seeds():
    recs = ([_rec("Red", 0, "nominal", s) for s in (1, 1, 0, 0)]       # semilla 0: 50 %
            + [_rec("Red", 1, "nominal", s) for s in (1, 1, 1, 0)])     # semilla 1: 75 %
    cell: Cell = table(recs, ["nominal"])["Red"]["total"]
    assert cell.success_mean == pytest.approx(0.625) and cell.success_std == pytest.approx(0.125)
    assert cell.n_seeds == 2 and cell.n_episodes == 8
    tex = latex_table(table(recs, ["nominal"]), ["nominal"], "c", "tab:x")
    assert r"\textbf{62\% ± 12\%}" in tex and "Nominal" in tex


def test_run_benchmark_quick(tmp_path):
    spec = importlib.util.spec_from_file_location("rb", ROOT / "scripts" / "run_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--episodes", "1", "--results", str(tmp_path), "--paper", str(tmp_path / "tablas"),
              "--checkpoints", str(tmp_path / "sin_pesos")])
    assert (tmp_path / "v14_tabla_principal.csv").read_text(encoding="utf-8").count("\n") == 1 + 5
    assert (tmp_path / "tablas" / "v14_tabla_principal.tex").exists()
    assert (tmp_path / "figures" / "v14_benchmark.png").stat().st_size > 10_000
    assert (tmp_path / "v14_ablacion_3.csv").exists()          # la ablación de retorno tiene 2 variantes clásicas
