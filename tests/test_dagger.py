"""DAgger: partes sin PyTorch (configuración, recolección con estudiante, evaluación en
lazo cerrado, figuras) y una corrida mínima real con PyTorch."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from neuroflybee.config import ProjectConfig
from neuroflybee.control import classical  # noqa: F401
from neuroflybee.control import registry
from neuroflybee.data.generate import collect_fly_episodes
from neuroflybee.domain.drone import Action
from neuroflybee.evaluation.closed_loop import evaluate, summarize
from neuroflybee.training.dagger_config import DaggerConfig, load_dagger_config

ROOT = Path(__file__).resolve().parents[1]


def test_dagger_config():
    d = load_dagger_config(ROOT / "configs" / "dagger.yaml")
    assert d.beta(1) == 0.5 and d.beta(3) == 0.0 and d.beta(9) == 0.0
    assert d.new_seed_offset(1) == 400 and d.new_seed_offset(2) == 400 + d.episodes_per_iteration
    assert d.new_seed_offset(d.iterations) + d.episodes_per_iteration <= 10_000   # sigue en entrenamiento


class CountingStudent:
    def __init__(self):
        self.calls = 0

    def __call__(self, obs, goal):
        self.calls += 1
        return Action.HOVER


def test_student_is_queried_every_step_and_labels_stay_expert():
    student = CountingStudent()
    ds = collect_fly_episodes(ProjectConfig(), "train", 1, behavior=student, beta=0.0, max_steps=30)
    assert student.calls == len(ds)                       # memoria completa del estudiante
    assert (ds.actions != Action.HOVER).any()             # las etiquetas son del experto, no del estudiante


def test_closed_loop_evaluation_with_a_classical_controller():
    res = evaluate(ProjectConfig(), lambda: registry.create("astar"), ["nominal"], [10_000, 10_001])
    s = summarize(res)
    assert s["episodes"] == 2 and 0 <= s["success"] <= 1 and res[0].mean_latency_ms >= 0


def _fake_report(seed, successes):
    it = lambda i, s: {"iteration": i, "beta": 1.0 if i == 0 else 0.5, "samples": 1000 * (i + 1),  # noqa: E731
                       "val_f1_macro": 0.6 + 0.05 * i,
                       "closed_loop": {"all": {"success": s, "collision": 1 - s},
                                       "by_family": {"nominal": {"success": s}, "dynamic": {"success": s / 2}}}}
    return {"model": "fly_cnn_gru", "seed": seed, "iterations": [it(i, s) for i, s in enumerate(successes)]}


def test_dagger_plots(tmp_path):
    for seed in (0, 1):
        (tmp_path / f"v10_dagger_fly_cnn_gru_s{seed}.json").write_text(json.dumps(_fake_report(seed, [0.3, 0.5, 0.6])))
    spec = importlib.util.spec_from_file_location("p", ROOT / "scripts" / "plot_dagger.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--results", str(tmp_path)])
    assert (tmp_path / "figures" / "v10_dagger_mejora.png").stat().st_size > 10_000
    assert (tmp_path / "figures" / "v10_dagger_esquema.png").exists()
    assert len((tmp_path / "v10_dagger.csv").read_text().splitlines()) == 4


def test_dagger_minimal_run_with_torch(tmp_path):
    pytest.importorskip("torch")
    from neuroflybee.models.fly import build_fly_model
    from neuroflybee.models.spec import FlyNetConfig
    from neuroflybee.training.checkpoints import save_fly_checkpoint
    from neuroflybee.training.config import TrainConfig
    from neuroflybee.training.dagger import run_dagger

    cfg = ProjectConfig()
    base = collect_fly_episodes(cfg, "train", 2, max_steps=40)
    val = collect_fly_episodes(cfg, "val", 1, max_steps=40)
    model = build_fly_model("fly_cnn_gru", FlyNetConfig())
    save_fly_checkpoint(tmp_path / "ck" / "fly_cnn_gru_s0.pt", "fly_cnn_gru", model, 0, {}, 0)
    dcfg = DaggerConfig(iterations=1, episodes_per_iteration=1, epochs_per_iteration=1,
                        eval_families=("nominal",), eval_episodes_per_family=1)
    report = run_dagger(cfg, dcfg, TrainConfig(num_workers=0, device="cpu", batch_size=32), 0, base, val,
                        tmp_path / "ck", tmp_path)
    assert [it["iteration"] for it in report["iterations"]] == [0, 1]
    assert (tmp_path / "ck" / "fly_cnn_gru_dagger1_s0.pt").exists()
