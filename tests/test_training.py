"""Pruebas del entrenamiento: configuración y gráficos sin PyTorch; bucle real con PyTorch."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from neuroflybee.evaluation.reports import aggregate, best_rows, load_histories
from neuroflybee.training.config import TrainConfig, load_train_config

ROOT = Path(__file__).resolve().parents[1]


def test_train_yaml_loads():
    cfg = load_train_config(ROOT / "configs" / "train_fly.yaml")
    assert isinstance(cfg, TrainConfig) and cfg.seeds == (0, 1, 2) and cfg.patience < cfg.epochs


def _fake_history(model, seed, f1s):
    return {"model": model, "seed": seed, "parameters": 1000, "best_epoch": int(np.argmax(f1s)) + 1,
            "best_score": max(f1s), "total_seconds": 10.0, "device": "cpu",
            "confusion_val": np.eye(6, dtype=int).tolist(),
            "history": [{"epoch": i + 1, "train_loss": 1 / (i + 1), "val_loss": 1.2 / (i + 1),
                         "metrics": {"f1_macro": f, "accuracy": f}, "lr": 1e-3, "seconds": 1.0}
                        for i, f in enumerate(f1s)]}


def test_reports_and_plots_from_histories(tmp_path):
    """Los gráficos y la tabla se construyen a partir de historiales (aquí sintéticos,
    sólo para probar el código; los reales salen de scripts/train_fly.py)."""
    for model in ("fly_cnn", "fly_cnn_gru"):
        for seed, f1s in ((0, [0.5, 0.7, 0.6]), (1, [0.55, 0.65, 0.8])):
            (tmp_path / f"v9_historial_{model}_s{seed}.json").write_text(json.dumps(_fake_history(model, seed, f1s)))
    summary = aggregate(best_rows(load_histories(tmp_path)))
    assert summary["fly_cnn"]["f1_mean"] == pytest.approx(0.75) and summary["fly_cnn"]["n_seeds"] == 2
    spec = importlib.util.spec_from_file_location("p", ROOT / "scripts" / "plot_fly_training.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--results", str(tmp_path), "--data", str(tmp_path / "sin_datos")])
    for name in ("v9_curvas_mosca.png", "v9_confusion_mosca.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 10_000
    assert (tmp_path / "v9_mosca_validacion.csv").exists()


def test_training_loop_with_torch(tmp_path):
    torch = pytest.importorskip("torch")
    from neuroflybee.data.generate import FlyDataset
    from neuroflybee.models.spec import FlyNetConfig
    from neuroflybee.training.checkpoints import load_fly_checkpoint
    from neuroflybee.training.fly import train_fly_model

    rng = np.random.default_rng(0)
    n = 300
    actions = rng.integers(0, 6, n)
    patches = (rng.random((n, 32, 32)) < 0.1).astype(np.uint8)
    patches[np.arange(n), 0, actions] = 1                 # señal aprendible: la acción está "dibujada"
    ds = FlyDataset(patches, rng.normal(size=(n, 3)).astype(np.float32), actions.astype(np.int64),
                    np.repeat(np.arange(10), 30).astype(np.int32), np.tile(np.arange(30), 10).astype(np.int32),
                    np.zeros(n, dtype=np.int8))
    tcfg = TrainConfig(epochs=3, batch_size=64, num_workers=0, patience=3, device="cpu")
    result = train_fly_model("fly_cnn_gru", ds, ds, FlyNetConfig(), tcfg, 0, tmp_path / "ck", tmp_path)
    assert len(result.history) == 3 and result.history[-1].train_loss < result.history[0].train_loss
    model, payload = load_fly_checkpoint(tmp_path / "ck" / "fly_cnn_gru_s0.pt")
    assert payload["model_name"] == "fly_cnn_gru"
    out = model(torch.zeros(2, 4, 32, 32), torch.zeros(2, 3))
    assert out.shape == (2, 6)
    assert (tmp_path / "v9_historial_fly_cnn_gru_s0.json").exists()
