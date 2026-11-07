"""Entrena las redes de la mosca por imitación del experto (v9). Requiere PyTorch.

Uso:
    python scripts/generate_data.py                                  # una vez
    python scripts/train_fly.py                                      # 4 modelos × 3 semillas
    python scripts/train_fly.py --models fly_cnn_gru --seeds 0       # prueba rápida
    python scripts/plot_fly_training.py                              # figuras y tabla
Genera:
    checkpoints/<modelo>_s<semilla>.pt               (mejor época según F1 macro de validación)
    results/v9_historial_<modelo>_s<semilla>.json   (curvas, pesos de clase, matriz de confusión)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroflybee.data.generate import FlyDataset  # noqa: E402
from neuroflybee.models.spec import FLY_MODEL_NAMES, load_fly_config  # noqa: E402
from neuroflybee.training.config import load_train_config  # noqa: E402
from neuroflybee.training.fly import train_fly_model  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(FLY_MODEL_NAMES))
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="por defecto las de train_fly.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data", default=str(ROOT / "data" / "generated"))
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "fly_models.yaml"))
    parser.add_argument("--train-config", default=str(ROOT / "configs" / "train_fly.yaml"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    net_cfg = load_fly_config(args.model_config)
    tcfg = load_train_config(args.train_config)
    if args.epochs is not None:
        from dataclasses import replace
        tcfg = replace(tcfg, epochs=args.epochs)
    data = Path(args.data)
    if not (data / "fly_train.npz").exists():
        raise SystemExit("Faltan los datos: ejecute primero  python scripts/generate_data.py")
    train, val = FlyDataset.load(data / "fly_train.npz"), FlyDataset.load(data / "fly_val.npz")
    print(f"datos: {len(train):,} entrenamiento / {len(val):,} validación")
    for model_name in args.models.split(","):
        for seed in (args.seeds if args.seeds is not None else tcfg.seeds):
            result = train_fly_model(model_name, train, val, net_cfg, tcfg, seed,
                                     Path(args.checkpoints), Path(args.results))
            print(f"-> {model_name} s{seed}: mejor F1 macro {result.best_score:.4f} "
                  f"en la época {result.best_epoch} ({result.total_seconds / 60:.1f} min, {result.device})")


if __name__ == "__main__":
    main()
