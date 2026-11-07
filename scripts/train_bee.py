"""Entrena la red de la abeja (v11). Requiere PyTorch.

Uso:
    python scripts/generate_data.py      # una vez
    python scripts/train_bee.py          # 3 semillas (configs/train_bee.yaml)
    python scripts/show_bee_homing.py    # figura comparando red, ALV y dirección real
Genera:
    checkpoints/bee_s<semilla>.pt   (mejor época según el error angular mediano de validación)
    results/v11_historial_bee_s<semilla>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroflybee.data.generate import BeeDataset  # noqa: E402
from neuroflybee.models.bee_spec import load_bee_config  # noqa: E402
from neuroflybee.training.bee import train_bee_model  # noqa: E402
from neuroflybee.training.config import load_train_config  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--data", default=str(ROOT / "data" / "generated"))
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "bee_model.yaml"))
    parser.add_argument("--train-config", default=str(ROOT / "configs" / "train_bee.yaml"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    net_cfg, tcfg = load_bee_config(args.model_config), load_train_config(args.train_config)
    data = Path(args.data)
    train, val = BeeDataset.load(data / "bee_train.npz"), BeeDataset.load(data / "bee_val.npz")
    print(f"datos: {len(train):,} entrenamiento / {len(val):,} validación")
    for seed in (args.seeds if args.seeds is not None else tcfg.seeds):
        r = train_bee_model(train, val, net_cfg, tcfg, seed, Path(args.checkpoints), Path(args.results))
        print(f"-> s{seed}: error angular mediano {r.best_score:.1f}° (época {r.best_epoch}, "
              f"{r.total_seconds / 60:.1f} min, {r.device})")


if __name__ == "__main__":
    main()
