"""DAgger para la mosca (v10). Requiere PyTorch y los checkpoints de la v9.

Uso:
    python scripts/train_dagger.py                    # semillas de configs/dagger.yaml
    python scripts/train_dagger.py --seeds 0          # una semilla (≈ 15 min en T4, estimado)
    python scripts/plot_dagger.py                     # figura y tabla
Genera:
    checkpoints/<modelo>_dagger<k>_s<semilla>.pt
    results/v10_dagger_<modelo>_s<semilla>.json      (por iteración: datos, F1, éxito en lazo cerrado)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroflybee.config import load_config  # noqa: E402
from neuroflybee.data.generate import FlyDataset  # noqa: E402
from neuroflybee.training.config import load_train_config  # noqa: E402
from neuroflybee.training.dagger import run_dagger  # noqa: E402
from neuroflybee.training.dagger_config import load_dagger_config  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--dagger-config", default=str(ROOT / "configs" / "dagger.yaml"))
    parser.add_argument("--train-config", default=str(ROOT / "configs" / "train_fly.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "generated"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    cfg, dcfg = load_config(args.config), load_dagger_config(args.dagger_config)
    tcfg = load_train_config(args.train_config)
    data = Path(args.data)
    train, val = FlyDataset.load(data / "fly_train.npz"), FlyDataset.load(data / "fly_val.npz")
    for seed in (args.seeds if args.seeds is not None else dcfg.seeds):
        report = run_dagger(cfg, dcfg, tcfg, seed, train, val, Path(args.checkpoints), Path(args.results))
        first, last = report["iterations"][0], report["iterations"][-1]
        print(f"-> s{seed}: éxito {first['closed_loop']['all']['success']:.2f} → "
              f"{last['closed_loop']['all']['success']:.2f}  ({report['total_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
