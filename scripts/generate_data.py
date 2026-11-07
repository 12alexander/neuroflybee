"""Genera los conjuntos de datos de la mosca y la abeja (v7).

Uso:
    python scripts/generate_data.py                       # tamaños de configs/data.yaml
    python scripts/generate_data.py --scale 0.1           # versión reducida (prueba rápida)
Genera (no se suben al repositorio):
    data/generated/fly_train.npz, fly_val.npz, bee_train.npz, bee_val.npz, bee_test.npz (v20)
y el resumen versionado:
    results/v7_datasets.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from neuroflybee.config import from_dict, load_config  # noqa: E402
from neuroflybee.data.generate import (DataConfig, collect_bee_samples,  # noqa: E402
                                       collect_fly_episodes)
from neuroflybee.domain.drone import ACTION_LABELS_ES, Action  # noqa: E402
from neuroflybee.utils.seeding import set_global_seed  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data-config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--scale", type=float, default=1.0, help="fracción de los tamaños configurados")
    parser.add_argument("--out", default=None, help="carpeta de salida (por defecto la de data.yaml)")
    parser.add_argument("--summary", default=str(ROOT / "results" / "v7_datasets.csv"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    with open(args.data_config, encoding="utf-8") as fh:
        dcfg = from_dict(DataConfig, yaml.safe_load(fh))
    set_global_seed(dcfg.seed)
    out = Path(args.out) if args.out else ROOT / dcfg.out_dir
    n = lambda x: max(1, int(round(x * args.scale)))  # noqa: E731

    rows = []
    for split, episodes in (("train", dcfg.fly.train_episodes), ("val", dcfg.fly.val_episodes)):
        t0 = time.time()
        ds = collect_fly_episodes(cfg, split, n(episodes), noise_eps=dcfg.fly.noise_eps,
                                  seed=dcfg.seed, max_steps=dcfg.fly.max_steps)
        ds.save(out / f"fly_{split}.npz")
        counts = ds.action_counts()
        rows.append(["mosca", split, n(episodes), len(ds), *[int(c) for c in counts], f"{time.time() - t0:.0f}"])
        print(f"mosca/{split}: {n(episodes)} episodios, {len(ds)} muestras ({time.time() - t0:.0f} s)")
    for split in ("train", "val", "test"):
        scenarios = dcfg.bee.scenarios_for(split)
        t0 = time.time()
        ds = collect_bee_samples(cfg, split, n(scenarios), dcfg.bee.per_scenario, dcfg.bee.max_radius,
                                 seed=dcfg.seed)
        ds.save(out / f"bee_{split}.npz")
        rows.append(["abeja", split, n(scenarios), len(ds), *[""] * len(Action), f"{time.time() - t0:.0f}"])
        print(f"abeja/{split}: {n(scenarios)} escenarios, {len(ds)} muestras ({time.time() - t0:.0f} s)")

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modulo", "particion", "episodios_o_escenarios", "muestras",
                    *[ACTION_LABELS_ES[a] for a in Action], "segundos"])
        w.writerows(rows)
    print(f"Resumen en {args.summary}")


if __name__ == "__main__":
    main()
