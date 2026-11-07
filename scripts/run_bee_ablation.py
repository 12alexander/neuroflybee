"""Ablación de la red de la abeja (v20), evaluada en la partición de PRUEBA.

Variantes (configs/bee_ablation.yaml): siamesa de v11, sólo vista actual, siamesa sin la
resta explícita y siamesa con lectura vectorial compacta; el ALV clásico como referencia.
Cada variante se entrena con las mismas semillas, datos e hiperparámetros que la red principal.
La variante «completa» reutiliza los pesos de v11 (checkpoints/bee_s<semilla>.pt).

Uso:
    python scripts/run_bee_ablation.py                 # 3 semillas (configs/train_bee.yaml)
    python scripts/run_bee_ablation.py --seeds 0       # rápido
    python scripts/run_bee_ablation.py --sin-redes     # sólo ALV y tamaños (sin PyTorch)
Genera:
    checkpoints/bee_ablacion_<variante>_s<semilla>.pt
    results/v20_ablacion_abeja.csv            (una fila por variante y semilla)
    results/v20_ablacion_abeja_resumen.csv    (media ± d.e. entre semillas)
    results/figures/v20_ablacion_abeja.png
    paper/tablas/v20_ablacion_abeja.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from neuroflybee.config import from_dict, load_config  # noqa: E402
from neuroflybee.data.generate import BeeDataConfig, BeeDataset, load_or_collect_bee_split  # noqa: E402
from neuroflybee.evaluation.bee_ablation import (AblationVariant, VariantScore, latex_table,  # noqa: E402
                                                 load_bee_ablation_config, score_alv, score_predictions,
                                                 summarize, write_scores_csv, write_summary_csv)
from neuroflybee.models.bee_spec import count_bee_parameters, load_bee_config  # noqa: E402
from neuroflybee.training.config import load_train_config  # noqa: E402
from neuroflybee.viz.bee_ablation_plot import plot_bee_ablation  # noqa: E402


class VariantRunner:
    """Entrena (si hace falta) y evalúa una variante; aísla todo lo que necesita PyTorch."""

    def __init__(self, base_cfg, tcfg, data_dir: Path, checkpoints: Path, results: Path, retrain: bool):
        self.base_cfg, self.tcfg = base_cfg, tcfg
        self.data_dir, self.checkpoints, self.results, self.retrain = data_dir, checkpoints, results, retrain
        self._train_val: tuple[BeeDataset, BeeDataset] | None = None

    def _datasets(self) -> tuple[BeeDataset, BeeDataset]:
        """Entrenamiento y validación, cargados sólo si alguna variante debe entrenarse."""
        if self._train_val is None:
            self._train_val = (BeeDataset.load(self.data_dir / "bee_train.npz"),
                               BeeDataset.load(self.data_dir / "bee_val.npz"))
        return self._train_val

    def score(self, variant: AblationVariant, seed: int, test: BeeDataset) -> VariantScore:
        from neuroflybee.training.bee import load_bee_checkpoint, predict_bee, train_bee_model
        from neuroflybee.training.trainer import resolve_device

        net_cfg = variant.config(self.base_cfg)
        ckpt = self.checkpoints / variant.checkpoint_name(seed)
        if self.retrain or not ckpt.exists():
            if variant.reuse_v11:
                print(f"   (no hay {ckpt.name}: se entrena la red principal con la semilla {seed})")
            train, val = self._datasets()
            train_bee_model(train, val, net_cfg, self.tcfg, seed, self.checkpoints, self.results,
                            run_name=variant.run_name(seed), history_tag="v11" if variant.reuse_v11 else "v20")
        device = resolve_device(self.tcfg.device)
        model, _ = load_bee_checkpoint(ckpt, device)
        if model.cfg != net_cfg:
            raise ValueError(f"{ckpt.name} no corresponde a la variante '{variant.name}': {model.cfg}")
        pred = predict_bee(model, test, device=device)
        return score_predictions(variant, seed, count_bee_parameters(net_cfg), pred, test.target)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data-config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--ablation-config", default=str(ROOT / "configs" / "bee_ablation.yaml"))
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "bee_model.yaml"))
    parser.add_argument("--train-config", default=str(ROOT / "configs" / "train_bee.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "generated"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--paper", default=str(ROOT / "paper"))
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=None, help="subconjunto de variantes")
    parser.add_argument("--retrain", action="store_true", help="reentrena aunque existan los pesos")
    parser.add_argument("--sin-redes", action="store_true", help="sólo ALV (no necesita PyTorch)")
    parser.add_argument("--scale", type=float, default=1.0, help="fracción del conjunto de prueba")
    args = parser.parse_args(argv)

    cfg, base_cfg = load_config(args.config), load_bee_config(args.model_config)
    abl = load_bee_ablation_config(args.ablation_config, base_cfg)
    with open(args.data_config, encoding="utf-8") as fh:
        raw = (yaml.safe_load(fh) or {}).get("bee", {})
    bee_data = from_dict(BeeDataConfig, {**raw, "test_scenarios": abl.test_scenarios,
                                         "per_scenario": abl.per_scenario})
    data_dir, results = Path(args.data), Path(args.results)
    test = load_or_collect_bee_split(cfg, bee_data, "test", data_dir, scale=args.scale)
    print(f"prueba: {len(test):,} vistas")

    scores = [score_alv(test.current, test.home, test.target)]
    variants = [abl.variant(n) for n in args.variants] if args.variants else list(abl.variants)
    if not args.sin_redes:
        tcfg = load_train_config(args.train_config)
        runner = VariantRunner(base_cfg, tcfg, data_dir, Path(args.checkpoints), results, args.retrain)
        for variant in variants:
            for seed in (args.seeds if args.seeds is not None else tcfg.seeds):
                s = runner.score(variant, seed, test)
                scores.append(s)
                print(f"-> {variant.name:15s} s{seed}: error mediano {s.angle_error_median:5.1f}° | "
                      f"≤30°: {s.within_30deg:.0%} | {s.parameters:,} parámetros")

    summary = summarize(scores)
    write_scores_csv(results / "v20_ablacion_abeja.csv", scores)
    write_summary_csv(results / "v20_ablacion_abeja_resumen.csv", summary)
    plot_bee_ablation(summary, results / "figures" / "v20_ablacion_abeja.png", len(test))
    tables = Path(args.paper) / "tablas"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "v20_ablacion_abeja.tex").write_text(latex_table(summary, len(test)), encoding="utf-8")
    for s in summary:
        print(f"{s.label:30s} {s.median_mean:6.1f}° ± {s.median_sd:4.1f}  ({s.parameters:,} parámetros)")


if __name__ == "__main__":
    main()
