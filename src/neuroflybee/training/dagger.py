"""DAgger (Ross et al., 2011) para la mosca.

Problema que resuelve: con imitación pura la red sólo vio estados por los que pasó el
experto; cuando ella se equivoca llega a estados nuevos y no sabe salir (desplazamiento
de distribución). DAgger deja volar a la red, pide al experto la acción correcta en esos
estados, agrega los datos y reentrena.

    iteración 0 : modelo de la v9 (imitación)            → evaluación en lazo cerrado
    iteración k : volar con β_k → etiquetar → agregar → reentrenar → evaluación
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import torch

from ..config import ProjectConfig
from ..control.composite import NavigationController
from ..control.navigation import MissionNavigator
from ..control.neural import FlySteering
from ..data.generate import FLY_TRAIN_FAMILIES, FlyDataset, collect_fly_episodes
from ..evaluation.closed_loop import evaluate, summarize
from ..evaluation.metrics import classification_report
from ..inference.fly_runner import FlyPolicyRunner
from ..models.fly import FlyPolicyNet
from ..sim.scenarios import seeds_for
from .checkpoints import load_fly_checkpoint
from .config import TrainConfig
from .dagger_config import DaggerConfig
from .fly import train_fly_model
from .torch_data import FlySequenceDataset, make_loader
from .trainer import resolve_device


def closed_loop_eval(cfg: ProjectConfig, model: FlyPolicyNet, dcfg: DaggerConfig, device) -> dict:
    """Éxito y choques de la mosca volando sola en escenarios de VALIDACIÓN."""
    runner = FlyPolicyRunner(model, device)
    make = lambda: NavigationController("mosca", MissionNavigator(), FlySteering(runner))  # noqa: E731
    seeds = seeds_for("val", dcfg.eval_episodes_per_family)
    results = evaluate(cfg, make, dcfg.eval_families, seeds)
    by_family = {f: summarize([r for r in results if r.family == f]) for f in dcfg.eval_families}
    return {"all": summarize(results), "by_family": by_family}


@torch.no_grad()
def offline_f1(model: FlyPolicyNet, val: FlyDataset, device, batch_size: int = 512) -> float:
    loader = make_loader(FlySequenceDataset(val, model.cfg.seq_len), batch_size, False, 0, 0, False)
    model.eval()
    preds, targets = [], []
    for frames, goal, target in loader:
        preds.append(model(frames.to(device), goal.to(device)).argmax(-1).cpu())
        targets.append(target)
    return classification_report(torch.cat(targets).numpy(), torch.cat(preds).numpy(),
                                 model.cfg.n_actions).f1_macro


def run_dagger(cfg: ProjectConfig, dcfg: DaggerConfig, tcfg: TrainConfig, seed: int, train: FlyDataset,
               val: FlyDataset, checkpoint_dir: Path, results_dir: Path,
               log: Callable[[str], None] = print) -> dict:
    device = resolve_device(tcfg.device)
    start_ckpt = Path(checkpoint_dir) / f"{dcfg.model}_s{seed}.pt"
    if not start_ckpt.exists():
        raise FileNotFoundError(f"Falta {start_ckpt}: entrene primero la v9 (scripts/train_fly.py)")
    model, _ = load_fly_checkpoint(start_ckpt, device)
    t0 = time.perf_counter()
    report = {"model": dcfg.model, "seed": seed, "config": asdict(dcfg), "iterations": []}
    log(f"== DAgger {dcfg.model} s{seed}: iteración 0 (imitación v9) ==")
    report["iterations"].append({"iteration": 0, "beta": 1.0, "samples": len(train),
                                 "val_f1_macro": offline_f1(model, val, device),
                                 "closed_loop": closed_loop_eval(cfg, model, dcfg, device)})
    data = train
    for it in range(1, dcfg.iterations + 1):
        beta = dcfg.beta(it)
        runner = FlyPolicyRunner(model, device)
        new = collect_fly_episodes(cfg, "train", dcfg.episodes_per_iteration, families=FLY_TRAIN_FAMILIES,
                                   offset=dcfg.new_seed_offset(it), behavior=runner, beta=beta,
                                   seed=seed + 100 * it)
        data = FlyDataset.concat([data, new])
        log(f"== iteración {it}: β={beta}, +{len(new):,} muestras → {len(data):,} ==")
        result = train_fly_model(dcfg.model, data, val, model.cfg, replace(tcfg, epochs=dcfg.epochs_per_iteration),
                                 seed, checkpoint_dir, results_dir, tag="v10", log=log,
                                 init_state_dict=model.state_dict(), run_name=f"{dcfg.model}_dagger{it}_s{seed}")
        model, _ = load_fly_checkpoint(Path(checkpoint_dir) / f"{dcfg.model}_dagger{it}_s{seed}.pt", device)
        report["iterations"].append({"iteration": it, "beta": beta, "samples": len(data), "new_samples": len(new),
                                     "val_f1_macro": result.best_score,
                                     "closed_loop": closed_loop_eval(cfg, model, dcfg, device)})
        log(f"   éxito en lazo cerrado: {report['iterations'][-1]['closed_loop']['all']['success']:.2f}")
    report["total_seconds"] = time.perf_counter() - t0
    report["device"] = str(device)
    out = Path(results_dir) / f"v10_dagger_{dcfg.model}_s{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    final = Path(checkpoint_dir) / f"{dcfg.model}_dagger{dcfg.iterations}_s{seed}.pt"
    log(f"Pesos finales: {final}")
    return report
