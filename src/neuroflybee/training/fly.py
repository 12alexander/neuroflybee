"""Entrenamiento por imitación (clonación de comportamiento) de las redes de la mosca."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn

from ..data.generate import FlyDataset
from ..evaluation.metrics import (classification_report, confusion_matrix, inverse_frequency_weights,
                                  roc_auc_ovr_macro)
from ..models.fly import build_fly_model, count_trainable
from ..models.spec import FlyNetConfig
from ..utils.seeding import set_global_seed
from .checkpoints import save_fly_checkpoint
from .config import TrainConfig
from .torch_data import FlySequenceDataset, make_loader
from .trainer import FitResult, Trainer, resolve_device


def classification_metrics(n_classes: int) -> Callable[[torch.Tensor, torch.Tensor], dict]:
    """Exactitud, F1 macro (criterio de selección), F1 por clase y AUROC macro (v22)."""
    def metric_fn(logits: torch.Tensor, target: torch.Tensor) -> dict:
        pred = logits.argmax(dim=-1).numpy()
        rep = classification_report(target.numpy(), pred, n_classes)
        probs = torch.softmax(logits.float(), dim=-1).numpy()
        return {"accuracy": rep.accuracy, "f1_macro": rep.f1_macro, "f1_per_class": list(rep.f1_per_class),
                "auroc_macro": roc_auc_ovr_macro(target.numpy(), probs, n_classes)}
    return metric_fn


def train_fly_model(model_name: str, train: FlyDataset, val: FlyDataset, net_cfg: FlyNetConfig,
                    tcfg: TrainConfig, seed: int, checkpoint_dir: Path, results_dir: Path,
                    tag: str = "v9", log: Callable[[str], None] = print,
                    init_state_dict: Optional[dict] = None, run_name: Optional[str] = None) -> FitResult:
    """Entrena un modelo con una semilla; guarda el mejor checkpoint (F1 macro de
    validación), el historial JSON y la matriz de confusión de validación.

    `init_state_dict` permite continuar desde pesos previos (DAgger, v10) y `run_name`
    cambia el nombre de los archivos (por defecto `<modelo>_s<semilla>`)."""
    set_global_seed(seed)
    device = resolve_device(tcfg.device)
    model = build_fly_model(model_name, net_cfg)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    weights = inverse_frequency_weights(train.action_counts(), tcfg.class_weight_power, tcfg.max_class_weight)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.epochs)
    trainer = Trainer(model, optimizer, loss_fn, classification_metrics(net_cfg.n_actions), "f1_macro",
                      device, mode="max", scheduler=scheduler, grad_clip=tcfg.grad_clip)
    pin = device.type == "cuda"
    train_loader = make_loader(FlySequenceDataset(train, net_cfg.seq_len), tcfg.batch_size, True, seed,
                               tcfg.num_workers, pin)
    val_loader = make_loader(FlySequenceDataset(val, net_cfg.seq_len), tcfg.batch_size, False, seed,
                             tcfg.num_workers, pin)
    run_name = run_name or f"{model_name}_s{seed}"
    ckpt_path = Path(checkpoint_dir) / f"{run_name}.pt"

    def checkpoint(m: nn.Module, epoch: int, metrics: dict) -> None:
        save_fly_checkpoint(ckpt_path, model_name, m, epoch, metrics, seed)

    log(f"== {model_name} | semilla {seed} | {count_trainable(model):,} parámetros | {device} ==")
    result = trainer.fit(train_loader, val_loader, tcfg.epochs, tcfg.patience, checkpoint, log)

    # matriz de confusión del MEJOR modelo en validación
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    trainer.model.load_state_dict(best["state_dict"])
    logits, target, _ = trainer.predict(val_loader)
    cm = confusion_matrix(target.numpy(), logits.argmax(-1).numpy(), net_cfg.n_actions)
    results_dir = Path(results_dir)
    result.save_json(results_dir / f"{tag}_historial_{run_name}.json",
                     extra={"model": model_name, "seed": seed, "parameters": count_trainable(model),
                            "class_weights": weights.tolist(), "confusion_val": cm.tolist()})
    return result
