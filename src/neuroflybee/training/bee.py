"""Entrenamiento de la red de la abeja (regresión del vector de retorno)."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.generate import BeeDataset
from ..evaluation.metrics import homing_report
from ..models.bee import BeeHomingLoss, BeeHomingNet
from ..models.bee_spec import BeeNetConfig, count_bee_parameters
from ..utils.seeding import set_global_seed
from .config import TrainConfig
from .torch_data import make_loader
from .trainer import FitResult, Trainer, resolve_device


class BeePairDataset(Dataset):
    """(vista actual, vista memorizada, objetivo). Con `max_shift > 0` la vista actual se
    rota circularmente ±`max_shift` bins (simula el error de la brújula, ±5° por bin)."""

    def __init__(self, data: BeeDataset, max_shift: int = 0):
        self.current = torch.from_numpy(np.ascontiguousarray(data.current)).float()
        self.home = torch.from_numpy(np.ascontiguousarray(data.home)).float()
        self.target = torch.from_numpy(np.ascontiguousarray(data.target)).float()
        self.max_shift = max_shift

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, i: int):
        cur = self.current[i]
        if self.max_shift:
            # torch.randint usa la semilla de cada worker (fijada por el DataLoader)
            cur = torch.roll(cur, int(torch.randint(-self.max_shift, self.max_shift + 1, (1,))))
        return cur, self.home[i], self.target[i]


def homing_metrics(output: torch.Tensor, target: torch.Tensor) -> dict:
    return homing_report(output.numpy(), target.numpy())


def save_bee_checkpoint(path: str | Path, model: BeeHomingNet, epoch: int, metrics: dict, seed: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "bee", "config": asdict(model.cfg), "state_dict": model.state_dict(),
                "epoch": epoch, "metrics": metrics, "seed": seed}, path)


def load_bee_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[BeeHomingNet, dict]:
    from ..config import from_dict

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "bee":
        raise ValueError(f"{path} no es un checkpoint de la abeja")
    model = BeeHomingNet(from_dict(BeeNetConfig, payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


def train_bee_model(train: BeeDataset, val: BeeDataset, net_cfg: BeeNetConfig, tcfg: TrainConfig, seed: int,
                    checkpoint_dir: Path, results_dir: Path, log: Callable[[str], None] = print,
                    run_name: Optional[str] = None, history_tag: str = "v11") -> FitResult:
    """Entrena una red de la abeja con una semilla y guarda el mejor checkpoint (error angular
    mediano de validación) y su historial.

    `run_name` cambia el nombre de los archivos (por defecto `bee_s<semilla>`, el de v11) y
    `history_tag` el prefijo del historial; la ablación de v20 usa ambos para no pisar los
    pesos del modelo principal."""
    run_name = run_name or f"bee_s{seed}"
    set_global_seed(seed)
    device = resolve_device(tcfg.device)
    model = BeeHomingNet(net_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.epochs)
    trainer = Trainer(model, optimizer, BeeHomingLoss(), homing_metrics, "angle_error_median", device,
                      mode="min", scheduler=scheduler, grad_clip=tcfg.grad_clip)
    pin = device.type == "cuda"
    train_loader = make_loader(BeePairDataset(train, max_shift=1), tcfg.batch_size, True, seed,
                               tcfg.num_workers, pin)
    val_loader = make_loader(BeePairDataset(val), tcfg.batch_size, False, seed, tcfg.num_workers, pin)
    ckpt = Path(checkpoint_dir) / f"{run_name}.pt"
    log(f"== abeja ({net_cfg.fusion}, cabeza {net_cfg.head}) | semilla {seed} | "
        f"{count_bee_parameters(net_cfg):,} parámetros | {device} ==")
    result = trainer.fit(train_loader, val_loader, tcfg.epochs, tcfg.patience,
                         lambda m, e, met: save_bee_checkpoint(ckpt, m, e, met, seed), log)
    result.save_json(Path(results_dir) / f"{history_tag}_historial_{run_name}.json",
                     extra={"model": "bee", "seed": seed, "parameters": count_bee_parameters(net_cfg),
                            "fusion": net_cfg.fusion, "head": net_cfg.head})
    return result


@torch.no_grad()
def predict_bee(model: BeeHomingNet, data: BeeDataset, batch_size: int = 2048,
                device: str | torch.device = "cpu") -> np.ndarray:
    """Salidas crudas (N, 3) = (cos, sin, log(1+d)) de la red sobre todo un conjunto."""
    model = model.to(device).eval()
    outputs = []
    for start in range(0, len(data), batch_size):
        cur = torch.from_numpy(np.ascontiguousarray(data.current[start:start + batch_size])).float().to(device)
        home = torch.from_numpy(np.ascontiguousarray(data.home[start:start + batch_size])).float().to(device)
        outputs.append(model(cur, home).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros((0, 3), dtype=np.float32)
