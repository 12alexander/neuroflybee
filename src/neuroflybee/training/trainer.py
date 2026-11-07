"""Entrenador genérico (PyTorch).

No sabe qué red entrena ni qué problema resuelve: recibe el modelo, el optimizador, la
pérdida y una función de métricas (inyección de dependencias). Lo reutilizan la mosca
(clasificación de acciones, v9–v10) y la abeja (regresión del vector de retorno, v11).
Cada lote es una tupla `(*entradas, objetivo)`.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

MetricFn = Callable[[torch.Tensor, torch.Tensor], dict]


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    metrics: dict
    lr: float
    seconds: float


@dataclass
class FitResult:
    history: list[EpochLog] = field(default_factory=list)
    best_epoch: int = -1
    best_score: float = float("nan")
    total_seconds: float = 0.0
    device: str = "cpu"

    def save_json(self, path: str | Path, extra: Optional[dict] = None) -> None:
        payload = {"best_epoch": self.best_epoch, "best_score": self.best_score,
                   "total_seconds": self.total_seconds, "device": self.device,
                   "history": [asdict(h) for h in self.history], **(extra or {})}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
                 metric_fn: MetricFn, monitor: str, device: torch.device, mode: str = "max",
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None, grad_clip: float = 1.0):
        if mode not in ("max", "min"):
            raise ValueError("mode debe ser 'max' o 'min'")
        self.model = model.to(device)
        self.optimizer, self.loss_fn = optimizer, loss_fn.to(device)
        self.metric_fn, self.monitor, self.mode = metric_fn, monitor, mode
        self.device, self.scheduler, self.grad_clip = device, scheduler, grad_clip

    def _to_device(self, batch) -> tuple[list[torch.Tensor], torch.Tensor]:
        *inputs, target = batch
        return [x.to(self.device, non_blocking=True) for x in inputs], target.to(self.device, non_blocking=True)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, n = 0.0, 0
        for batch in loader:
            inputs, target = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model(*inputs), target)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total += float(loss.detach()) * len(target)
            n += len(target)
        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Salidas y objetivos de todo el conjunto (en CPU) y la pérdida media."""
        self.model.eval()
        outputs, targets, total, n = [], [], 0.0, 0
        for batch in loader:
            inputs, target = self._to_device(batch)
            out = self.model(*inputs)
            total += float(self.loss_fn(out, target)) * len(target)
            n += len(target)
            outputs.append(out.cpu())
            targets.append(target.cpu())
        return torch.cat(outputs), torch.cat(targets), total / max(n, 1)

    def _better(self, score: float, best: float) -> bool:
        if np.isnan(best):
            return True
        return score > best if self.mode == "max" else score < best

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, patience: int,
            checkpoint: Optional[Callable[[nn.Module, int, dict], None]] = None,
            log: Callable[[str], None] = print) -> FitResult:
        result = FitResult(device=str(self.device))
        start, waited = time.perf_counter(), 0
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss = self.train_epoch(train_loader)
            outputs, targets, val_loss = self.predict(val_loader)
            metrics = self.metric_fn(outputs, targets)
            lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()
            entry = EpochLog(epoch, train_loss, val_loss, metrics, lr, time.perf_counter() - t0)
            result.history.append(entry)
            score = float(metrics[self.monitor])
            improved = self._better(score, result.best_score)
            if improved:
                result.best_epoch, result.best_score, waited = epoch, score, 0
                if checkpoint is not None:
                    checkpoint(self.model, epoch, metrics)
            else:
                waited += 1
            log(f"época {epoch:2d}  pérdida {train_loss:.4f}/{val_loss:.4f}  "
                f"{self.monitor} {score:.4f}{'  ★' if improved else ''}  ({entry.seconds:.1f} s)")
            if waited >= patience:
                log(f"parada temprana: {patience} épocas sin mejorar")
                break
        result.total_seconds = time.perf_counter() - start
        return result
