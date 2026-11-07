"""Guardado y carga de pesos con su configuración (para reproducir y para el dashboard)."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ..config import from_dict
from ..models.fly import FlyPolicyNet, build_fly_model
from ..models.spec import FlyNetConfig


def save_fly_checkpoint(path: str | Path, model_name: str, model: FlyPolicyNet, epoch: int,
                        metrics: dict, seed: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "fly", "model_name": model_name, "config": asdict(model.cfg),
                "state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics, "seed": seed}, path)


def load_fly_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[FlyPolicyNet, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "fly":
        raise ValueError(f"{path} no es un checkpoint de la mosca")
    cfg = from_dict(FlyNetConfig, payload["config"])
    model = build_fly_model(payload["model_name"], cfg)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload
