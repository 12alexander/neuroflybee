"""Lectura y resumen de los historiales de entrenamiento (sin PyTorch)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_histories(results_dir: str | Path, tag: str = "v9") -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(results_dir).glob(f"{tag}_historial_*.json"))]


def best_rows(histories: list[dict]) -> list[dict]:
    """Métricas de validación en la mejor época de cada (modelo, semilla)."""
    rows = []
    for h in histories:
        best = h["history"][h["best_epoch"] - 1]
        rows.append({"model": h["model"], "seed": h["seed"], "parameters": h["parameters"],
                     "best_epoch": h["best_epoch"], "f1_macro": best["metrics"]["f1_macro"],
                     "accuracy": best["metrics"]["accuracy"], "seconds": h["total_seconds"],
                     "device": h["device"]})
    return rows


def aggregate(rows: list[dict]) -> dict[str, dict]:
    """Media ± d.e. entre semillas por modelo."""
    out: dict[str, dict] = {}
    for model in sorted({r["model"] for r in rows}):
        sel = [r for r in rows if r["model"] == model]
        f1, acc = np.array([r["f1_macro"] for r in sel]), np.array([r["accuracy"] for r in sel])
        out[model] = {"n_seeds": len(sel), "f1_mean": float(f1.mean()), "f1_std": float(f1.std()),
                      "acc_mean": float(acc.mean()), "acc_std": float(acc.std()),
                      "parameters": sel[0]["parameters"], "train_seconds": float(sum(r["seconds"] for r in sel)),
                      "device": sel[0]["device"]}
    return out


def write_summary_csv(path: str | Path, summary: dict[str, dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo", "semillas", "f1_macro_media", "f1_macro_de", "accuracy_media", "accuracy_de",
                    "parametros", "segundos_entrenamiento", "dispositivo"])
        for model, s in summary.items():
            w.writerow([model, s["n_seeds"], f"{s['f1_mean']:.4f}", f"{s['f1_std']:.4f}", f"{s['acc_mean']:.4f}",
                        f"{s['acc_std']:.4f}", s["parameters"], f"{s['train_seconds']:.0f}", s["device"]])
