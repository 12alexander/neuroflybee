"""Registro del costo computacional (R6, v19): hardware, tiempo por etapa y horas de GPU.

No depende de PyTorch: si está instalado, detecta la GPU (p. ej. «Tesla T4» en Colab).
"""
from __future__ import annotations

import csv
import json
import platform
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional


def detect_hardware() -> dict:
    info = {"cpu": platform.processor() or platform.machine(), "sistema": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "gpu": "ninguna", "torch": "no instalado", "cuda": "-"}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda or "-"
    except ImportError:
        pass
    return info


@dataclass
class StageCost:
    stage: str
    seconds: float
    uses_gpu: bool
    status: str               # ok | error | omitida
    started: str

    @property
    def gpu_hours(self) -> float:
        return self.seconds / 3600 if self.uses_gpu else 0.0


@dataclass
class CostTracker:
    hardware: dict = field(default_factory=detect_hardware)
    stages: list[StageCost] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str, uses_gpu: bool = False) -> Iterator[None]:
        t0, started = time.perf_counter(), datetime.now().isoformat(timespec="seconds")
        status = "ok"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            gpu = uses_gpu and self.hardware.get("gpu", "ninguna") != "ninguna"
            self.stages.append(StageCost(name, time.perf_counter() - t0, gpu, status, started))

    def record(self, name: str, seconds: float, uses_gpu: bool, status: str = "ok") -> None:
        gpu = uses_gpu and self.hardware.get("gpu", "ninguna") != "ninguna"
        self.stages.append(StageCost(name, seconds, gpu, status, datetime.now().isoformat(timespec="seconds")))

    @property
    def total_hours(self) -> float:
        return sum(s.seconds for s in self.stages) / 3600

    @property
    def gpu_hours(self) -> float:
        return sum(s.gpu_hours for s in self.stages)

    def write(self, csv_path: str | Path, json_path: Optional[str | Path] = None) -> None:
        csv_path = Path(csv_path); csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["etapa", "segundos", "usa_gpu", "horas_gpu", "estado", "inicio"])
            for s in self.stages:
                w.writerow([s.stage, f"{s.seconds:.1f}", s.uses_gpu, f"{s.gpu_hours:.4f}", s.status, s.started])
            w.writerow(["TOTAL", f"{self.total_hours * 3600:.1f}", "", f"{self.gpu_hours:.4f}", "", ""])
        if json_path:
            Path(json_path).write_text(json.dumps({"hardware": self.hardware, "horas_totales": self.total_hours,
                                                   "horas_gpu": self.gpu_hours,
                                                   "etapas": [asdict(s) for s in self.stages]},
                                                  indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, json_path: str | Path) -> "CostTracker":
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return cls(hardware=data["hardware"], stages=[StageCost(**s) for s in data["etapas"]])
