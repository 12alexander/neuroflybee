"""Ejecuta las etapas: registra el costo, guarda el registro de cada una y permite reanudar."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .. import __version__
from ..training.cost import CostTracker
from .stages import Stage

Log = Callable[[str], None]


@dataclass
class PipelineRunner:
    root: Path
    results: Path
    quick: bool = False
    force: bool = False
    log: Log = print

    @property
    def state_path(self) -> Path:
        return self.results / "v19_pipeline_estado.json"

    @property
    def cost_json(self) -> Path:
        return self.results / "v19_costo.json"

    def _load_state(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def _tracker(self) -> CostTracker:
        # acumula el costo de corridas anteriores (Colab puede desconectarse a mitad)
        return CostTracker.load(self.cost_json) if self.cost_json.exists() and not self.force else CostTracker()

    def run(self, stages: Sequence[Stage], dry_run: bool = False) -> bool:
        state, tracker = self._load_state(), self._tracker()
        mode = "rápido" if self.quick else "completo"
        self.log(f"Pipeline v{__version__} ({mode}) · hardware: {tracker.hardware.get('gpu')} · {len(stages)} etapas")
        for stage in stages:
            key = f"{stage.name}:{mode}"
            if state.get(key) == "ok" and not self.force:
                self.log(f"↷ {stage.name}: ya completada (use --force para repetir)")
                continue
            commands = stage.commands_for(self.quick)
            self.log(f"▶ {stage.name} — {stage.description}")
            for cmd in commands:
                self.log("    $ " + " ".join(cmd))
            if dry_run:
                continue
            ok, seconds = self._run_commands(stage, commands)
            tracker.record(stage.name, seconds, stage.uses_gpu, "ok" if ok else "error")
            tracker.write(self.results / "v19_costo.csv", self.cost_json)
            state[key] = "ok" if ok else "error"
            self._save_state(state)
            self.log(f"{'✔' if ok else '✘'} {stage.name}: {seconds / 60:.1f} min")
            if not ok:
                self.log(f"   Revise results/logs/v19_{stage.name}.log y vuelva a ejecutar: se reanuda aquí.")
                return False
        if not dry_run:
            self.log(f"Listo. Horas totales {tracker.total_hours:.2f} · horas de GPU {tracker.gpu_hours:.2f}")
        return True

    def _run_commands(self, stage: Stage, commands) -> tuple[bool, float]:
        log_path = self.results / "logs" / f"v19_{stage.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as fh:
            for cmd in commands:
                fh.write("$ " + " ".join(cmd) + "\n"); fh.flush()
                proc = subprocess.run(list(cmd), cwd=self.root, stdout=fh, stderr=subprocess.STDOUT)
                if proc.returncode != 0:
                    return False, time.perf_counter() - t0
        return True, time.perf_counter() - t0
