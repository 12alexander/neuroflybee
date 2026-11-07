"""Etapas del pipeline (datos puros: qué comandos corre cada una)."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    commands: tuple[tuple[str, ...], ...]
    quick_commands: tuple[tuple[str, ...], ...] | None = None
    uses_gpu: bool = False
    optional: bool = False            # sólo corre si se pide explícitamente (p. ej. el video)

    def commands_for(self, quick: bool) -> tuple[tuple[str, ...], ...]:
        return self.quick_commands if quick and self.quick_commands is not None else self.commands


def _py(script: str, *args: str) -> tuple[str, ...]:
    return (sys.executable, f"scripts/{script}", *args)


def default_stages() -> list[Stage]:
    return [
        Stage("pruebas", "Todas las pruebas, ahora con PyTorch (0 saltadas esperadas)",
              ((sys.executable, "-m", "pytest", "-q", "--deselect", "tests/test_video.py"),)),
        Stage("datos", "Demostraciones del experto (mosca) y pares de vistas (abeja)", (_py("generate_data.py"),)),
        Stage("validacion_sensor", "Sensor 2,5D: IDF, rumbo y área de captura del ALV (v21, sin GPU)",
              (_py("validate_sensor.py"),), (_py("validate_sensor.py", "--scenarios", "10"),)),
        Stage("mosca", "Imitación: MLP, CNN, CNN+GRU y CNN+Transformer × 3 semillas", (_py("train_fly.py"),),
              (_py("train_fly.py", "--models", "fly_cnn_gru", "--seeds", "0"),), uses_gpu=True),
        Stage("dagger", "3 iteraciones de DAgger sobre CNN+GRU", (_py("train_dagger.py"),),
              (_py("train_dagger.py", "--seeds", "0"),), uses_gpu=True),
        Stage("abeja", "Red siamesa de retorno visual", (_py("train_bee.py"),),
              (_py("train_bee.py", "--seeds", "0"),), uses_gpu=True),
        Stage("ablacion_abeja", "Abeja en PRUEBA: siamesa vs vista única, sin resta y lectura vectorial (v20)",
              (_py("run_bee_ablation.py"),), (_py("run_bee_ablation.py", "--seeds", "0"),), uses_gpu=True),
        Stage("benchmark", "Clásicos + neuronales × 6 condiciones × 30 escenarios de prueba, y ablaciones",
              (_py("run_benchmark.py"),), (_py("run_benchmark.py", "--episodes", "10"),), uses_gpu=True),
        Stage("significancia", "IC bootstrap pareado, permutación exacta, Holm y h de Cohen (v22, sin GPU)",
              (_py("significance.py"),)),
        Stage("figuras", "Curvas, DAgger, retorno, controladores y visor",
              (_py("plot_fly_training.py"), _py("plot_dagger.py"), _py("show_bee_homing.py"),
               _py("show_controllers.py"), _py("build_viewer.py"))),
        Stage("interpretabilidad", "Análisis de errores, Grad-CAM e Integrated Gradients",
              (_py("explain.py", "--episodes", "30"),), (_py("explain.py", "--episodes", "10"),), uses_gpu=True),
        Stage("video", "Video de respaldo con los controladores neuronales", (_py("record_video.py"),), optional=True),
        Stage("entorno", "Versiones exactas del entorno (requirements-lock.txt)", (_py("freeze_env.py"),)),
        Stage("pesos", "Manifiesto SHA-256 y .zip de los pesos para Hugging Face / Drive", (_py("package_weights.py", "pack"),)),
        Stage("numeros", "Números y tablas del artículo desde los CSV", (_py("collect_results.py"),)),
    ]


def select(stages: Sequence[Stage], only: Sequence[str] = (), skip: Sequence[str] = (),
           include_optional: Sequence[str] = ()) -> list[Stage]:
    names = {s.name for s in stages}
    unknown = (set(only) | set(skip) | set(include_optional)) - names
    if unknown:
        raise ValueError(f"Etapas desconocidas: {sorted(unknown)}. Hay: {[s.name for s in stages]}")
    chosen = [s for s in stages if (not only or s.name in only) and s.name not in skip]
    return [s for s in chosen if not s.optional or s.name in include_optional or s.name in only]
