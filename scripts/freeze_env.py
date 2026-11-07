"""Congela las versiones EXACTAS del entorno donde se generaron los resultados (v19, C5).

`requirements.txt` declara versiones mínimas; `requirements-lock.txt` registra las que
realmente se usaron (en Colab), más Python, GPU y CUDA, para poder reproducir los números.

Uso:  python scripts/freeze_env.py [--out requirements-lock.txt]
"""
from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroflybee.training.cost import detect_hardware  # noqa: E402


def declared_packages(requirements: Path) -> list[str]:
    names = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.append(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0])
    return names


def freeze(requirements: Path) -> tuple[list[str], list[str]]:
    pinned, missing = [], []
    for name in declared_packages(requirements):
        try:
            pinned.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            missing.append(name)
    return pinned, missing


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requirements", default=str(ROOT / "requirements.txt"))
    parser.add_argument("--out", default=str(ROOT / "requirements-lock.txt"))
    args = parser.parse_args(argv)
    pinned, missing = freeze(Path(args.requirements))
    hw = detect_hardware()
    header = [f"# Versiones exactas usadas para los resultados (generado por scripts/freeze_env.py)",
              f"# Python {hw['python']} · {hw['sistema']} · GPU: {hw['gpu']} · CUDA {hw['cuda']} · torch {hw['torch']}"]
    if missing:
        header.append(f"# No instalados en este entorno: {', '.join(missing)}")
    Path(args.out).write_text("\n".join(header + pinned) + "\n", encoding="utf-8")
    print(f"{len(pinned)} paquetes fijados en {args.out}" + (f" ({len(missing)} no instalados)" if missing else ""))


if __name__ == "__main__":
    main()
