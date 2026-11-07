"""v23: cuadernos de Colab válidos y demostración de inferencia de punta a punta."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


def _python_source(cell: dict) -> str:
    """Código de una celda con los comandos de shell de IPython (`!…`) reemplazados por `pass`."""
    lines = []
    for line in "".join(cell["source"]).splitlines():
        stripped = line.lstrip()
        lines.append(line[: len(line) - len(stripped)] + "pass" if stripped.startswith(("!", "%")) else line)
    return "\n".join(lines)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_and_references_existing_scripts(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["cells"]
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile(_python_source(cell), f"{path.name}[{i}]", "exec")
            assert not cell.get("outputs"), "los cuadernos se versionan sin salidas"
    text = "".join("".join(c["source"]) for c in nb["cells"])
    for script in set(re.findall(r"scripts/(\w+)\.py", text)):
        assert (ROOT / "scripts" / f"{script}.py").exists(), script


def test_both_notebooks_exist():
    assert {p.name for p in NOTEBOOKS} >= {"01_entrenar_colab.ipynb", "02_demo_inferencia.ipynb"}


def _demo():
    spec = importlib.util.spec_from_file_location("demo", ROOT / "scripts" / "demo_inference.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["demo"] = mod            # @dataclass necesita el módulo registrado
    spec.loader.exec_module(mod)
    return mod


def test_demo_inference_with_classical_controllers(tmp_path):
    _demo().main(["--familia", "gps_loss", "--semilla", "20001", "--checkpoints", str(tmp_path / "sin_pesos"),
                  "--results", str(tmp_path)])
    assert (tmp_path / "figures" / "v23_demo_gps_loss_20001.png").stat().st_size > 30_000
    rows = (tmp_path / "v23_demo_gps_loss_20001.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("controlador") and {r.split(",")[0] for r in rows[1:]} == {"astar", "astar_alv"}


def test_demo_rejects_non_test_seeds(tmp_path):
    with pytest.raises(SystemExit):
        _demo().main(["--semilla", "5", "--results", str(tmp_path)])
