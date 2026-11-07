"""v19: pipeline de resultados reales, costo (R6), pesos, entorno y números del artículo."""
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

from neuroflybee.pipeline.runner import PipelineRunner
from neuroflybee.pipeline.stages import Stage, default_stages, select
from neuroflybee.reporting.numbers import NumberSet, collect, cost_table
from neuroflybee.training.cost import CostTracker

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ costo
def test_cost_tracker_roundtrip(tmp_path):
    t = CostTracker(hardware={"gpu": "Tesla T4"})
    with t.stage("a", uses_gpu=True):
        pass
    t.record("b", 3600, uses_gpu=False)
    t.record("c", 1800, uses_gpu=True)
    assert t.gpu_hours == pytest.approx(0.5, abs=1e-3)
    t.write(tmp_path / "c.csv", tmp_path / "c.json")
    back = CostTracker.load(tmp_path / "c.json")
    assert [s.stage for s in back.stages] == ["a", "b", "c"] and back.gpu_hours == pytest.approx(t.gpu_hours)
    assert "TOTAL" in (tmp_path / "c.csv").read_text()


def test_no_gpu_means_zero_gpu_hours():
    t = CostTracker(hardware={"gpu": "ninguna"})
    t.record("mosca", 7200, uses_gpu=True)
    assert t.gpu_hours == 0 and t.total_hours == pytest.approx(2)


# ------------------------------------------------------------------ etapas
def test_stage_selection():
    stages = default_stages()
    names = [s.name for s in select(stages)]
    assert names[0] == "pruebas" and names[-1] == "numeros" and "video" not in names
    assert "video" in [s.name for s in select(stages, include_optional=["video"])]
    assert [s.name for s in select(stages, only=["benchmark", "numeros"])] == ["benchmark", "numeros"]
    assert "mosca" not in [s.name for s in select(stages, skip=["mosca"])]
    with pytest.raises(ValueError):
        select(stages, only=["entrenar_todo"])
    mosca = next(s for s in stages if s.name == "mosca")
    assert mosca.commands_for(True) != mosca.commands_for(False) and mosca.uses_gpu


def _fake(name, code):
    return Stage(name, name, ((sys.executable, "-c", code),))


def test_runner_records_cost_and_resumes(tmp_path):
    runner = PipelineRunner(ROOT, tmp_path, log=lambda m: None)
    assert runner.run([_fake("uno", "print(1)"), _fake("dos", "print(2)")])
    cost = json.loads((tmp_path / "v19_costo.json").read_text())
    assert [s["stage"] for s in cost["etapas"]] == ["uno", "dos"]
    assert (tmp_path / "logs" / "v19_uno.log").read_text().count("print(1)") == 1
    # segunda corrida: ambas ya están completas → no se repiten
    assert runner.run([_fake("uno", "raise SystemExit(1)"), _fake("dos", "raise SystemExit(1)")])


def test_runner_stops_at_failure(tmp_path):
    runner = PipelineRunner(ROOT, tmp_path, log=lambda m: None)
    assert not runner.run([_fake("falla", "raise SystemExit(3)"), _fake("nunca", "print(1)")])
    state = json.loads(runner.state_path.read_text())
    assert state == {"falla:completo": "error"}


def test_dry_run_runs_nothing(tmp_path):
    assert PipelineRunner(ROOT, tmp_path, log=lambda m: None).run(default_stages(), dry_run=True)
    assert not (tmp_path / "v19_costo.json").exists()


# ------------------------------------------------------------------ pesos y entorno
def test_weights_manifest_pack_and_verify(tmp_path):
    pw = _script("package_weights")
    ck = tmp_path / "checkpoints"; ck.mkdir()
    (ck / "fly_cnn_gru_s0.pt").write_bytes(b"pesos" * 100)
    (ck / "bee_s0.pt").write_bytes(b"abeja")
    manifest = pw.pack(ck, tmp_path / "pesos.zip")
    assert {e["archivo"] for e in manifest["archivos"]} == {"fly_cnn_gru_s0.pt", "bee_s0.pt"}
    assert "checkpoints/MANIFEST.json" in zipfile.ZipFile(tmp_path / "pesos.zip").namelist()
    assert pw.verify(ck) == []
    (ck / "bee_s0.pt").write_bytes(b"otra cosa")
    assert pw.verify(ck) == ["bee_s0.pt no coincide con el manifiesto"]


def test_freeze_env_pins_installed_packages(tmp_path):
    fe = _script("freeze_env")
    req = tmp_path / "r.txt"
    req.write_text("numpy>=1.24  # numérico\npaquete-que-no-existe>=1\n\n")
    pinned, missing = fe.freeze(req)
    assert pinned[0].startswith("numpy==") and missing == ["paquete-que-no-existe"]


# ------------------------------------------------------------------ números del artículo
def test_numbers_from_current_results():
    ns, cost = collect(ROOT / "results")
    v = {k: val for k, (val, _) in ns.values.items()}
    assert v["ExitoAstarMoviles"] == "40\\,\\%"          # results/v14_tabla_principal.csv
    assert v["MuestrasMoscaTrain"] == "76\\,077"
    assert v["ErrorDerivaRango"] == "21--28\\,\\%"
    assert all(n.isalpha() for n in ns.values)
    if not list((ROOT / "results").glob("v9_historial_*.json")):
        assert ns.is_pending("FUnoGRU") and "pendiente" in v["FUnoGRU"]


def test_numbers_use_training_outputs(tmp_path):
    import shutil
    for f in (ROOT / "results").glob("v*.csv"):
        shutil.copy(f, tmp_path)
    for seed, score in ((0, 0.80), (1, 0.84), (2, 0.82)):
        (tmp_path / f"v9_historial_fly_cnn_gru_s{seed}.json").write_text(json.dumps({"best_score": score}))
    t = CostTracker(hardware={"gpu": "Tesla_T4"}); t.record("mosca", 3600, True)
    t.write(tmp_path / "v19_costo.csv", tmp_path / "v19_costo.json")
    ns, cost = collect(tmp_path)
    assert ns.values["FUnoGRU"][0] == "0.820 $\\pm$ 0.020"
    assert ns.values["HorasGPU"][0] == "1.00" and ns.values["GPU"][0] == "Tesla\\_T4"
    assert "mosca & 60.0 & 1.00" in cost_table(cost)


def test_pending_text_is_latex_safe():
    ns = NumberSet()
    ns.pending("Costo", "correr scripts/run_pipeline.py al 100%")
    assert ns.values["Costo"][0] == "\\pendiente{correr scripts/run\\_pipeline.py al 100\\%}"
    with pytest.raises(ValueError):
        ns.put("Mal_Nombre", "1", "x")


def test_notebook_uses_pipeline():
    nb = json.loads((ROOT / "notebooks" / "01_entrenar_colab.ipynb").read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert "scripts/run_pipeline.py" in src and "resultados_v24.zip" in src
