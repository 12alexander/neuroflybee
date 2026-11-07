import importlib.util
import json
from pathlib import Path

import numpy as np

from neuroflybee.viz.export import pack_bits, unpack_bits

ROOT = Path(__file__).resolve().parents[1]


def test_eye_bits_roundtrip():
    patch = (np.random.default_rng(0).random((32, 32)) > 0.7).astype(np.float32)
    assert np.array_equal(unpack_bits(pack_bits(patch), 32), patch.astype(np.uint8))
    assert len(pack_bits(patch)) == 256


def test_build_viewer_writes_html_and_data(tmp_path):
    spec = importlib.util.spec_from_file_location("b", ROOT / "scripts" / "build_viewer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--families", "dynamic", "--controllers", "astar,direct_supervised", "--out", str(tmp_path),
              "--checkpoints", str(tmp_path / "sin_pesos")])
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert '<script src="datos.js">' in html
    text = (tmp_path / "datos.js").read_text(encoding="utf-8")
    data = json.loads(text[len("window.NFB_DATA = "):].rstrip().rstrip(";"))
    sc = data["scenarios"][0]
    assert [r["controller"] for r in sc["runs"]] == ["astar", "direct_supervised"]
    assert sc["world"]["dynamic"] and len(sc["home_panorama"]) == 72
    step = sc["runs"][0]["steps"][5]
    assert {"p", "h", "a", "s", "g", "v", "eye", "rg", "pn"} <= set(step)
    assert len(step["rg"]) == 32 and len(step["pn"]) == 72
