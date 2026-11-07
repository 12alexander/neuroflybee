import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_video_requires_viewer(tmp_path):
    with pytest.raises(SystemExit):
        _load("record_video").main(["--viewer", str(tmp_path / "index.html")])


def test_record_short_video(tmp_path):
    pytest.importorskip("playwright")
    pytest.importorskip("imageio_ffmpeg")
    _load("build_viewer").main(["--families", "gps_loss", "--controllers", "astar", "--out", str(tmp_path / "visor"),
                                "--checkpoints", str(tmp_path / "sin_pesos")])
    try:
        _load("record_video").main(["--viewer", str(tmp_path / "visor" / "index.html"), "--stride", "60",
                                    "--hold", "0.2", "--title", "0.2", "--size", "800x512",
                                    "--out", str(tmp_path / "video"), "--poster", str(tmp_path / "p.png")])
    except Exception as exc:                     # navegador de Playwright no instalado
        if "Executable doesn't exist" in str(exc):
            pytest.skip("Chromium de Playwright no instalado")
        raise
    assert (tmp_path / "video" / "v16_demo.mp4").stat().st_size > 1_000
    assert (tmp_path / "video" / "v16_demo.gif").exists() and (tmp_path / "p.png").exists()
