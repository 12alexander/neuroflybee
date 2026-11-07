"""Cada versión produce su figura: se comprueba que el script corre y genera los PNG."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(script: str):
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_scenario_figures(tmp_path):
    _load("show_scenarios").main(["--out", str(tmp_path)])
    for name in ("v3_escenarios.png", "v3_mapa_previo_vs_real.png"):
        assert (tmp_path / name).stat().st_size > 10_000


def test_v4_sensor_figures(tmp_path):
    _load("show_sensors").main(["--out", str(tmp_path), "--scenarios", "3"])
    for name in ("v4_sensores.png", "v4_panorama_vs_distancia.png"):
        assert (tmp_path / name).stat().st_size > 10_000


def test_v5_episode_figure(tmp_path):
    _load("show_episode").main(["--out", str(tmp_path), "--families", "nominal"])
    assert (tmp_path / "v5_episodios.png").stat().st_size > 10_000


def test_v6_baseline_figures(tmp_path):
    figs = tmp_path / "figures"
    _load("show_baselines").main(["--out", str(figs), "--episodes", "1", "--families", "nominal"])
    assert (figs / "v6_lineas_base.png").stat().st_size > 10_000
    assert (figs / "v6_preliminar.png").stat().st_size > 10_000
    assert (tmp_path / "v6_preliminar.csv").read_text(encoding="utf-8").count("\n") == 1 + 4 * 6


def test_v7_dataset_figures(tmp_path):
    figs = tmp_path / "figures"
    _load("show_dataset").main(["--out", str(figs), "--quick", "--expert-episodes", "1"])
    for name in ("v7_dataset_mosca.png", "v7_dataset_abeja.png"):
        assert (figs / name).stat().st_size > 10_000
    assert (tmp_path / "v7_experto.csv").exists()


def test_v7_generate_data_scaled(tmp_path):
    _load("generate_data").main(["--scale", "0.005", "--out", str(tmp_path / "gen"),
                                 "--summary", str(tmp_path / "summary.csv")])
    assert {p.name for p in (tmp_path / "gen").iterdir()} == {"fly_train.npz", "fly_val.npz", "bee_train.npz",
                                                             "bee_val.npz", "bee_test.npz"}


def test_v8_architecture_figure(tmp_path):
    figs = tmp_path / "figures"
    _load("show_fly_models").main(["--out", str(figs)])
    assert (figs / "v8_redes_mosca.png").stat().st_size > 10_000
    assert "fly_cnn_gru" in (tmp_path / "v8_parametros.csv").read_text(encoding="utf-8")


def test_v12_controller_figures(tmp_path):
    _load("show_controllers").main(["--results", str(tmp_path), "--episodes", "1",
                                    "--checkpoints", str(tmp_path / "sin_pesos")])
    for name in ("v12_arquitectura_hibrida.png", "v12_supervisor.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 10_000


def test_v22_significance_figure(tmp_path):
    csv_path = tmp_path / "v22_significancia.csv"
    csv_path.write_text(
        "grupo,a,b,condicion,escenarios,exito_a,exito_b,diferencia,ic_inf,ic_sup,p,p_holm,h_cohen,significativa,macro\n"
        "principal,Híbrido,A* (clásico),gps_loss,30,0.32,0.03,0.29,0.14,0.44,0.0015,0.030,0.84,1,HybAstar\n"
        "principal,Híbrido,A* + ALV (clásico),total,180,0.57,0.62,-0.04,-0.11,0.02,0.2,1.0,-0.09,0,HybAlv\n",
        encoding="utf-8")
    _load("plot_significance").main(["--csv", str(csv_path), "--out", str(tmp_path / "figures")])
    assert (tmp_path / "figures" / "v22_significancia.png").stat().st_size > 10_000
