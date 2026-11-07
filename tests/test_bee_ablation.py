"""v20: ablación de la red de la abeja (variantes, puntajes, tablas, figura, script y números)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from neuroflybee.config import ProjectConfig
from neuroflybee.data.generate import BeeDataConfig, collect_bee_samples, load_or_collect_bee_split
from neuroflybee.evaluation.bee_ablation import (AblationVariant, latex_table, load_bee_ablation_config,
                                                 read_summary_csv, score_alv, score_predictions, summarize,
                                                 write_scores_csv, write_summary_csv)
from neuroflybee.models.bee_spec import BeeNetConfig, count_bee_parameters
from neuroflybee.reporting.numbers import NumberSet, bee_ablation
from neuroflybee.viz.bee_ablation_plot import plot_bee_ablation

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARAMS = {"completa": 172_547, "vista_unica": 160_259, "sin_diferencia": 166_403, "compacta": 24_725}


@pytest.fixture(scope="module")
def ablation():
    return load_bee_ablation_config(ROOT / "configs" / "bee_ablation.yaml")


@pytest.fixture(scope="module")
def small_test():
    return collect_bee_samples(ProjectConfig(), "test", 3, per_scenario=24)


# ------------------------------------------------------------------ especificación
def test_variant_parameter_counts(ablation):
    counts = {v.name: count_bee_parameters(v.config(BeeNetConfig())) for v in ablation.variants}
    assert counts == EXPECTED_PARAMS
    assert counts["completa"] / counts["compacta"] > 6        # la cabeza plana domina el tamaño


def test_invalid_variants_fail_early(tmp_path):
    with pytest.raises(ValueError):
        BeeNetConfig(fusion="triple")
    with pytest.raises(ValueError):
        BeeNetConfig(head="global")
    bad = tmp_path / "a.yaml"
    bad.write_text("variants:\n  - {name: x, overrides: {head: nada}}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_bee_ablation_config(bad)
    dup = tmp_path / "b.yaml"
    dup.write_text("variants:\n  - {name: x}\n  - {name: x}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_bee_ablation_config(dup)


def test_checkpoint_names_never_overwrite_v11(ablation):
    assert ablation.variant("completa").checkpoint_name(2) == "bee_s2.pt"
    assert ablation.variant("compacta").checkpoint_name(2) == "bee_ablacion_compacta_s2.pt"
    assert ablation.variant("vista_unica").run_name(0) == "bee_ablacion_vista_unica_s0"
    with pytest.raises(KeyError):
        ablation.variant("inexistente")


# ------------------------------------------------------------------ datos de prueba
def test_bee_test_split_is_cached(tmp_path):
    bee = BeeDataConfig(test_scenarios=2, per_scenario=8)
    assert bee.scenarios_for("test") == 2 and bee.scenarios_for("train") == 600
    first = load_or_collect_bee_split(ProjectConfig(), bee, "test", tmp_path)
    assert (tmp_path / "bee_test.npz").exists() and len(first) == 16
    again = load_or_collect_bee_split(ProjectConfig(), bee, "test", tmp_path)
    assert np.array_equal(first.current, again.current)


# ------------------------------------------------------------------ puntajes y salidas
def test_scores_summary_table_and_figure(tmp_path, small_test):
    alv = score_alv(small_test.current, small_test.home, small_test.target)
    assert alv.seed is None and alv.parameters == 0 and 0 <= alv.angle_error_median <= 180
    perfect = AblationVariant("completa", "Siamesa (v11)", reuse_v11=True)
    noisy = AblationVariant("compacta", "Compacta", {"head": "vectorial"})
    scores = [alv, score_predictions(perfect, 0, 172_547, small_test.target, small_test.target),
              score_predictions(perfect, 1, 172_547, small_test.target, small_test.target)]
    rng = np.random.default_rng(0)
    for seed in (0, 1):
        pred = small_test.target + rng.normal(0, 0.3, small_test.target.shape)
        scores.append(score_predictions(noisy, seed, 24_725, pred, small_test.target))
    assert scores[1].angle_error_median == 0 and scores[1].distance_mae == pytest.approx(0, abs=1e-5)
    summary = summarize(scores)
    assert [s.variant for s in summary] == ["alv", "completa", "compacta"]
    assert summary[0].seeds == 0 and summary[1].seeds == 2 and summary[1].median_sd == 0
    assert summary[2].median_mean > 0

    write_scores_csv(tmp_path / "s.csv", scores)
    write_summary_csv(tmp_path / "r.csv", summary)
    back = read_summary_csv(tmp_path / "r.csv")
    assert [r["variante"] for r in back] == ["alv", "completa", "compacta"] and back[0]["error_distancia_m"] == ""
    tex = latex_table(summary, len(small_test))
    assert "\\label{tab:ablacion_abeja}" in tex and "172\\,547" in tex and tex.count("\\\\") >= 4
    assert tex.count("{") == tex.count("}")                        # LaTeX balanceado
    png = plot_bee_ablation(summary, tmp_path / "f.png", len(small_test))
    assert png.stat().st_size > 20_000


def test_numbers_reader(tmp_path, small_test):
    ns = NumberSet()
    bee_ablation(ns, tmp_path)                                     # sin resultados: errores pendientes…
    assert ns.is_pending("AblAbejaCompleta") and ns.is_pending("AblAbejaCompactaTreinta")
    assert ns.values["ParamsAbejaCompacta"][0] == "24\\,725"       # …pero los tamaños salen de la especificación
    assert ns.values["AbejaReduccionParams"][0] == "7"
    full = AblationVariant("completa", "Siamesa (v11)", reuse_v11=True)
    small = AblationVariant("compacta", "Compacta", {"head": "vectorial"})
    scores = [score_alv(small_test.current, small_test.home, small_test.target)]
    scores += [score_predictions(v, s, n, small_test.target, small_test.target)
               for v, n in ((full, 172_547), (small, 24_725)) for s in (0, 1, 2)]
    write_summary_csv(tmp_path / "v20_ablacion_abeja_resumen.csv", summarize(scores))
    ns = NumberSet()
    bee_ablation(ns, tmp_path)
    assert not ns.is_pending("AblAbejaCompleta") and "pm" in ns.values["AblAbejaCompleta"][0]
    assert ns.values["ParamsAbejaCompacta"][0] == "24\\,725"
    assert ns.values["AbejaReduccionParams"][0] == "7"
    assert ns.is_pending("AblAbejaVistaUnica")                     # variante que no se corrió


def test_script_without_networks(tmp_path):
    spec = importlib.util.spec_from_file_location("rba", ROOT / "scripts" / "run_bee_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--sin-redes", "--scale", "0.03", "--data", str(tmp_path / "gen"), "--results", str(tmp_path),
              "--paper", str(tmp_path / "paper")])
    assert (tmp_path / "gen" / "bee_test.npz").exists()
    assert read_summary_csv(tmp_path / "v20_ablacion_abeja_resumen.csv")[0]["variante"] == "alv"
    assert (tmp_path / "figures" / "v20_ablacion_abeja.png").exists()
    assert (tmp_path / "paper" / "tablas" / "v20_ablacion_abeja.tex").exists()


# ------------------------------------------------------------------ con PyTorch
def test_variants_with_torch(tmp_path, ablation, small_test):
    torch = pytest.importorskip("torch")
    from neuroflybee.models.bee import BeeHomingNet, VectorReadoutHead
    from neuroflybee.training.bee import load_bee_checkpoint, predict_bee, train_bee_model
    from neuroflybee.training.config import TrainConfig

    for v in ablation.variants:
        cfg = v.config(BeeNetConfig())
        net = BeeHomingNet(cfg).eval()
        assert sum(p.numel() for p in net.parameters()) == EXPECTED_PARAMS[v.name], v.name
        cur, home = torch.rand(5, 72), torch.rand(5, 72)
        assert net(cur, home).shape == (5, 3)
        if v.name == "vista_unica":                               # la vista de casa no influye
            assert torch.allclose(net(cur, home), net(cur, torch.rand(5, 72)))

    # los nombres de los pesos de v11 no cambian: sus checkpoints siguen cargando
    keys = set(BeeHomingNet().state_dict())
    assert {"head.1.weight", "head.4.weight", "fusion.0.weight"} <= keys

    # la lectura vectorial apunta al sector que vota
    head = VectorReadoutHead(BeeNetConfig(head="vectorial"))
    with torch.no_grad():
        head.score.weight.zero_(); head.score.bias.zero_()
        feats = torch.zeros(1, 64, 18)
        feats[0, 0, 4] = 1.0
        head.score.weight[0, 0, 0] = 1.0
        out = head(feats)
    expected = -np.pi + 4.5 * 2 * np.pi / 18
    assert float(torch.atan2(out[0, 1], out[0, 0])) == pytest.approx(expected, abs=1e-5)

    compact = ablation.variant("compacta")
    tcfg = TrainConfig(epochs=1, batch_size=32, num_workers=0, device="cpu", patience=3)
    train_bee_model(small_test, small_test, compact.config(BeeNetConfig()), tcfg, 0, tmp_path / "ck", tmp_path,
                    run_name=compact.run_name(0), history_tag="v20")
    assert (tmp_path / "v20_historial_bee_ablacion_compacta_s0.json").exists()
    model, _ = load_bee_checkpoint(tmp_path / "ck" / "bee_ablacion_compacta_s0.pt")
    assert model.cfg.head == "vectorial"
    assert predict_bee(model, small_test, batch_size=10).shape == (len(small_test), 3)
