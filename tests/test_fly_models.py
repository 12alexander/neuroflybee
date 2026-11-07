"""Pruebas de las redes de la mosca en PyTorch (se omiten si PyTorch no está instalado)."""
import pytest

torch = pytest.importorskip("torch")

from neuroflybee.models.fly import build_fly_model, count_trainable  # noqa: E402
from neuroflybee.models.spec import FLY_MODEL_NAMES, FlyNetConfig, count_parameters  # noqa: E402


@pytest.mark.parametrize("name", FLY_MODEL_NAMES)
def test_output_shape_and_param_count(name):
    cfg = FlyNetConfig()
    model = build_fly_model(name, cfg).eval()
    frames = torch.randint(0, 2, (5, cfg.seq_len, cfg.patch_size, cfg.patch_size)).float()
    goal = torch.randn(5, cfg.goal_dim)
    logits = model(frames, goal)
    assert logits.shape == (5, cfg.n_actions)
    assert torch.allclose(model.predict_proba(frames, goal).sum(-1), torch.ones(5))
    assert count_trainable(model) == count_parameters(name, cfg)     # el conteo analítico es exacto


@pytest.mark.parametrize("name", FLY_MODEL_NAMES)
def test_gradients_reach_every_parameter(name):
    cfg = FlyNetConfig(dropout=0.0)
    model = build_fly_model(name, cfg).train()
    frames = torch.randint(0, 2, (4, cfg.seq_len, cfg.patch_size, cfg.patch_size)).float()
    loss = torch.nn.functional.cross_entropy(model(frames, torch.randn(4, 3)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"sin gradiente: {missing}"


def test_sequence_models_use_past_frames_single_frame_models_do_not():
    cfg = FlyNetConfig(dropout=0.0)
    torch.manual_seed(0)
    a = torch.randint(0, 2, (2, cfg.seq_len, 32, 32)).float()
    b = a.clone()
    b[:, 0] = 1 - b[:, 0]                       # cambia sólo el cuadro más antiguo
    goal = torch.randn(2, 3)
    for name in FLY_MODEL_NAMES:
        model = build_fly_model(name, cfg).eval()
        same = torch.allclose(model(a, goal), model(b, goal))
        assert same == (not model.uses_sequence), name


def test_bad_input_shape():
    with pytest.raises(ValueError):
        build_fly_model("fly_cnn")(torch.zeros(2, 32, 32), torch.zeros(2, 3))
