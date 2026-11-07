"""Grad-CAM (Selvaraju et al., 2017) para las redes de la mosca con codificador CNN.

Responde: ¿qué zona del ojo hizo que la red eligiera esta maniobra? Se calcula sobre la
última convolución del codificador, para cada cuadro de la secuencia, y se reescala a
32×32 para superponerlo al ojo.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ..models.fly import FlyPolicyNet


def grad_cam(model: FlyPolicyNet, frames: torch.Tensor, goal: torch.Tensor,
             action: int | None = None) -> tuple[torch.Tensor, int, torch.Tensor]:
    """frames (T, S, S), goal (3,). Devuelve (mapas (T, S, S) en [0, 1], acción explicada, probabilidades)."""
    if not hasattr(model, "encoder"):
        raise TypeError("Grad-CAM requiere un modelo con codificador convolucional (no el MLP)")
    model.eval()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = model.encoder.last_conv

    def keep(module, inputs, output):
        activations.append(output)
        output.register_hook(gradients.append)          # gradiente respecto de la activación

    handle = layer.register_forward_hook(keep)
    try:
        x = frames[None].float().requires_grad_(False)
        logits = model(x, goal[None].float())
        probs = torch.softmax(logits, -1)[0].detach()
        chosen = int(probs.argmax()) if action is None else int(action)
        model.zero_grad(set_to_none=True)
        logits[0, chosen].backward()
    finally:
        handle.remove()
    act, grad = activations[-1], gradients[-1]            # (N, C, h, w); N = T (secuencia) o 1
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=frames.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    if cam.shape[0] == 1 and frames.shape[0] > 1:           # modelos de un cuadro: sólo el último
        cam = torch.cat([torch.zeros(frames.shape[0] - 1, *cam.shape[1:]), cam.cpu()])
    return cam.detach().cpu(), chosen, probs.cpu()
