"""Video de respaldo para la sustentación (v16), grabado del visor HTML.

Abre `results/visor/index.html` en un navegador sin ventana (Playwright + Chromium),
recorre paso a paso las condiciones más ilustrativas con 4 controladores lado a lado y
arma el video. Si el visor tiene controladores neuronales (tras entrenar), el video los
muestra automáticamente.

Requisitos (una vez):
    pip install playwright imageio imageio-ffmpeg
    python -m playwright install chromium
Uso:
    python scripts/record_video.py                    # MP4 + GIF + portada
    python scripts/record_video.py --stride 3 --fps 20
Genera:
    results/video/v16_demo.mp4        ≈ 30 s, 1440×896
    results/video/v16_demo.gif        versión liviana (una condición) para diapositivas
    results/figures/v16_portada.png   un cuadro para el README y las diapositivas
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

SCENES = [  # (condición, rótulo)
    ("dynamic", "1 · Obstáculos móviles: ¿quién los esquiva?"),
    ("gps_loss", "2 · Pérdida de GPS al regresar: ¿quién vuelve a la base?"),
    ("combined", "3 · Todo junto: obstáculos no previstos, móviles, viento y sin GPS"),
]
CAPTION_JS = """
(text) => {
  let el = document.getElementById('nfb-caption');
  if (!el) {
    el = document.createElement('div'); el.id = 'nfb-caption';
    el.style.cssText = 'position:fixed;top:58px;left:50%;transform:translateX(-50%);z-index:9;' +
      'background:rgba(15,20,25,.92);color:#fff;padding:8px 18px;border-radius:8px;' +
      'font:600 18px Segoe UI,system-ui,sans-serif;border:1px solid #2ecc71';
    document.body.appendChild(el);
  }
  el.textContent = text; el.style.display = text ? 'block' : 'none';
}
"""
TITLE_JS = """
(show) => {
  let el = document.getElementById('nfb-title');
  if (!el) {
    el = document.createElement('div'); el.id = 'nfb-title';
    el.style.cssText = 'position:fixed;inset:0;z-index:10;display:flex;flex-direction:column;' +
      'align-items:center;justify-content:center;background:rgba(10,14,18,.93);color:#fff;' +
      'font-family:Segoe UI,system-ui,sans-serif;text-align:center';
    el.innerHTML = '<div style="font-size:44px;font-weight:700">NeuroFly-Bee Perú</div>' +
      '<div style="font-size:22px;margin-top:10px;color:#f1c40f">Navegación autónoma bioinspirada en ' +
      '<b>abejas</b> y <b style="color:#e74c3c">moscas</b></div>' +
      '<div style="font-size:16px;margin-top:22px;color:#8b9bab">MIA-203 Redes Neuronales y Aprendizaje ' +
      'Profundo · UNI · simulación (escenarios de prueba)</div>';
    document.body.appendChild(el);
  }
  el.style.display = show ? 'flex' : 'none';
}
"""


def grab(page) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB"))


def scene_indices(page) -> dict[str, int]:
    return page.evaluate("Object.fromEntries(DATA.scenarios.map((s, i) => [s.family, i]))")


def record(viewer: Path, stride: int, hold_s: float, fps: int, title_s: float,
           width: int, height: int) -> tuple[list[np.ndarray], dict[str, tuple[int, int]]]:
    from playwright.sync_api import sync_playwright

    frames: list[np.ndarray] = []
    spans: dict[str, tuple[int, int]] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(viewer.resolve().as_uri())
        page.wait_for_function("window.NFB_DATA && document.querySelector('#maps canvas')")
        index = scene_indices(page)
        first = True
        for family, caption in SCENES:
            if family not in index:
                print(f"  (el visor no tiene la condición '{family}', se omite)")
                continue
            page.evaluate(f"state.sc = {index[family]}; state.run = 0; state.quad = true; state.t = 0; "
                          "$('scenario').value = state.sc; buildMaps();")
            page.evaluate(CAPTION_JS, caption)
            if first:                                   # portada
                page.evaluate(TITLE_JS, True)
                frames += [grab(page)] * int(title_s * fps)
                page.evaluate(TITLE_JS, False)
                first = False
            horizon = page.evaluate("horizon()")
            start = len(frames)
            for t in list(range(0, horizon + 1, stride)) + [horizon]:
                page.evaluate(f"state.t = {t}; render();")
                frames.append(grab(page))
            frames += [frames[-1]] * int(hold_s * fps)
            spans[family] = (start, len(frames))
            print(f"  {caption}: {horizon} pasos → {len(frames) - start} cuadros")
        browser.close()
        if errors:
            raise RuntimeError(f"Errores de JavaScript en el visor: {errors}")
    return frames, spans


def write_outputs(frames, spans, out_dir: Path, poster: Path, fps: int, gif_scene: str, gif_width: int) -> None:
    import imageio.v2 as imageio

    out_dir.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_dir / "v16_demo.mp4", fps=fps, codec="libx264", quality=7,
                            macro_block_size=8, ffmpeg_log_level="error") as w:
        for f in frames:
            w.append_data(f)
    a, b = spans.get(gif_scene, (0, len(frames)))
    gif = [Image.fromarray(f) for f in frames[a:b:2]]
    h = int(gif[0].height * gif_width / gif[0].width)
    gif = [g.resize((gif_width, h), Image.LANCZOS).quantize(colors=128) for g in gif]
    gif[0].save(out_dir / "v16_demo.gif", save_all=True, append_images=gif[1:], duration=int(2000 / fps), loop=0,
                optimize=True)
    mid = spans.get("dynamic", (0, len(frames)))
    poster.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frames[(mid[0] + mid[1]) // 2]).save(poster)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--viewer", default=str(ROOT / "results" / "visor" / "index.html"))
    parser.add_argument("--stride", type=int, default=2, help="pasos de simulación por cuadro")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--hold", type=float, default=1.5, help="segundos de pausa al final de cada escena")
    parser.add_argument("--title", type=float, default=2.5, help="segundos de portada")
    parser.add_argument("--size", default="1440x896")
    parser.add_argument("--gif-scene", default="gps_loss")
    parser.add_argument("--gif-width", type=int, default=900)
    parser.add_argument("--out", default=str(ROOT / "results" / "video"))
    parser.add_argument("--poster", default=str(ROOT / "results" / "figures" / "v16_portada.png"))
    args = parser.parse_args(argv)

    viewer = Path(args.viewer)
    if not (viewer.parent / "datos.js").exists():
        raise SystemExit("Falta el visor: ejecute primero  python scripts/build_viewer.py")
    width, height = (int(v) for v in args.size.split("x"))
    frames, spans = record(viewer, args.stride, args.hold, args.fps, args.title, width, height)
    write_outputs(frames, spans, Path(args.out), Path(args.poster), args.fps, args.gif_scene, args.gif_width)
    mp4 = Path(args.out) / "v16_demo.mp4"
    print(f"Video: {mp4} ({mp4.stat().st_size / 1e6:.1f} MB, {len(frames) / args.fps:.0f} s) · "
          f"GIF: {(Path(args.out) / 'v16_demo.gif').stat().st_size / 1e6:.1f} MB · portada: {args.poster}")


if __name__ == "__main__":
    main()
