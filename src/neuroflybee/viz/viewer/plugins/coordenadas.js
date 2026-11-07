// Plugin de ejemplo (v18): muestra las coordenadas en metros bajo el cursor.
// Base para el sabotaje (v20): convertir un clic en una posición del mundo.
NFB.use({
  name: "coordenadas",
  init(nfb) {
    const tip = document.createElement("div");
    tip.style.cssText = "position:fixed;pointer-events:none;background:#111c;color:#fff;font:12px sans-serif;" +
                        "padding:2px 6px;border-radius:4px;display:none;z-index:10";
    document.body.appendChild(tip);
    nfb.toWorld = (canvas, ev) => {            // utilidad compartida para otros plugins
      const r = canvas.getBoundingClientRect(), w = nfb.scenario().world;
      return [(ev.clientX - r.left) / r.width * w.width, (1 - (ev.clientY - r.top) / r.height) * w.height];
    };
    nfb.on("afterBuildMaps", ({maps}) => maps.querySelectorAll("canvas").forEach(c => {
      c.addEventListener("mousemove", ev => {
        const [x, y] = nfb.toWorld(c, ev);
        tip.textContent = `x ${x.toFixed(1)} m · y ${y.toFixed(1)} m`;
        tip.style.left = ev.clientX + 14 + "px"; tip.style.top = ev.clientY + 10 + "px"; tip.style.display = "block";
      });
      c.addEventListener("mouseleave", () => { tip.style.display = "none"; });
    }));
  },
});
