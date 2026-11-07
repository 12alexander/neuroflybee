# Plugins del visor (v18)

Cada archivo `*.js` de esta carpeta se une en `plugins.js` (lo sirve `scripts/serve.py` y lo
copia `scripts/build_viewer.py`). Así cada integrante agrega su función **sin editar
`index.html`**.

```js
NFB.use({
  name: "mi-plugin",
  init(nfb) {
    nfb.on("afterBuildMaps", ({maps}) => { /* se crearon los mapas */ });
    nfb.on("afterDrawMap", ({ctx, run, t, cur, X, Y, scale}) => { /* dibujar encima del mapa */ });
  },
});
```

`nfb` expone `DATA`, `state`, `scenario()`, `horizon()`, `render()`, `buildMaps()` y `live`
(true cuando lo sirve el servidor). Ejemplo real: `coordenadas.js`.
