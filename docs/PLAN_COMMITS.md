# Plan de commits (v1–v24) y mapeo contra el Examen Final MIA-203

Cada commit queda respaldado en una carpeta `neuroflybee_vN` con el código completo
de esa versión. **Desde v3, cada versión produce al menos una figura en
`results/figures/` (prefijo `vN_`) para que el avance sea visible.** La columna *Autor* debe reemplazarse por el integrante real: el
enunciado pide un historial de commits que refleje el trabajo de todos (§3.3).

| v | Autor | Commit | Contenido | Verificación | Rúbrica |
|---|---|---|---|---|---|
| 1 | D | `chore: estructura del repositorio` | Árbol §3.3, config YAML tipada, semillas, esqueleto del paper, plan | `pytest` | C5, C6 |
| 2 | A | `feat(domain): mundo, obstáculos y dron` | Obstáculos (círculo, muro, móvil), mapa de ocupación, cinemática, acciones, misión | Pruebas de distancia y movimiento | C3 |
| 3 | A | `feat(sim): generador de escenarios` | 6 condiciones; semillas train/val/test disjuntas; ruta factible | Prueba de no-fuga | R4 |
| 4 | A | `feat(sim): sensores` | Ojo de mosca 32×32, distancias, panorama 72 bins, GPS, brújula, odometría | Pruebas de sensores | C3 |
| 5 | A | `feat(sim): entorno y vista rápida` | Simulación con viento, móviles y pérdida de GPS; interfaz `Controller`; gráfico estático | PNG de un episodio | C3 |
| 6 | C | `feat(control): líneas base clásicas` | Directo, campos de potencial, A* con replanificación, retorno ALV | Tabla preliminar | R3 |
| 7 | A | `feat(data): experto y datasets` | Experto privilegiado; datasets mosca/abeja (.npz) | Éxito del experto > 90 % | R4 |
| 8 | B | `feat(models): redes de la mosca` | CNN, CNN+GRU, CNN+Transformer, MLP | Formas y parámetros | R1, C1 |
| 9 | B | `feat(train): entrenador e imitación` | Entrenador genérico, imitación, **F1 macro y matriz de confusión** | Curvas de validación | C3, C4 |
| 10 | B | `feat(train): DAgger` | 3 iteraciones DAgger | Mejora vs imitación pura | R4 |
| 11 | B | `feat(models): red de la abeja` | Red siamesa 1D de retorno visual | Error angular vs ALV | R1, R3 |
| 12 | C | `feat(control): controladores neuronales e híbrido` | Mosca, abeja, híbrido, supervisor de seguridad | Episodio por controlador | C3 |
| 13 | D | `feat(viz): visor HTML de la simulación` | Episodios exportados a `datos.js` + visor offline (mapa animado, 4 controladores lado a lado, sensores, decisiones, vetos) | Abrir `results/visor/index.html` | C3, C7 |
| 14 | C | `feat(eval): benchmark y ablaciones` | Todos los controladores × 6 condiciones, media ± d.e.; GRU vs CNN vs Transformer; sin abeja/supervisor/DAgger | Tablas CSV y LaTeX | R3, R4, C4 |
| 15 | C | `feat(xai): interpretabilidad` | Grad-CAM, Integrated Gradients, causas de choque | Figuras | R5 |
| 16 | D | `feat(viz): video de respaldo` | GIF/MP4 grabado del visor para la sustentación | Video por condición | C7 |
| 17 | D | `feat(app): simulación en vivo` | Servidor local: elegir semilla y controlador en el navegador y verlo volar al momento (reutiliza el visor) | Endpoints + visor | C3, C7 |
| 18 | P1 | `refactor: contratos y puntos de extensión` | `WindModel`, `SensorPerturbation`, `Conditions`, rutas y plugins; CODEOWNERS; CI; paper por secciones | Pruebas de contrato | C3, C5 |
| 19 | P5 | `feat(train): pipeline de pesos reales y benchmark neuronal` | Orquestador de Colab con costo (R6), manifiesto de pesos, dependencias congeladas, números del paper generados | Pipeline de punta a punta | C3, C4, C5 |
| 20 | P4 + P5 | `feat(eval): ablación de la red de la abeja` | Siamesa vs sólo vista actual, sin resta y lectura vectorial (24 725 parámetros), en la partición de PRUEBA | Tabla + figura | R4, R1, C4 |
| 21 | P2 | `feat(eval): validación del sensor panorámico 2,5D` | IDF traslacional y rotacional, rumbo, área de captura del ALV (defensa del 2,5D) | Figura + CSV | C2, R6 |
| 22 | P4 | `feat(eval): significancia y AUROC` | IC bootstrap pareado, permutación exacta, Holm, h de Cohen; AUROC macro de la mosca | Tabla | C4 |
| 23 | P5 | `feat(notebook): demostración de inferencia` | Cuaderno que verifica pesos (SHA-256), vuela todos los controladores, Grad-CAM e IG | Notebook + figura | C5, R5, C7 |
| 24 | Todos | `docs: artículo y documentación al cierre` | Artículo con 2,5D, datos de los originales, ablación, significancia, limitaciones y contribuciones reales; preguntas de la defensa | PDF ≤ 12 págs. | C5, C6, R6 |

## Entregable visual por versión

| v | Figura / demo |
|---|---|
| 3 | Las 6 condiciones de evaluación; mapa previo vs realidad |
| 4 | Lo que ve cada sensor: ojo de mosca, anillo de distancias, panorama de la abeja |
| 5 | Recorrido de un episodio sobre el mapa (PNG) y animación corta (GIF) |
| 6 | Las 4 líneas base en un mismo escenario + gráfico preliminar de éxito/choques |
| 7 | Ejemplos del dataset (secuencias del ojo con la acción del experto) y distribución de acciones |
| 8 | Diagrama de las redes de la mosca con dimensiones y parámetros |
| 9 | Curvas de pérdida / F1 macro y matriz de confusión |
| 10 | Mejora por iteración de DAgger |
| 11 | Campo de flechas de retorno a la base: red de la abeja vs ALV vs real |
| 12 | Trayectorias de mosca, abeja e híbrido; dónde intervino el supervisor |
| 13 | **Visor HTML**: mapa animado, 4 controladores lado a lado, ojo, rayos, panorama, decisiones y vetos |
| 14 | Barras de éxito y choques por condición y controlador; gráfico de ablaciones (media ± d.e.) |
| 15 | Grad-CAM sobre el ojo, Integrated Gradients sobre el panorama, mapa de causas de choque |
| 16 | Video GIF/MP4 de respaldo para la sustentación |
| 17 | Simulación en vivo en el navegador |
| 18 | Plugin de coordenadas en el visor; artículo compilado por secciones |
| 19 | Tabla de costo (GPU) y números del artículo generados desde los CSV |
| 20 | Ablación de la abeja: error en prueba y tamaño, con Bee-Nav como referencia |
| 21 | Validación del sensor: IDF, IDF rotacional y campo del ALV |
| 22 | Tabla de comparaciones pareadas (Δ, IC 95 %, p de Holm, h) |
| 23 | Los controladores en un mismo escenario de prueba (cuaderno de demostración) |
| 24 | Artículo completo (11 págs.) y su versión Word |

## Calendario (entrega Sección C: viernes 6-nov-2026, 7 pm) — replanificado el 4-nov

| Día | Versiones | Hito |
|---|---|---|
| Dom 1 – Mar 3 | v1–v19 | Núcleo, simulación en vivo, contratos, pipeline, borrador del artículo; corrida rápida en Colab |
| Mié 4 (madrugada) | v20–v24 | Ablación de la abeja, validación del sensor, significancia, demo, artículo al cierre |
| Mié 4 | Colab (P5) | Corrida rápida de las etapas nuevas y corrida COMPLETA (`QUICK = False`, ~2–3 h) |
| Mié 4 (noche) | — | Números reales en el artículo (`collect_results.py`), diapositivas con los CSV |
| Jue 5 | — | Ensayo de 25 min con las preguntas de [`PREGUNTAS_DEFENSA.md`](PREGUNTAS_DEFENSA.md) |
| Vie 6 | — | Margen y entrega |

### Trabajo futuro (antes v20–v31, no implementado)
Sabotaje del dron en vivo, turbulencia Dryden, datos meteorológicos peruanos y ventana
operativa, sensores calibrados (VL53L1X, IMU EuRoC), prueba de estrés, cuantización INT8/ONNX,
red en el celular y piloto humano. Se mencionan como trabajo futuro en el artículo (§VI), junto
con panoramas 3D (PyBullet) y la prueba de simulación a realidad con los datos CC0 de
Firlefyn et al. (2024).

## Equipo (5 integrantes; el enunciado indica 3–4: confirmarlo con el docente)

| Rol | Carpetas | Versiones | Paper | Exposición |
|---|---|---|---|---|
| P1 Integración y app (propone el proyecto) | `sim/`, `app/`, `viz/`, `contracts/` | v18, v20, v30 | §I, §VI | 5 min |
| P2 Estándar internacional y celular | `disturbances/international/`, `sensors_real/`, `interpret/`, `web/movil/` | v21, v23, v28 | §II, §IV-C | 5 min |
| P3 Datos peruanos y edición del artículo | `weather/`, `disturbances/peru/`, `data/peru/` | v22, v24 | Resumen, §III-A, §V | 6 min |
| P4 Evaluación y diapositivas | `robustness/`, `evaluation/`, `slides/` | v25, v26 | §III-B, §IV-A/B | 5 min |
| P5 Modelos y Colab | `models/`, `training/`, `deploy/`, `notebooks/` | v19, v27, v29 | §III-C/D, §IV-D | 4 min |

Reglas de trabajo: [`CONTRIBUTING.md`](../CONTRIBUTING.md) · dueños: [`CODEOWNERS`](../CODEOWNERS).

## Mapeo contra el enunciado

| Requisito del examen | Dónde se cumple |
|---|---|
| R1 Arquitecturas del curso | v8, v11 (CNN, GRU, Transformer, red siamesa) |
| R2 Originalidad | v3, v12 (aplicación nueva + combinación de arquitecturas) |
| R3 Línea base | v6, v11 (A*, campos de potencial, ALV) |
| R4 Ablación, partición sin fuga, semillas | v3, v10, v14, v20, v22 |
| R5 Interpretación | v15 |
| R6 Costo y responsabilidad | v19 (costo), v21 (validación del sensor) + paper §IV-D y §V |
| §3.1 Paper IEEE con Contribuciones | `paper/` desde v1, completado en v24 |
| §3.2 Diapositivas | v18–v20 (`slides/`) |
| §3.3 Repo, README, semillas, pesos, notebook, historial | v1, v19, v23 |
| §4 F1 macro (clases desbalanceadas) | v9 |
