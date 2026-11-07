# Historial de versiones

> Los fragmentos de `changelog.d/` (v18–v24) se unieron aquí en la v24. Desde la siguiente
> versión, cada una vuelve a escribir su fragmento en [`changelog.d/`](changelog.d/).

## v24 — Artículo y documentación al cierre (todos)
- Artículo (11 págs.): defensa del 2,5D y validación del sensor (§III-A, Fig. 3); tabla con
  los datos de entrenamiento de Bee-Nav, Firlefyn et al., FlyVis y este trabajo (§II);
  variantes de la abeja y lectura vectorial (§III-C); dropout, recorte de gradientes, detalles
  de la abeja, AUROC y metodología estadística (§III-D); ablación de la abeja y significancia
  (§IV); figuras de Grad-CAM e IG que aparecen solas al existir; limitaciones y trabajo futuro
  actualizados; contribuciones alineadas con lo implementado.
- 9 referencias nuevas (Bee-Nav y Firlefyn verificadas en la fuente; el resto, a confirmar).
- `reporting/numbers.py`: toda macro declarada en las configuraciones queda definida (en rojo
  si falta) y el artículo siempre compila; parámetros de las variantes desde su especificación.
- `results/v7_datasets.csv`: fila de la partición de prueba de la abeja (6 400 vistas).
- `docs/PREGUNTAS_DEFENSA.md`: preguntas probables con su respuesta y su evidencia.
- `docs/PLAN_COMMITS.md` replanificado (v20–v24; lo demás pasa a trabajo futuro);
  `CODEOWNERS` con los archivos nuevos; fragmentos v18–v24 unidos en `CHANGELOG.md`.
- 2 pruebas nuevas (LaTeX balanceado en las tablas generadas).

## v23 — Demostración de inferencia de punta a punta (P5)
- `notebooks/02_demo_inferencia.ipynb`: sube y verifica los pesos (SHA-256), hace volar los
  controladores en escenarios de prueba, muestra Grad-CAM e Integrated Gradients y arma el
  visor con los controladores neuronales. No entrena.
- `scripts/demo_inference.py`: A*, A* + ALV y, si hay pesos, mosca, abeja e híbrido en el MISMO
  escenario de prueba → `results/figures/v23_demo_<familia>_<semilla>.png` y su CSV de
  desenlaces (éxito, choque, error de retorno, cobertura, latencia). Rechaza semillas que no
  son de prueba.
- `notebooks/01_entrenar_colab.ipynb`: etapas nuevas (validación del sensor, ablación de la
  abeja, significancia), tiempos reales de la corrida rápida y descarga `resultados_v24.zip`.
- `__version__` del paquete sincronizada con `pyproject.toml`.
- 5 pruebas nuevas (cuadernos válidos y sin salidas, scripts referenciados existen, demo); 156 en total.

## v22 — Significancia estadística y AUROC de la mosca (P4)
- `evaluation/significance.py` (sin PyTorch): éxito por escenario (media entre semillas de
  entrenamiento), diferencias pareadas, IC 95 % bootstrap, permutación por cambio de signo
  (exacta = McNemar exacta con éxitos 0/1), corrección de Holm, h de Cohen, CSV y tabla LaTeX.
- `configs/significance.yaml`: 14 comparaciones (híbrido vs líneas base por condición,
  cada módulo neuronal vs su contraparte clásica, clásicos entre sí y ablaciones de v14).
- `scripts/significance.py` + etapa `significancia` del pipeline (CPU, ~1 s).
- `evaluation/metrics.py`: `roc_auc_binary` (U de Mann–Whitney con empates) y
  `roc_auc_ovr_macro`; `training/fly.py` registra `auroc_macro` en cada época.
- Números nuevos: `\nfbSig…Dif`, `\nfbSig…P`, `\nfbSigSignificativas`, `\nfbAuroc…`.
- Medido con los clásicos: A* + ALV vs A*, +6 pp de éxito total, p_Holm = 0,017.
- 9 pruebas nuevas (AUROC verificada contra scikit-learn cuando está instalado).

## v21 — Validación del sensor panorámico 2,5D (P2)
- `evaluation/sensor_validation.py` (sin PyTorch): IDF traslacional (Zeil et al., 2003) limpia
  y con el ruido del sensor, IDF rotacional y recuperación del rumbo, campo y área de captura
  del ALV, correlación de Spearman, radio de llegada; resumen y archivos.
- `viz/sensor_validation_plot.py`: figura de tres paneles con ejes, leyendas y barra de color.
- `scripts/validate_sensor.py` + `configs/sensor_validation.yaml` + etapa `validacion_sensor`
  del pipeline (CPU, ~5 s).
- Resultados medidos (30 escenarios de prueba): IDF monótona en el 100 % (ρ = 0,99), rumbo
  recuperado con error mediano 0° a 2 m, área de captura del ALV 74 %.
- README: defensa del 2,5D (movimiento plano a altura fija, percepción con alturas y oclusiones)
  y lo que el sensor no modela.
- Números nuevos del artículo: `\nfbIdf…`, `\nfbRumboError…`, `\nfbAlvCaptura`, `\nfbAlvErrorCampo`.
- 6 pruebas nuevas (incluye las propiedades que el sensor debe cumplir).

## v20 — Ablación de la red de la abeja en la partición de prueba (P4 + P5)
- `models/bee_spec.py`: la red describe sus variantes (`fusion`: siamesa, sin_diferencia,
  vista_unica; `head`: plana, vectorial) y cuenta los parámetros de cada una; valida los nombres.
- `models/bee.py`: fusión y cabeza intercambiables (`fuse`, `FlatHead`, `VectorReadoutHead`,
  `build_head`). La configuración por defecto y los nombres de los pesos son los de v11: sus
  checkpoints siguen cargando.
- `VectorReadoutHead`: lectura vectorial (cada sector vota; dirección = suma vectorial), 24 725
  parámetros en total frente a 172 547.
- Datos: partición de **prueba** de la abeja (`bee_test.npz`, 100 escenarios, semillas
  20000–20099); `load_or_collect_bee_split` la genera si falta.
- `evaluation/bee_ablation.py` (sin PyTorch): variantes desde `configs/bee_ablation.yaml`,
  puntajes por semilla, resumen media ± d.e., CSV y tabla LaTeX.
- `scripts/run_bee_ablation.py` + etapa `ablacion_abeja` del pipeline; `training/bee.py`
  admite `run_name` (no pisa los pesos de v11) y `predict_bee`.
- `control/homing.py`: `alv_directions` (ALV vectorizado) reutilizado por v11 y v20.
- Figura `v20_ablacion_abeja.png` (error mediano con d.e., parámetros en escala log con las
  redes de Bee-Nav como referencia) y tabla `paper/tablas/v20_ablacion_abeja.tex`.
- Números nuevos del artículo: `\nfbAblAbeja…`, `\nfbParamsAbeja…`, `\nfbAbejaReduccionParams`.
- 8 pruebas nuevas (7 sin PyTorch + 1 con PyTorch).

## v19 — Pipeline de resultados reales y primer borrador del artículo (P5)
- `scripts/run_pipeline.py` (+ `pipeline/`): un comando en Colab corre pruebas, datos, mosca,
  DAgger, abeja, benchmark, figuras, interpretabilidad, entorno, pesos y números; registra el
  log de cada etapa y **se reanuda** si Colab se desconecta. `--quick`, `--dry-run`, `--only`.
- `training/cost.py`: hardware, minutos y horas de GPU por etapa → `results/v19_costo.csv` (R6).
- `scripts/package_weights.py`: manifiesto SHA-256, `.zip` para el enlace y `verify`; subida
  opcional a Hugging Face. `scripts/freeze_env.py`: `requirements-lock.txt` con versiones exactas.
- `reporting/numbers.py` + `scripts/collect_results.py`: 96 números del artículo como macros
  LaTeX con su archivo de origen; lo que falta (resultados neuronales) sale **en rojo**.
- Artículo: primer borrador completo de todas las secciones (8 págs.), 7 referencias nuevas
  del dominio, relevancia replanteada (GPS degradado en quebradas, puentes, canales y centelleo
  ionosférico ecuatorial). Compila con TeX Live mínimo u Overleaf.
- Bloque de 5 autores en formato IEEE (3 + 2: nombre, programa, FIIS–UNI, Lima, correo).
- `scripts/paper_to_word.py`: versión Word con formato IEEE (carta, Times 10 pt, dos columnas,
  figuras anchas a todo el ancho, ecuaciones nativas, referencias IEEE numeradas, pendientes
  en rojo) para revisar y comentar en equipo; el fuente oficial sigue siendo LaTeX.
- Notebook de Colab simplificado al pipeline; figura `v19_pipeline.png`.
- 15 pruebas nuevas (129 en total).

## v18 — Contratos y puntos de extensión (P1)
- `contracts/`: `WindModel`, `SensorPerturbation`, `Conditions`, catálogo de condiciones y
  pruebas de conformidad reutilizables. El entorno las recibe inyectadas (inversión de
  dependencias); sin condiciones se comporta exactamente como en v17 (verificado).
- Las perturbaciones de sensores usan un flujo aleatorio propio: agregar ruido no cambia las
  ráfagas (comparación controlada para la prueba de estrés).
- Servidor: tabla de rutas y extensiones en `app/extensions/` (`/api/estado` de ejemplo).
- Visor: API `window.NFB` con eventos y plugins en `viz/viewer/plugins/` (`coordenadas.js`).
- Carpetas con dueño para v19–v31, `CODEOWNERS`, GitHub Actions y `changelog.d/`.
- Artículo dividido en `paper/secciones/` (un archivo por dueño); compila con IEEEtran incluido.
- 10 pruebas nuevas (114 en total).

## v17 — Simulación en vivo en el navegador
- `app/service.py`: `SimulationService` (capa de aplicación, sin HTTP): catálogo,
  validación (`ValidationError`: condición, semilla sólo del split de PRUEBA, 1–4
  controladores conocidos) y simulación en el formato del visor.
- `app/http_server.py`: adaptador HTTP con la biblioteca estándar (`ThreadingHTTPServer`):
  `/`, `/datos.js`, `/api/catalogo`, `/api/simular`; errores 400/404 en JSON.
- Visor: barra **● EN VIVO** que sólo aparece cuando lo sirve el servidor; abierto como
  archivo sigue funcionando igual que en v13.
- `scripts/serve.py` y `make serve`; con pesos entrenados agrega mosca, abeja e híbrido.
- 10 pruebas nuevas (104 en total), incluido el servidor real en un puerto libre.

## v16 — Video de respaldo
- `scripts/record_video.py`: abre el visor en Chromium sin ventana (Playwright), recorre
  tres escenas con 4 controladores lado a lado (obstáculos móviles, pérdida de GPS,
  combinado) con portada y rótulos, y genera `results/video/v16_demo.mp4` (≈ 30 s, H.264),
  `results/video/v16_demo.gif` y `results/figures/v16_portada.png`.
- Tras entrenar, el mismo comando graba mosca, abeja e híbrido (celda nueva en Colab).
- `requirements.txt`: `imageio-ffmpeg` y `playwright` (opcionales).
- 2 pruebas nuevas (94 en total), incluida una grabación corta real.

## v15 — Interpretabilidad y análisis de errores
- `evaluation/error_analysis.py` (sin PyTorch): clasifica cada episodio por su causa
  (choque con obstáculo previo / no previsto / móvil / borde, deriva sin GPS, atascado, no
  encuentra la base, aterriza sin inspeccionar todo).
- `interpret/gradcam.py`: Grad-CAM por cuadro sobre la última convolución de la mosca.
- `interpret/integrated_gradients.py`: IG sobre el panorama de la abeja (referencia = vista
  memorizada) y prueba de fidelidad por borrado.
- `scripts/explain.py`: `v15_analisis_errores.png` + `results/v15_errores.csv` (medido con
  los clásicos) y, con pesos, `v15_gradcam_mosca.png`, `v15_ig_abeja.png`, `results/v15_fidelidad_abeja.csv`.
- 3 pruebas nuevas sin PyTorch (92 en total) + 1 con PyTorch (formas, rango, completitud de IG).

## v14 — Benchmark y ablaciones
- `evaluation/benchmark.py` (sin PyTorch): especificación declarativa (`configs/benchmark.yaml`),
  ejecución por variante (modelo, semilla de entrenamiento, con/sin DAgger), media ± d.e.
  entre semillas, IC 95 % de Wilson, CSV por episodio, tablas CSV y LaTeX.
- `control/neural_registry.py`: `build_neural_controller` construye cualquier variante
  (arquitectura, semilla, DAgger) con caché de modelos.
- `scripts/run_benchmark.py`: tabla principal (6 condiciones × 30 escenarios de prueba),
  4 ablaciones (se omiten las que no tienen pesos), figuras y tablas para el artículo.
- Resultados clásicos medidos (180 episodios por controlador): A* 56 %, A*+ALV 62 %,
  campos de potencial 48 %, A*+supervisor 58 %, directo 3 %.
- Visual: `v14_benchmark.png`, `v14_ablaciones.png`; `paper/tablas/*.tex` en la sección de Resultados.
- 4 pruebas nuevas (89 en total); celda de benchmark en el notebook de Colab.

## v13 — Visor HTML de la simulación
- `viz/export.py`: exporta cada paso de cada vuelo (posición real y estimada, acción,
  módulo que decidió, fase, GPS, vetos, probabilidades de la red y los tres sensores; el ojo
  de la mosca se empaqueta en hexadecimal) a `results/visor/datos.js`.
- `viz/viewer/index.html`: visor offline (HTML + JavaScript, sin dependencias): selector de
  escenario y controlador, vista de 4 controladores lado a lado, reproducción con velocidad
  y barra de tiempo, paneles de decisión, probabilidades, ojo, radar de rayos y panorama.
- `scripts/build_viewer.py`: simula y arma el visor; agrega los controladores neuronales
  automáticamente si hay pesos.
- Plan reordenado: v13 visor, v14 benchmark + ablaciones, v17 simulación en vivo.
- 2 pruebas nuevas (85 en total); visor verificado en Chromium sin errores de consola.

## v12 — Controladores neuronales, híbrido y supervisor
- `control/safety.py`: supervisor de seguridad (decorador) que predice la posición tras
  cada maniobra y la compara con los puntos detectados por los rayos; veta y elige la
  maniobra segura más parecida o mantiene la posición. No usa redes ni GPS.
- `control/neural_registry.py`: registra `fly`, `bee`, `hybrid` e `hybrid_no_supervisor`
  cuando hay pesos (prefiere la última iteración de DAgger).
- `control/classical.py`: ablaciones `direct_supervised` y `astar_supervised`.
- Visual: `scripts/show_controllers.py` genera `v12_arquitectura_hibrida.png`,
  `v12_supervisor.png` + `results/v12_supervisor.csv` y, con pesos, `v12_controladores.png`.
- 4 pruebas nuevas sin PyTorch (83 en total) + 1 con PyTorch que hace volar los 4
  controladores neuronales.

## v11 — Red de la abeja
- `models/bee.py` (PyTorch): red siamesa 1D con relleno circular (vista actual vs vista
  memorizada → dirección y distancia a la base) y su pérdida (coseno + Smooth-L1).
- `models/bee_spec.py` (sin PyTorch): hiperparámetros (`configs/bee_model.yaml`) y conteo
  de parámetros (172 547).
- `training/bee.py`: conjunto de pares con aumento por error de brújula (±1 bin), entrenamiento
  con el mismo `Trainer` de la mosca (minimiza el error angular mediano), checkpoints.
- `inference/bee_homing.py`: la red como `HomingProvider`; el navegador la usa igual que al ALV.
- `evaluation/metrics.py`: error angular y reporte de retorno.
- `configs/train_bee.yaml`, `scripts/train_bee.py`, `scripts/show_bee_homing.py`; celda en Colab.
- Visual: `v11_retorno_abeja.png` (campo de flechas, error vs distancia, distribución) y
  `results/v11_abeja_validacion.csv`. Línea base ALV medida: 51.4° de error mediano.
- 3 pruebas nuevas sin PyTorch (79 en total) + 1 con PyTorch.

## v10 — DAgger
- `training/dagger.py`: 3 iteraciones (β 0.5 → 0.25 → 0), datos agregados, reentrenamiento
  desde los pesos anteriores y evaluación en lazo cerrado en validación por iteración.
- `inference/fly_runner.py`: ejecuta la red paso a paso con memoria de cuadros (la usarán
  también el controlador y el dashboard).
- `control/neural.py`: `FlySteering`, la mosca detrás de la interfaz `Steering`.
- `evaluation/closed_loop.py` (sin PyTorch): vuelo de episodios y resumen de métricas para
  cualquier controlador (base del benchmark de la v13).
- `data/generate.py`: con DAgger el estudiante se consulta en todos los pasos.
- `training/fly.py`: continuar desde pesos previos y nombrar las corridas.
- `configs/dagger.yaml`, `scripts/train_dagger.py`, `scripts/plot_dagger.py`; celda nueva en el notebook de Colab.
- Visual: `v10_dagger_esquema.png`; la mejora por iteración se genera al entrenar.
- 4 pruebas nuevas sin PyTorch (76 en total) + 1 corrida mínima de DAgger con PyTorch.

## v9 — Entrenador e imitación de la mosca
- `training/trainer.py`: entrenador genérico (modelo, optimizador, pérdida y métricas
  inyectados; parada temprana; checkpoint del mejor; historial JSON con tiempos y dispositivo).
- `training/fly.py`: clonación de comportamiento con entropía cruzada ponderada por clase,
  AdamW + coseno, selección por F1 macro de validación y matriz de confusión del mejor modelo.
- `training/torch_data.py`: ventanas temporales que nunca cruzan episodios; cargadores con semilla.
- `training/checkpoints.py`: guardar/cargar pesos con su configuración.
- `evaluation/metrics.py` y `evaluation/reports.py` (sin PyTorch): F1 macro, matriz de
  confusión, pesos de clase, ventanas temporales, resúmenes media ± d.e.
- `configs/train_fly.yaml`, `scripts/train_fly.py`, `scripts/plot_fly_training.py`.
- `notebooks/01_entrenar_colab.ipynb`: todo el flujo en Colab con GPU.
- Visual: `v9_pesos_clase.png`; las curvas y la matriz de confusión se generan al entrenar.
- 6 pruebas nuevas sin PyTorch (72 en total) + 1 de entrenamiento real con PyTorch.

## v8 — Redes de la mosca
- `models/fly.py` (PyTorch): codificador CNN (3 bloques conv3×3 + BN + ReLU + MaxPool),
  cabeza de decisión común y cuatro redes con la misma firma — MLP, CNN, CNN+GRU
  (propuesta) y CNN+Transformer — construidas con un método plantilla y un registro.
- `models/spec.py` (sin PyTorch): hiperparámetros (`configs/fly_models.yaml`), conteo
  analítico de parámetros y descripción de bloques con las formas de los tensores.
- Visual: `scripts/show_fly_models.py` genera `v8_redes_mosca.png` y `results/v8_parametros.csv`.
- Pruebas: 8 sin PyTorch (66 en total) + 10 con PyTorch que se ejecutan en Colab o en
  un equipo con PyTorch (formas, gradientes, uso de la secuencia, conteo exacto).

## v7 — Experto y conjuntos de datos
- `data/expert.py`: experto privilegiado (A* sobre el mapa real + verificación de los
  obstáculos móviles en un horizonte de 4 pasos). Éxito del 100 % en validación.
- `data/generate.py`: recolección de datos de imitación para la mosca (objetivo estimado
  como en la prueba, etiqueta del experto, ruido de acción DART, gancho para DAgger) y de
  pares de panoramas para la abeja; formatos `.npz` con carga y concatenación.
- `configs/data.yaml` y `scripts/generate_data.py`: tamaños y generación reproducible
  (76 077 + 10 593 muestras de la mosca; 38 400 + 6 400 de la abeja).
- Visual: `scripts/show_dataset.py` genera `v7_dataset_mosca.png` y `v7_dataset_abeja.png`;
  tablas `results/v7_datasets.csv` y `results/v7_experto.csv`.
- 7 pruebas nuevas (58 en total).

## v6 — Líneas base clásicas
- `control/estimation.py`: estimador de posición (filtro GPS + integración de trayectoria).
- `control/navigation.py`: navegador de misión (memoria de la ruta, fases, aterrizaje),
  con un punto de extensión `HomingProvider` para el retorno visual.
- `control/homing.py`: retorno visual clásico ALV con llegada por diferencia de imagen.
- `control/steering.py`: estrategias de control local intercambiables — directo,
  campos de potencial y A* con replanificación sobre obstáculos detectados.
- `control/composite.py` + `control/registry.py` + `control/classical.py`: los
  controladores se arman por composición y se registran por nombre (abierto/cerrado).
- `sim/grid.py`: A* 8-conexo con heurística octil.
- `domain/mission.py`: criterio de éxito de misión.
- Visual: `scripts/show_baselines.py` genera `v6_lineas_base.png` (trayectorias) y
  `v6_preliminar.png` + `results/v6_preliminar.csv` (éxito y choques por condición).
- 9 pruebas nuevas (51 en total).

## v5 — Entorno de simulación y primer vuelo
- `sim/env.py`: bucle de simulación con viento y ráfagas, colisiones, pérdida de GPS al
  completar la inspección, aterrizaje y límite de pasos; agrupa los sensores (`SensorSuite`).
- `sim/observation.py`: lo único que ve un controlador (sin posición real) y el mapa previo.
- `control/base.py`: interfaz `Controller` y `Decision` (acción, fase, módulo que decidió,
  probabilidades, objetivo y posición estimada). Todo lo que sigue depende de esta interfaz.
- `sim/rollout.py`: ejecuta un episodio y registra cada paso (posición real, estimada, latencia).
- Visual: `scripts/show_episode.py` genera `v5_episodios.png` con un controlador de
  demostración: choques sin evasión y deriva sin GPS.
- 6 pruebas nuevas (42 en total).

## v4 — Sensores
- `sim/sensors.py`: ojo de la mosca (parche egocéntrico de ocupación 32×32 con ruido
  y píxeles muertos opcionales), anillo de 32 distancias, panorama de la abeja
  (elevación del horizonte en 72 direcciones, alineado con brújula), GPS (puede
  faltar), brújula y odometría (no percibe el viento → deriva).
- `config.py`: nuevo parámetro `sensors.panorama_noise`.
- Visual: `scripts/show_sensors.py` genera `v4_sensores.png` (mapa + ojo + distancias
  + panorama) y `v4_panorama_vs_distancia.png` (la vista se aleja de la memorizada al
  alejarse de la base: la señal que usará la abeja para volver).
- `viz/sensor_plot.py`: dibujo reutilizable de sensores (lo usará el dashboard).
- 7 pruebas nuevas (36 en total).

## v3 — Generador de escenarios
- `sim/scenarios.py`: 6 condiciones (nominal, obstáculos no previstos, móviles, viento,
  pérdida de GPS al regresar, combinada) definidas de forma declarativa en `FAMILIES`.
- Particiones por semilla: entrenamiento 0–9 999, validación 10 000–10 999,
  prueba 20 000–20 999 (sin solapamiento → sin fuga de datos).
- Cada escenario se verifica con una búsqueda en anchura (`sim/grid.py`): siempre hay
  ruta libre de la base a todos los puntos de inspección.
- Visual: `scripts/show_scenarios.py` genera `results/figures/v3_escenarios.png` y
  `results/figures/v3_mapa_previo_vs_real.png`.
- `viz/`: estilo común y dibujo de escenarios reutilizable en las versiones siguientes.
- 13 pruebas nuevas (29 en total), incluida una que verifica que las figuras se generan.

## v2 — Dominio: mundo, obstáculos y dron
- `domain/geometry.py`: ángulos, rotaciones y cambio al marco del dron (vectorizado).
- `domain/world.py`: protocolo `Obstacle` común a tres tipos (roca/árbol circular,
  muro rectangular y obstáculo móvil que va y viene); distancia con signo; mapa de
  ocupación (estático en caché + móviles por instante); conversión celda ↔ punto.
- `domain/drone.py`: 6 acciones discretas (avanzar, girar, giro cerrado, mantener
  posición) y cinemática tipo uniciclo con viento opcional.
- `domain/mission.py`: fases (INSPECT, RETURN, LANDED) y seguimiento de puntos
  visitados con la posición real (sólo para evaluar).
- `tests/test_domain.py`: 13 pruebas (16 en total con las de v1).

## v1 — Estructura del repositorio
- Árbol de carpetas según §3.3 del enunciado (`data/`, `src/`, `notebooks/`, `configs/`, `results/`).
- Configuración tipada (`src/neuroflybee/config.py`) cargada desde `configs/default.yaml`.
- Control centralizado de semillas (`src/neuroflybee/utils/seeding.py`).
- Esqueleto del artículo IEEE (`paper/`) y plan de 20 commits (`docs/PLAN_COMMITS.md`).
- Primeras pruebas (`tests/test_config.py`).
