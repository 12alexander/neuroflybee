# Preguntas probables en la sustentación (v24)

Respuestas cortas, con el número o el archivo que las respalda. Lo marcado **[Colab]** se
completa con la corrida completa (`QUICK = False`): los números salen de `results/` y del
artículo (`paper/generado/numeros.tex`). No usar los números de la corrida rápida.

## Datos

**¿Por qué datos sintéticos?**
Etiquetas exactas (posición, dirección y distancia a casa), escenarios controlados y
reproducibles por semilla, y seis condiciones que en vuelo real serían costosas o peligrosas.
Es la práctica habitual de esta línea: FlyVis se entrenó con videos sintéticos (Sintel) y
Firlefyn et al. principalmente en un bosque simulado. La limitación está declarada (§V).

**¿Con qué datos se entrenaron los originales y con cuáles el suyo?**

| Trabajo | Datos de entrenamiento | Tipo | Métrica |
|---|---|---|---|
| Bee-Nav (Ou et al., 2026) | panoramas del vuelo de aprendizaje del propio dron; objetivos de odometría | real | éxito del regreso |
| Firlefyn et al. (2024) | bosque simulado (35 640 imágenes) + 2 cuadrículas de bosque real | sintético y real | error angular |
| FlyVis (Lappalainen et al., 2024) | videos MPI Sintel en el ojo hexagonal (verificado en su código) | sintético | flujo óptico |
| Nuestra abeja | siluetas de 72 sectores del simulador | sintético | error angular |
| Nuestra mosca | ojo 32×32 + acciones del experto + DAgger | sintético | F1 macro, AUROC, éxito |

**¿Por qué no comparan con un dataset de EE. UU. o Europa (EuRoC, Blackbird, UZH-FPV)?**
Miden estimación de estado y percepción (odometría, SLAM), no regreso visual con panorama ni
evasión con pérdida de GPS: no tienen la tarea ni las etiquetas. Comparar métricas obtenidas
sobre datos distintos no mide la diferencia entre modelos.

**¿Cómo evitan la fuga entre entrenamiento y prueba?**
Partición por semilla: entrenamiento 0–9 999, validación 10 000–10 999, prueba 20 000–20 999.
Cada semilla es un mundo distinto; la prueba nunca se usó para elegir épocas. Hay una prueba
automática que lo verifica.

## Simulador 2,5D

**Su simulador es 2D, ¿no es una limitación grave?**
El movimiento es plano porque el dron vuela a altura fija y el vector a casa es horizontal
(Firlefyn et al. fijan el cuadricóptero a 1,89 m). La percepción no es plana: la abeja ve la
silueta del horizonte con alturas y oclusiones. Se validó (v21, 30 escenarios de prueba):
- la diferencia con la vista de casa crece con la distancia: ρ de Spearman 0,99, monótona en
  el 100 % (IDF de Zeil et al., 2003);
- girar el panorama recupera el rumbo: error mediano 0° a 1 y 2 m, 2,5° a 4 m;
- el ALV apunta a casa con error < 90° en el 74 % del entorno.
Figura `results/figures/v21_validacion_sensor.png`.

**¿Qué NO modela?** Textura, iluminación, inclinación del dron por el viento (la principal
causa de fallo en exterior de Bee-Nav) y la altura del ojo. Trabajo futuro: panoramas 3D con
PyBullet y la prueba con las imágenes reales CC0 de Firlefyn et al.

## Modelo

**¿Por qué una red siamesa?** Cada mundo de prueba es distinto: la red debe comparar lo que ve
con lo que memorizó al despegar, como el modelo de instantáneas de las abejas. El codificador
compartido garantiza que ambas vistas se representen igual, y la resta explícita resalta lo
que cambió. El relleno circular respeta que el panorama da la vuelta.

**¿Cómo saben que funciona?** Ablación en prueba (v20): sólo vista actual **[Colab]** frente a
la siamesa **[Colab]**; sin la resta **[Colab]**; ALV clásico sobre las mismas vistas: 55,4°.

**¿Por qué 172 547 parámetros si Bee-Nav vuela con 868?** El 86 % (≈148 000) está en la capa
lineal de la cabeza. La variante con lectura vectorial tiene 24 725 (7 veces menos): cada
sector vota y la dirección es la suma vectorial, como el ALV con pesos aprendidos. Su error:
**[Colab]**. Bee-Nav usa otra entrada (imagen RGB 192×1800) y se entrena por entorno.

**¿Por qué no usaron directamente la red de Bee-Nav?** Su entrada es una imagen RGB de
192×1800 y se entrena con el vuelo de aprendizaje de cada lugar; nuestra entrada es una
silueta de 72 sectores. Sus pesos no sirven sin reentrenar. Reentrenar su arquitectura con
nuestros datos es trabajo futuro (comparación en igualdad de condiciones).

**¿Por qué CNN+GRU en la mosca?** Los obstáculos móviles necesitan memoria de varios cuadros.
Ablación de arquitecturas (MLP, CNN, CNN+GRU, CNN+Transformer) con 3 semillas: **[Colab]**.

**¿Es una reproducción de FlyVis?** No. FlyVis modela el sistema visual de la mosca
(conectoma, flujo óptico); nuestra mosca es una política de control inspirada en ella. No hay
un trabajo publicado con CNN+GRU + DAgger para drones inspirado en moscas con el cual
comparar: se presenta como aporte propio.

## Entrenamiento y métricas

**¿Qué aporta DAgger?** Corrige el desplazamiento de distribución de la imitación pura;
comparación pareada **[Colab]**.

**¿Por qué no reportan AUROC de la abeja?** La abeja es una regresión de dirección: se mide en
grados. El AUROC macro uno-contra-el-resto se reporta para la mosca, que es un clasificador
de 6 acciones (**[Colab]**), junto al F1 macro.

**¿Las diferencias son significativas?** Análisis pareado por escenario: IC 95 % bootstrap,
permutación exacta (= McNemar), Holm y h de Cohen (v22). Ya medido entre clásicos: A* + ALV
supera a A* en +6 pp de éxito total (IC [+3, +10], p_Holm = 0,017). Híbrido frente a las
líneas base: **[Colab]**. Con 30 escenarios por condición, diferencias de ~10 pp pueden no
ser significativas tras Holm; por eso se reportan intervalo y tamaño de efecto.

## Resultados y relevancia

**¿Dónde gana el híbrido y dónde no?** Tabla por condición y análisis de errores (v14, v15):
**[Colab]**.

**¿Qué miran las redes?** Grad-CAM (mosca) e Integrated Gradients con prueba de fidelidad por
borrado (abeja): `v15_gradcam_mosca.png`, `v15_ig_abeja.png` **[Colab]**.

**¿Cuánto cuesta entrenarlo?** Tabla de costo por etapa (`results/v19_costo.csv`) **[Colab]**.

**¿Funcionaría en un dron real?** No está validado. Bee-Nav muestra que el regreso visual
funciona en drones reales con redes pequeñas; la ruta es: panoramas 3D, prueba con imágenes
reales, cuantización y vuelo físico.

## Equipo

**¿Qué hizo cada integrante?** Tabla de Contribuciones del artículo (`08_contribuciones.tex`),
alineada con lo implementado en v24. Cada integrante debe confirmar sus tareas y poner su
nombre completo.
