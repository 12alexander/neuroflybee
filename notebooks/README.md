# Notebooks

- `01_entrenar_colab.ipynb` (v19, actualizado en v23): obtiene el código y ejecuta el pipeline
  completo (pruebas, datos, validación del sensor, mosca, DAgger, abeja, ablación de la abeja,
  benchmark, significancia, figuras, interpretabilidad, entorno, pesos y números del
  artículo). Descarga `resultados_v24.zip`.
- `02_demo_inferencia.ipynb` (v23): **no entrena**. Carga y verifica los pesos (SHA-256), hace
  volar todos los controladores en escenarios de prueba (`scripts/demo_inference.py`), muestra
  Grad-CAM e Integrated Gradients y arma el visor con los controladores neuronales.
