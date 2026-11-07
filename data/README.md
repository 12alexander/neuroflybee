# Datos

Este proyecto **no usa datos descargados**: todos los datos se generan de forma
procedural y reproducible con el simulador (`src/neuroflybee/sim`). Así se evita la
fuga de datos y cualquier problema de licencia.

| Conjunto | Cómo se genera | Semillas de escenario |
|---|---|---|
| Entrenamiento | `python -m neuroflybee.cli generate-data` | 0 – 9 999 |
| Validación | ídem | 10 000 – 10 999 |
| Prueba (benchmark) | se genera al evaluar | 20 000 – 20 999 |

Los rangos de semillas son disjuntos y se verifican en `tests/test_scenarios.py`.
Los archivos generados (`data/generated/*.npz`) no se suben al repositorio.

## Cómo generarlos (v7)

```bash
python scripts/generate_data.py            # tamaños de configs/data.yaml (≈ 3 min en CPU)
python scripts/generate_data.py --scale 0.1  # versión reducida para probar
```

| Archivo | Contenido |
|---|---|
| `generated/fly_{train,val}.npz` | ojo de la mosca (uint8 32×32), objetivo relativo, acción del experto, episodio, paso, condición |
| `generated/bee_{train,val}.npz` | panorama actual, panorama memorizado en la base, dirección y log-distancia a la base |

El resumen de tamaños queda versionado en `results/v7_datasets.csv`.
