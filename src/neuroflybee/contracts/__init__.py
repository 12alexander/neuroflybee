"""Contratos del proyecto (v18): las interfaces que permiten a cada integrante trabajar en su
propia carpeta sin tocar el núcleo.

- `WindModel`: cómo sopla el viento (Dryden internacional, viento peruano calibrado…).
- `SensorPerturbation`: cómo se degradan los sensores (ruido, latencia, rayos que fallan…).
- `Conditions`: un paquete de viento + perturbaciones que se inyecta al entorno.
- `conditions_registry`: catálogo con nombre para usarlas desde configs y scripts.
"""
from .disturbances import Conditions, SensorPerturbation, WindModel
from . import conditions_registry

__all__ = ["Conditions", "SensorPerturbation", "WindModel", "conditions_registry"]
