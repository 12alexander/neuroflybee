"""Controladores clásicos (líneas base) registrados en el catálogo."""
from __future__ import annotations

from .composite import NavigationController
from .homing import AverageLandmarkVectorHoming
from .navigation import MissionNavigator
from .registry import register
from .safety import SafetySupervisor
from .steering import AStarSteering, DirectSteering, PotentialFieldSteering


@register("direct", "Línea base convencional: recta al objetivo, sin evasión")
def direct() -> NavigationController:
    return NavigationController("direct", MissionNavigator(), DirectSteering())


@register("potential_field", "Reactivo clásico (campos de potencial)")
def potential_field() -> NavigationController:
    return NavigationController("potential_field", MissionNavigator(), PotentialFieldSteering())


@register("astar", "Planificador A* con mapa previo y replanificación")
def astar_controller() -> NavigationController:
    return NavigationController("astar", MissionNavigator(), AStarSteering())


@register("astar_alv", "A* + retorno visual clásico (ALV) sin GPS")
def astar_alv() -> NavigationController:
    return NavigationController("astar_alv", MissionNavigator(AverageLandmarkVectorHoming()), AStarSteering())


@register("direct_supervised", "Directo + supervisor de seguridad (ablación del supervisor)")
def direct_supervised() -> SafetySupervisor:
    return SafetySupervisor(direct(), name="direct_supervised")


@register("astar_supervised", "A* + supervisor de seguridad (ablación del supervisor)")
def astar_supervised() -> SafetySupervisor:
    return SafetySupervisor(astar_controller(), name="astar_supervised")
