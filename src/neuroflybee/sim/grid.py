"""Búsquedas sobre la grilla de ocupación."""
from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

Cell = tuple[int, int]
NEIGHBORS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def reachable(occupied: np.ndarray, start: Cell) -> np.ndarray:
    """Máscara de celdas alcanzables desde `start` (búsqueda en anchura, 8 vecinos)."""
    rows, cols = occupied.shape
    seen = np.zeros_like(occupied, dtype=bool)
    if occupied[start]:
        return seen
    seen[start] = True
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and not occupied[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def all_reachable(occupied: np.ndarray, start: Cell, goals: Iterable[Cell]) -> bool:
    mask = reachable(occupied, start)
    return all(mask[g] for g in goals)


_STEP_COST = {(dr, dc): (1.4142 if dr and dc else 1.0) for dr, dc in NEIGHBORS_8}


def astar(occupied: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """A* 8-conexo con heurística octil (admisible). Devuelve la lista de celdas o None."""
    import heapq

    rows, cols = occupied.shape
    if occupied[goal] or occupied[start]:
        return None

    def h(cell: Cell) -> float:
        dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        return (dr + dc) + (1.4142 - 2) * min(dr, dc)

    open_heap: list[tuple[float, float, Cell]] = [(h(start), 0.0, start)]
    came_from: dict[Cell, Cell] = {}
    best_g = {start: 0.0}
    while open_heap:
        _, g, cell = heapq.heappop(open_heap)
        if cell == goal:
            path = [cell]
            while cell in came_from:
                cell = came_from[cell]
                path.append(cell)
            return path[::-1]
        if g > best_g.get(cell, np.inf):
            continue
        for (dr, dc), step in _STEP_COST.items():
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or occupied[nxt]:
                continue
            ng = g + step
            if ng < best_g.get(nxt, np.inf):
                best_g[nxt] = ng
                came_from[nxt] = cell
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None
