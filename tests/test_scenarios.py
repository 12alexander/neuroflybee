import numpy as np
import pytest

from neuroflybee.config import ProjectConfig
from neuroflybee.sim.grid import all_reachable, reachable
from neuroflybee.sim.scenarios import (FAMILIES, SPLIT_RANGES, ScenarioGenerator, Split, seeds_for,
                                       split_of)


@pytest.fixture(scope="module")
def gen() -> ScenarioGenerator:
    return ScenarioGenerator(ProjectConfig())


def test_split_ranges_are_disjoint():
    ranges = [set(r) for r in SPLIT_RANGES.values()]
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            assert not ranges[i] & ranges[j], "fuga de datos: particiones solapadas"


def test_seeds_for_and_split_of():
    seeds = seeds_for(Split.TEST, 5)
    assert seeds == [20_000, 20_001, 20_002, 20_003, 20_004]
    assert all(split_of(s) == Split.TEST for s in seeds)
    with pytest.raises(ValueError):
        seeds_for(Split.VAL, 5_000)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_generation_is_deterministic_and_feasible(gen, family):
    a, b = gen.generate(family, 20_001), gen.generate(family, 20_001)
    assert np.allclose(a.world.base, b.world.base)
    assert [o.center().tolist() for o in a.world.obstacles] == [o.center().tolist() for o in b.world.obstacles]
    w = a.world
    assert w.clearance(w.base)[0] > 2.0
    assert all(w.clearance(wp)[0] > 2.0 for wp in w.waypoints)
    assert 3 <= len(w.waypoints) <= 5
    assert a.route[0] is w.base and a.route[-1] is w.base


def test_different_seeds_give_different_worlds(gen):
    assert not np.allclose(gen.generate("nominal", 1).world.base, gen.generate("nominal", 2).world.base)


def test_families_differ_as_designed(gen):
    assert any(not o.known for o in gen.generate("unknown_obstacles", 5).world.obstacles)
    assert any(o.is_dynamic for o in gen.generate("dynamic", 5).world.obstacles)
    assert gen.generate("gps_loss", 5).gps_loss_on_return
    assert np.linalg.norm(gen.generate("wind", 5).world.wind) >= 0.1
    nominal = gen.generate("nominal", 5)
    assert all(o.known for o in nominal.world.obstacles) and not nominal.gps_loss_on_return


def test_unknown_family_rejected(gen):
    with pytest.raises(KeyError):
        gen.generate("tormenta", 1)


def test_reachability():
    occ = np.zeros((5, 5), bool)
    occ[:, 2] = True                     # muro que corta el mapa
    assert not all_reachable(occ, (0, 0), [(0, 4)])
    occ[4, 2] = False                    # se abre un hueco
    assert reachable(occ, (0, 0))[0, 4]


def test_astar_finds_path_around_wall():
    from neuroflybee.sim.grid import astar
    occ = np.zeros((7, 7), bool)
    occ[1:7, 3] = True                       # muro con hueco arriba (fila 0)
    path = astar(occ, (6, 0), (6, 6))
    assert path[0] == (6, 0) and path[-1] == (6, 6)
    assert all(not occ[c] for c in path) and any(r == 0 for r, _ in path)
    occ[0, 3] = True
    assert astar(occ, (6, 0), (6, 6)) is None
