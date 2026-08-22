"""Local search: monotone cost, incremental delta == full recompute, termination."""
import numpy as np
import pytest

from qtrail.config import PostProcessConfig
from qtrail.devices import build_tianyan287_spec
from qtrail.problems import random_program_graph
from qtrail.search.local_search import AdaptiveLocalSearch


@pytest.fixture(scope="module")
def spec():
    return build_tianyan287_spec()


def _mk(graph_n=20, seed=0):
    rng = np.random.default_rng(seed)
    g = random_program_graph(graph_n, p=0.3, rng=rng, weighted=True)
    return g


def test_cost_monotone_non_increasing(spec):
    g = _mk()
    cfg = PostProcessConfig(max_moves=500, big_prob=0.3, big_moves=3)
    rng = np.random.default_rng(1)
    start = np.random.default_rng(2).permutation(spec.n)[:g.n]
    ls = AdaptiveLocalSearch(g, spec.dist, cfg, rng=rng)
    best, cost = ls.search(start, max_moves=500)
    start_cost = ls.cost(start)
    assert cost <= start_cost + 1e-9
    assert cost == pytest.approx(ls.cost(best))


def test_incremental_swap_delta_matches_recompute(spec):
    """Fuzz: swap delta computed incrementally equals full cost difference."""
    g = _mk()
    rng = np.random.default_rng(3)
    ls = AdaptiveLocalSearch(g, spec.dist, PostProcessConfig(), rng=rng)
    for _ in range(20):
        pi = np.random.default_rng(4).permutation(spec.n)[:g.n]
        a, b = int(rng.integers(0, g.n)), int(rng.integers(0, g.n))
        if a == b:
            continue
        delta = ls._swap_delta(pi, a, b)
        pi2 = pi.copy()
        pi2[a], pi2[b] = pi2[b], pi2[a]
        full = ls.cost(pi2) - ls.cost(pi)
        assert abs(delta - full) < 1e-4, f"delta {delta} vs full {full}"


def test_terminates_within_budget(spec):
    g = _mk(15)
    cfg = PostProcessConfig(max_moves=2000, patience=30)
    ls = AdaptiveLocalSearch(g, spec.dist, cfg, rng=np.random.default_rng(5))
    start = np.arange(15)
    best, cost = ls.search(start, max_moves=2000)
    assert cost <= ls.cost(start) + 1e-9


def test_noise_aware_cost_uses_effective_distance(spec):
    g = _mk(10)
    # effective distance at lambda=0.5 differs from pure topology
    d_eff = spec.distance_matrix(noise_lambda=0.5)
    assert not np.allclose(d_eff, spec.dist)
    ls = AdaptiveLocalSearch(g, d_eff, PostProcessConfig())
    pi = np.arange(10)
    c_eff = ls.cost(pi)
    ls_topo = AdaptiveLocalSearch(g, spec.dist, PostProcessConfig())
    c_topo = ls_topo.cost(pi)
    assert c_eff != pytest.approx(c_topo)


def test_search_many_keeps_best(spec):
    g = _mk(12)
    cfg = PostProcessConfig(max_moves=300)
    ls = AdaptiveLocalSearch(g, spec.dist, cfg, rng=np.random.default_rng(6))
    starts = [np.random.default_rng(7 + i).permutation(spec.n)[:12] for i in range(3)]
    best, cost = ls.search_many(starts, max_moves=300)
    assert cost <= min(ls.cost(s) for s in starts) + 1e-9
    assert cost == pytest.approx(ls.cost(best))
