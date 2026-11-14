"""ProgramGraph construction and feature invariants."""
import numpy as np
import pytest

from qtrail.problems import build_program_graph, random_program_graph


def _hand_built_circuit():
    """4 qubits:
    q0-q1 cx x3, q1-q2 cx x2 (q1 control, q2 target), q2-q3 cx x1,
    h on q0 x2, h on q1 x1
    """
    ops = []
    for _ in range(3):
        ops.append(("cx", (0, 1)))
    for _ in range(2):
        ops.append(("cx", (1, 2)))
    ops.append(("cx", (2, 3)))
    ops.append(("h", (0,)))
    ops.append(("h", (0,)))
    ops.append(("h", (1,)))
    return ops


def test_adjacency_weights():
    g = build_program_graph(4, _hand_built_circuit())
    assert g.n == 4
    assert g.adj[0, 1] == 3 and g.adj[1, 0] == 3
    assert g.adj[1, 2] == 2
    assert g.adj[2, 3] == 1
    assert g.adj[0, 2] == 0
    assert g.total_weight == 6.0


def test_feature_shape_and_ranges():
    g = build_program_graph(4, _hand_built_circuit())
    f = g.node_features
    assert f.shape == (4, 6)
    assert np.all(f >= -1e-6) and np.all(f <= 1 + 1e-6)
    assert np.allclose(f[:, 0].sum(), 3 / 9)  # single-gate density sums
    # control density: q0: 3/6, q1: 2/6, q2: 1/6, q3: 0
    assert np.allclose(f[:, 1], [0.5, 1 / 3, 1 / 6, 0.0])
    # target density: q0: 0, q1: 3/6, q2: 2/6, q3: 1/6
    assert np.allclose(f[:, 2], [0.0, 0.5, 1 / 3, 1 / 6])


def test_cost_hand_computed():
    g = build_program_graph(4, _hand_built_circuit())
    # 3x3 grid distances: pi = identity-ish mapping [0,1,2,3] -> [0,1,2,4]
    dist = np.full((9, 9), 5.0, dtype=np.float32)
    for i in range(9):
        dist[i, i] = 0
    r, c = np.mgrid[0:3, 0:3]
    idx = (r * 3 + c).reshape(-1)
    for a in range(9):
        for b in range(9):
            dist[a, b] = abs(a // 3 - b // 3) + abs(a % 3 - b % 3)
    pi = np.array([0, 1, 2, 4])
    # d(0,1)=1, d(1,2)=1, d(2,4)=|0-1|+|2-1|=2
    # cost = 2 * (3*1 + 2*1 + 1*2) = 14
    assert g.cost(pi, dist) == pytest.approx(14.0)
    assert g.cost(pi, dist, dist_mult=1.0) == pytest.approx(7.0)


def test_random_graph_distribution():
    rng = np.random.default_rng(0)
    g = random_program_graph(20, p=0.3, rng=rng)
    assert g.n == 20
    assert g.node_features is None
    n_edges = (g.adj > 0).sum() / 2
    # 190 pairs * 0.3 = ~57 edges; loose sanity bound
    assert 30 <= n_edges <= 90
    # symmetric binary (unweighted mode)
    assert np.allclose(g.adj, g.adj.T)


def test_order_default_and_custom():
    g = build_program_graph(4, _hand_built_circuit())
    assert np.array_equal(g.logical_order, [0, 1, 2, 3])
    g2 = build_program_graph(4, _hand_built_circuit(),
                             order=np.array([3, 2, 1, 0]))
    assert np.array_equal(g2.logical_order, [3, 2, 1, 0])


def test_temporal_alpha_zero_unchanged():
    g = build_program_graph(4, _hand_built_circuit(), temporal_alpha=0.0)
    g0 = build_program_graph(4, _hand_built_circuit())
    assert np.allclose(g.adj, g0.adj)


def test_temporal_weighting_counts_layers():
    # (0,1) 在 3 个不同层出现，(1,2) 2 个层，(2,3) 1 个层
    ops = [("cx", (0, 1)), ("cx", (0, 1)), ("cx", (1, 2)),
           ("cx", (0, 1)), ("cx", (1, 2)), ("cx", (2, 3))]
    # 层划分：L1:(0,1) L2:(0,1) L3:(1,2) L4:(0,1) L5:(1,2) L6:(2,3)
    g = build_program_graph(4, ops, temporal_alpha=0.5, compute_feats=False)
    # (0,1): 3 次交互 × 3 层 = 3 + 0.5*3 = 4.5
    assert g.adj[0, 1] == pytest.approx(3 + 0.5 * 3)
    # (1,2): 2 + 0.5*2 = 3.0
    assert g.adj[1, 2] == pytest.approx(2 + 0.5 * 2)
    # (2,3): 1 + 0.5*1 = 1.5
    assert g.adj[2, 3] == pytest.approx(1 + 0.5 * 1)
    assert g.adj[0, 2] == 0
