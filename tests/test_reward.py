"""Terminal reward correctness: hand-computed costs, invariances."""
import numpy as np
import pytest
import torch

from qtrail.envs import terminal_cost_np, terminal_reward
from qtrail.problems import build_program_graph, collate_instances
from qtrail.devices import build_grid3x3_spec


def _two_node_instance():
    ops = [("cx", (0, 1))] * 3
    return build_program_graph(2, ops)


def test_hand_computed_cost_np():
    spec = build_grid3x3_spec()
    g = _two_node_instance()
    # place q0->0, q1->4 (distance 2 on 3x3 grid): cost = 3 * 2 * 2 = 12
    assert terminal_cost_np(np.array([0, 4]), g.adj, spec.dist) == pytest.approx(12.0)
    assert terminal_cost_np(np.array([0, 1]), g.adj, spec.dist) == pytest.approx(6.0)
    assert terminal_cost_np(np.array([0, 4]), g.adj, spec.dist, dist_mult=1.0) == 6.0


def test_tensor_reward_matches_np():
    spec = build_grid3x3_spec()
    g = _two_node_instance()
    batch = collate_instances([g], spec)
    pi = torch.tensor([[0, 4]])
    r = terminal_reward(pi, batch.p_adj, batch.dist, batch.w_sum, normalize=False)
    assert r.item() == pytest.approx(-12.0)
    rn = terminal_reward(pi, batch.p_adj, batch.dist, batch.w_sum, normalize=True)
    assert rn.item() == pytest.approx(-12.0 / 3.0)


def test_physical_relabeling_invariance():
    spec = build_grid3x3_spec()
    g = _two_node_instance()
    # any two physical qubits at distance 1 give the same cost
    pairs = [(0, 1), (2, 5), (6, 7), (3, 4)]
    costs = {terminal_cost_np(np.array(p), g.adj, spec.dist) for p in pairs}
    assert costs == {6.0}


def test_lambda_zero_reduces_to_topology():
    spec = build_grid3x3_spec()
    assert np.allclose(spec.distance_matrix(noise_lambda=0.0), spec.dist)
    # and the effective distance at lambda=1 equals noise_dist
    assert np.allclose(spec.distance_matrix(noise_lambda=1.0), spec.noise_dist)


def test_padded_rows_do_not_affect_cost():
    spec = build_grid3x3_spec()
    g = _two_node_instance()
    g_big = build_program_graph(4, [("cx", (0, 1))] * 3 + [("cx", (2, 3))])
    # rows 2,3 of g_big map to physical 6,7 at distance 1: independent cost
    pi = np.array([0, 1, 6, 7])
    cost_full = terminal_cost_np(pi, g_big.adj, spec.dist)
    cost_two = terminal_cost_np(pi[:2], g.adj, spec.dist)
    assert cost_full == pytest.approx(cost_two + 2 * 1 * 1)


def test_depth_penalty_reduces_to_standard_at_zero():
    """depth_lambda=0 与标准终局奖励完全一致（稀疏结构保持）。"""
    spec = build_grid3x3_spec()
    g = _two_node_instance()
    batch = collate_instances([g], spec)
    pi = torch.tensor([[0, 4]])
    r0 = terminal_reward(pi, batch.p_adj, batch.dist, batch.w_sum,
                         normalize=False, depth_lambda=0.0)
    r_std = terminal_reward(pi, batch.p_adj, batch.dist, batch.w_sum,
                            normalize=False)
    assert r0.item() == pytest.approx(r_std.item())
    assert r_std.item() == pytest.approx(-12.0)


def test_depth_penalty_hand_computed():
    """精确手算：2 节点 3 次交互、距离 2 布局的深度惩罚 = 12。"""
    spec = build_grid3x3_spec()
    g = _two_node_instance()
    batch = collate_instances([g], spec)
    pi = torch.tensor([[0, 4]])  # d(0,4)=2，cost=12，h=[12,12]
    # depth_pen = (12² + 12²) / (12 + 12) = 12
    r = terminal_reward(pi, batch.p_adj, batch.dist, batch.w_sum,
                        normalize=False, depth_lambda=1.0)
    assert r.item() == pytest.approx(-(12 + 12))


def test_depth_penalty_nonnegative_and_relabel_invariant():
    """惩罚非负；物理比特重标号不影响深度惩罚。"""
    spec = build_grid3x3_spec()
    ops = [("cx", (0, 1)), ("cx", (0, 2)), ("cx", (0, 3)), ("cx", (1, 2))]
    g = build_program_graph(4, ops, compute_feats=False)
    batch = collate_instances([g], spec)
    pi_a = torch.tensor([[4, 0, 1, 3]])
    c0 = terminal_reward(pi_a, batch.p_adj, batch.dist, batch.w_sum,
                         normalize=False, depth_lambda=0.0)
    c1 = terminal_reward(pi_a, batch.p_adj, batch.dist, batch.w_sum,
                         normalize=False, depth_lambda=1.0)
    assert c1.item() <= c0.item()  # 惩罚只增不减
    # 平移重标号（+2 mod 9 网格对称性不严格，但距离结构在 3x3 上平移等价于
    # 中心对称：4->4 不动，0<->8, 1<->7, 3<->5 是 3x3 的对称变换）
    pi_b = torch.tensor([[4, 8, 7, 5]])
    assert pi_b[0, 0].item() == 4
    d_a = spec.dist[np.ix_(pi_a[0].numpy(), pi_a[0].numpy())]
    d_b = spec.dist[np.ix_(pi_b[0].numpy(), pi_b[0].numpy())]
    if np.allclose(d_a, d_b):  # 对称成立时才断言
        r_b = terminal_reward(pi_b, batch.p_adj, batch.dist, batch.w_sum,
                              normalize=False, depth_lambda=1.0)
        assert r_b.item() == pytest.approx(c1.item())
