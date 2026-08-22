"""路由 oracle 终局奖励测试：合成线路与奖励合理性。"""
import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit

from qtrail.training.oracle import (oracle_rewards, routing_swaps,
                                    synthesize_layered_circuit)
from qtrail.problems import build_program_graph, random_program_graph
from qtrail.pipeline.routing import coupling_map_from_spec
from qtrail.devices import build_grid3x3_spec, build_grid8x8_spec


def test_synthesize_layered_circuit_valid():
    """合成线路的层结构合法：同层内比特互斥（可并行）。"""
    ops = [("cx", (0, 1)), ("cx", (0, 1)), ("cx", (1, 2)), ("cx", (2, 3)),
           ("cx", (3, 0))]
    g = build_program_graph(4, ops, compute_feats=False)
    qc = synthesize_layered_circuit(g, cap=3, seed=0)
    # 线路为 CX 且比特索引合法
    for inst in qc.data:
        assert inst.operation.name == "cx"
        assert all(0 <= qc.find_bit(q).index < 4 for q in inst.qubits)
    # 2Q 门总数 = Σ min(w, cap)
    n_gates = sum(min(int(round(float(g.adj[i, j]))), 3)
                  for i in range(4) for j in range(i + 1, 4))
    assert qc.count_ops().get("cx", 0) == n_gates


def test_good_layout_fewer_swaps_than_bad():
    """好布局（交互比特相邻）的真实路由 SWAP 数应少于坏布局。"""
    spec = build_grid8x8_spec()
    cm = coupling_map_from_spec(spec)
    # 链式图 0-1-2-3-4
    ops = [("cx", (i, i + 1)) for i in range(4)]
    g = build_program_graph(5, ops, compute_feats=False)
    qc = synthesize_layered_circuit(g, cap=5, seed=0)
    good = {i: i for i in range(5)}        # 相邻放置
    bad = {0: 0, 1: 7, 2: 14, 3: 21, 4: 28}  # 分散放置
    s_good = routing_swaps(qc, good, cm, seed=0)
    s_bad = routing_swaps(qc, bad, cm, seed=0)
    assert s_good <= s_bad
    assert s_good == 0  # 完美布局零 SWAP


def test_oracle_reward_sparsity_structure():
    """oracle 奖励仍是终局一次性（函数不接触中间状态），且好布局奖励更高。"""
    spec = build_grid8x8_spec()
    cm = coupling_map_from_spec(spec)
    ops = [("cx", (i, i + 1)) for i in range(4)]
    g = build_program_graph(5, ops, compute_feats=False)
    n_max = 5
    pi_good = torch.zeros(1, n_max, dtype=torch.int64)
    pi_bad = torch.zeros(1, n_max, dtype=torch.int64)
    for i in range(5):
        pi_good[0, i] = i
        pi_bad[0, i] = [0, 7, 14, 21, 28][i]
    r_good = oracle_rewards([g], pi_good, cm, seed=0, oracle_max_n=25)
    r_bad = oracle_rewards([g], pi_bad, cm, seed=0, oracle_max_n=25)
    assert r_good.item() == pytest.approx(0.0)  # 完美布局 → 0 SWAP
    assert r_bad.item() < r_good.item()          # 坏布局奖励更低


def test_oracle_fallback_for_large_n():
    """超过 oracle_max_n 的 episode 回退静态奖励。"""
    spec = build_grid8x8_spec()
    cm = coupling_map_from_spec(spec)
    rng = np.random.default_rng(0)
    g = random_program_graph(40, p=0.3, rng=rng)
    n_max = 40
    pi = torch.zeros(1, n_max, dtype=torch.int64)
    pi[0] = torch.arange(n_max)
    static = torch.tensor([-12.34])
    r = oracle_rewards([g], pi, cm, seed=0, oracle_max_n=25,
                       static_rewards=static)
    assert r.item() == pytest.approx(-12.34)  # 回退


def test_oracle_rewards_mixed_batch():
    """混合规模批次：小图 oracle、大图静态，输出形状正确。"""
    spec = build_grid8x8_spec()
    cm = coupling_map_from_spec(spec)
    ops = [("cx", (i, i + 1)) for i in range(3)]
    g_small = build_program_graph(4, ops, compute_feats=False)
    rng = np.random.default_rng(1)
    g_big = random_program_graph(30, p=0.3, rng=rng)
    pi = torch.zeros(2, 30, dtype=torch.int64)
    pi[0, :4] = torch.tensor([0, 1, 2, 3])
    pi[1] = torch.arange(30)
    static = torch.tensor([-1.0, -2.0])
    r = oracle_rewards([g_small, g_big], pi, cm, seed=0, oracle_max_n=25,
                       static_rewards=static)
    assert r.shape == (2,)
    assert r[1].item() == pytest.approx(-2.0)      # 大图回退
    assert r[0].item() == pytest.approx(0.0)       # 完美布局小图 oracle
