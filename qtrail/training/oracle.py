"""路由 oracle 终局奖励：以候选布局的真实路由 SWAP 数作为训练奖励。

程序图不含门序列，无法直接路由——从交互图合成"分层线路"：
  1. 把加权边展开为多重边（权重上限 cap），对多重图做贪心边染色，
     每个颜色类 = 一个并行层（同层边互斥比特，可并行执行）
  2. 逐层发射 CX 门，得到与程序图交互结构一致的合成线路
  3. 对候选布局运行 SabreSwap（固定种子）→ 真实 SWAP 数 → 归一化奖励
     reward = -swaps / sum(w)（与静态奖励同归一化，稀疏终局结构不变）

仅对 n ≤ oracle_max_n 的 episode 启用（路由开销可控）；其余沿用静态奖励。
"""
from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit

from qtrail.problems import ProgramGraph

# 模块级 QASM 解析缓存（池图对象跨 epoch 复用，避免重复解析）
_QASM_CACHE: dict = {}


def synthesize_layered_circuit(g: ProgramGraph, cap: int = 5,
                               seed: int = 0) -> QuantumCircuit:
    """从加权交互图合成分层 CX 线路（贪心边染色）。"""
    rng = np.random.default_rng(seed)
    n = g.n
    # 多重边列表：(u, v, color) —— 贪心边染色
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = int(min(round(float(g.adj[i, j])), cap))
            edges.extend([(i, j)] * w)
    # 洗牌避免偏向（确定性种子）
    rng.shuffle(edges)
    colors = {}
    used_at_vertex = [set() for _ in range(n)]
    layers = []
    for (u, v) in edges:
        col = 0
        while col in used_at_vertex[u] or col in used_at_vertex[v]:
            col += 1
        used_at_vertex[u].add(col)
        used_at_vertex[v].add(col)
        while len(layers) <= col:
            layers.append([])
        layers[col].append((u, v))

    qc = QuantumCircuit(n)
    for layer in layers:
        for (u, v) in layer:
            qc.cx(u, v)
    return qc


def routing_swaps(qc: QuantumCircuit, layout: dict, cm, seed: int) -> int:
    """给定布局下 SabreSwap 的真实 SWAP 数（轻量路由链）。"""
    from qiskit.transpiler import Layout, PassManager
    from qiskit.transpiler.passes import (ApplyLayout, EnlargeWithAncilla,
                                          FullAncillaAllocation, SabreSwap,
                                          SetLayout)
    init = Layout({qc.qubits[k]: v for k, v in layout.items()})
    pm = PassManager([
        SetLayout(init),
        FullAncillaAllocation(coupling_map=cm),
        EnlargeWithAncilla(),
        ApplyLayout(),
        SabreSwap(coupling_map=cm, heuristic="decay", seed=seed),
    ])
    routed = pm.run(qc)
    return routed.count_ops().get("swap", 0)


def oracle_rewards(graphs: list[ProgramGraph], pi: torch.Tensor, cm,
                   seed: int, oracle_max_n: int = 25, cap: int = 5,
                   static_rewards: torch.Tensor | None = None,
                   noise_lambda: float = 0.0,
                   dist: np.ndarray | None = None,
                   noise_dist: np.ndarray | None = None) -> torch.Tensor:
    """逐 episode 计算奖励：n ≤ oracle_max_n 用路由 oracle，否则用静态奖励。

    oracle 电路优先用真实 QASM（g.ops_meta["qasm"]，图池重建时保存）；
    无 QASM 的图（随机图）从交互图合成边染色分层线路。

    noise_lambda > 0 时启用联合 oracle（swap+噪声）：
      cost = swaps + noise_lambda * Σ w_ij·2·(noise_dist−dist)[π_i,π_j]
    保持噪声感知（纯 swap oracle 的保真度回归修复）。

    pi: [B, n_max] int64（逻辑→物理）；graphs 与 batch 同序。
    """
    B = len(graphs)
    rewards = torch.zeros(B, dtype=torch.float32, device=pi.device)
    global _QASM_CACHE
    qasm_cache = _QASM_CACHE
    for b, g in enumerate(graphs):
        n = g.n
        w_sum = max(float(g.adj.sum() / 2.0), 1.0)
        if n > oracle_max_n or n < 2:
            # 回退静态奖励（与 policy 内部计算一致）
            if static_rewards is not None:
                rewards[b] = static_rewards[b]
            continue
        layout = {i: int(pi[b, i]) for i in range(n)}
        qasm_str = g.ops_meta.get("qasm") if g.ops_meta else None
        if qasm_str is not None:
            if id(g) not in qasm_cache:
                from qiskit import qasm2
                from qtrail.utils.qasm_io import sanitize_qasm
                qasm_cache[id(g)] = qasm2.loads(sanitize_qasm(qasm_str))
            qc = qasm_cache[id(g)]
        else:
            qc = synthesize_layered_circuit(g, cap=cap, seed=seed)
        swaps = routing_swaps(qc, layout, cm, seed)
        cost = float(swaps)
        if noise_lambda > 0 and dist is not None and noise_dist is not None:
            idx = np.array([layout[i] for i in range(n)], dtype=np.int64)
            d_t = dist[np.ix_(idx, idx)]
            d_n = noise_dist[np.ix_(idx, idx)]
            cost += noise_lambda * float(((g.adj * (d_n - d_t)).sum() / 2.0) * 2.0)
        rewards[b] = -cost / w_sum
    return rewards
