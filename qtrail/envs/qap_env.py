"""QAP environment: terminal (sparse) reward computation.

CO-MAP reward (paper Eq. 5):
    cost = sum_{(i,j) in Eq} 2 * d(pi(qi), pi(qj))
    reward = -cost
Only the terminal state yields a reward — no intermediate shaping (QTrail's
sparse-reward design choice as well).
"""
from __future__ import annotations

import numpy as np
import torch


def terminal_cost_np(pi: np.ndarray, adj: np.ndarray, dist: np.ndarray,
                     dist_mult: float = 2.0) -> float:
    """Cost of a full mapping pi (logical -> physical) for one instance.

    pi: [n] int; adj: [n, n] interaction weights; dist: [N, N].
    """
    idx = np.asarray(pi, dtype=np.int64)
    d = dist[np.ix_(idx, idx)]
    return float((adj * d).sum() / 2.0 * dist_mult)


def terminal_cost_tensor(pi: torch.Tensor, p_adj: torch.Tensor, dist: torch.Tensor,
                         dist_mult: float = 2.0) -> torch.Tensor:
    """Vectorized terminal cost [B] for a batch of mappings.

    pi: [B, n_max] int64; p_adj: [B, n_max, n_max]; dist: [B, N, N] (or [N, N]).
    """
    B, n_max = pi.shape
    N = dist.shape[-1]
    if dist.dim() == 2:
        dist = dist.unsqueeze(0).expand(B, N, N)
    pi_c = pi.clamp(0, N - 1)
    d_rows = dist.gather(1, pi_c.unsqueeze(2).expand(B, n_max, N))   # [B, n, N]
    d_pi = d_rows.gather(2, pi_c.unsqueeze(1).expand(B, n_max, n_max))
    cost = (p_adj * d_pi).sum(dim=(1, 2)) / 2.0 * dist_mult
    return cost


def terminal_reward(pi: torch.Tensor, p_adj: torch.Tensor, dist: torch.Tensor,
                    w_sum: torch.Tensor, dist_mult: float = 2.0,
                    normalize: bool = True,
                    depth_lambda: float = 0.0,
                    compactness_lambda: float = 0.0) -> torch.Tensor:
    """Terminal reward [B]: -cost or -cost / total_weight.

    depth_lambda > 0 时附加深度感知项（QTrail 优化，保持终局一次性稀疏结构）：
      depth_pen = sum_q h(q)^2 / sum_q h(q),
      h(q) = sum_e w(q,e) * dist_mult * d(pi(q), pi(e))  （比特 q 的交互负载）
    惩罚交互热点比特 → 负载均衡 → 布局偏向低深度。

    compactness_lambda > 0 时附加紧凑性项（借鉴 GraphPlacement 紧凑嵌入
    思想）：惩罚布局占据的物理直径（最远两比特距离）→ 紧凑布局对并行
    路由更友好。
    """
    B, n_max = pi.shape
    N = dist.shape[-1]
    if dist.dim() == 2:
        dist = dist.unsqueeze(0).expand(B, N, N)
    pi_c = pi.clamp(0, N - 1)
    d_rows = dist.gather(1, pi_c.unsqueeze(2).expand(B, n_max, N))
    d_pi = d_rows.gather(2, pi_c.unsqueeze(1).expand(B, n_max, n_max))

    cost = terminal_cost_tensor(pi, p_adj, dist, dist_mult)
    if depth_lambda > 0:
        h = (p_adj * d_pi).sum(dim=-1) * dist_mult        # [B, n] 每比特交互负载
        denom = h.sum(dim=-1).clamp(min=1.0)              # [B]
        cost = cost + depth_lambda * (h ** 2).sum(dim=-1) / denom
    if compactness_lambda > 0:
        diam = d_pi.amax(dim=(1, 2))                      # 布局物理直径
        cost = cost + compactness_lambda * diam
    if normalize:
        return -cost / w_sum.clamp(min=1.0)
    return -cost


def layout_cost_np(layout: dict, adj: np.ndarray, dist: np.ndarray,
                   dist_mult: float = 2.0) -> float:
    """Cost of a layout dict {logical: physical} (used by local search)."""
    n = adj.shape[0]
    pi = np.array([layout[i] for i in range(n)], dtype=np.int64)
    return terminal_cost_np(pi, adj, dist, dist_mult)
