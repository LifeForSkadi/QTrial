"""QTrail adaptive multi-start local search post-processing.

Paper's neighborhood ops (random swap / random reassign) extended with:
  Phase 1 (exploration): occasional large perturbations (chains of swaps among
      "hot" logical qubits) to escape local optima; hill-climbing with patience.
  Phase 2 (exploitation): first-improvement single swaps.

Cost = effective distance (topology + lambda_n * noise delta) so noise-aware
mode and topology mode share one code path (lambda_n=0 recovers CO-MAP).
Swap cost deltas are computed incrementally — O(n) per move.
"""
from __future__ import annotations

import numpy as np

from qtrail.config import PostProcessConfig
from qtrail.envs import terminal_cost_np
from qtrail.problems import ProgramGraph


class AdaptiveLocalSearch:
    def __init__(self, graph: ProgramGraph, dist_eff: np.ndarray,
                 cfg: PostProcessConfig,
                 rng: np.random.Generator | None = None):
        self.adj = graph.adj
        self.dist = dist_eff
        self.n = graph.n
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(0)

        # "hot" logical qubits: top hot_frac by weighted incident interaction
        degree_w = graph.adj.sum(axis=1)
        order = np.argsort(-degree_w)
        n_hot = max(1, int(graph.n * cfg.hot_frac))
        self.hot = order[:n_hot]

    # ------------------------------------------------------------- costs
    def cost(self, pi: np.ndarray) -> float:
        return terminal_cost_np(pi, self.adj, self.dist, dist_mult=2.0)

    def _swap_delta(self, pi: np.ndarray, a: int, b: int) -> float:
        """Cost change when swapping physical positions of logical a and b."""
        adj = self.adj
        d = self.dist
        pa, pb = pi[a], pi[b]
        nz = np.flatnonzero(adj[a])   # neighbors of a
        delta = 0.0
        for l in nz:
            if l == b:
                continue
            pl = pi[l]
            delta += float(adj[a, l]) * (d[pb, pl] - d[pa, pl])
        nz = np.flatnonzero(adj[b])
        for l in nz:
            if l == a:
                continue
            pl = pi[l]
            delta += float(adj[b, l]) * (d[pa, pl] - d[pb, pl])
        return 2.0 * delta  # dist_mult factor

    def _reassign_delta(self, pi: np.ndarray, a: int, p_new: int) -> float:
        """Cost change when moving logical a to free physical p_new."""
        adj = self.adj
        d = self.dist
        p_old = pi[a]
        delta = 0.0
        for l in np.flatnonzero(adj[a]):
            delta += float(adj[a, l]) * (d[p_new, pi[l]] - d[p_old, pi[l]])
        return 2.0 * delta

    # ------------------------------------------------ 嵌入感知（路径 B）
    def _embed_gain(self, pi: np.ndarray, a: int, b: int) -> int:
        """交换逻辑 a,b 后，相邻（dist==1）交互对数目的变化。

        借鉴 GraphPlacement 的结构匹配思想（自主实现）：优先让更多
        交互对落在物理相邻位置上。
        """
        d = self.dist
        gain = 0
        for l in np.flatnonzero(self.adj[a]):
            if l == b:
                continue
            was = int(d[pi[a], pi[l]] == 1)
            now = int(d[pi[b], pi[l]] == 1)
            gain += now - was
        for l in np.flatnonzero(self.adj[b]):
            if l == a:
                continue
            was = int(d[pi[b], pi[l]] == 1)
            now = int(d[pi[a], pi[l]] == 1)
            gain += now - was
        return gain

    def _embed_swap(self, pi: np.ndarray) -> tuple[np.ndarray, float, int, int, int]:
        """针对未满足交互对的定向交换：随机选一条 dist>1 的交互边，
        交换其一端与随机比特。返回 (cand, delta, embed_gain, a, b)。"""
        n = self.n
        unsat = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if self.adj[i, j] > 0 and self.dist[pi[i], pi[j]] > 1]
        if not unsat:
            return pi, 0.0, 0, 0, 0
        i, j = unsat[int(self.rng.integers(0, len(unsat)))]
        a = j if self.rng.random() < 0.5 else i
        b = int(self.rng.integers(0, n))
        if a == b:
            return pi, 0.0, 0, 0, 0
        delta = self._swap_delta(pi, a, b)
        gain = self._embed_gain(pi, a, b)
        pi_new = pi.copy()
        pi_new[a], pi_new[b] = pi_new[b], pi_new[a]
        return pi_new, delta, gain, a, b

    # ------------------------------------------------------------ moves
    def _swap_move(self, pi: np.ndarray, phase1: bool) -> tuple[np.ndarray, float, bool]:
        """Propose a swap (or a chain of swaps in phase 1). Returns
        (new_pi, delta, applied)."""
        rng = self.rng
        n = self.n
        if phase1 and rng.random() < self.cfg.big_prob:
            # large perturbation: chain of swaps among hot qubits
            pi_new = pi.copy()
            total = 0.0
            moved = False
            for _ in range(self.cfg.big_moves):
                a = int(rng.choice(self.hot))
                b = int(rng.integers(0, n))
                if a == b:
                    continue
                d = self._swap_delta(pi_new, a, b)
                pi_new[a], pi_new[b] = pi_new[b], pi_new[a]
                total += d
                moved = True
            return pi_new, total, moved
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        if a == b:
            return pi, 0.0, False
        delta = self._swap_delta(pi, a, b)
        pi_new = pi.copy()
        pi_new[a], pi_new[b] = pi_new[b], pi_new[a]
        return pi_new, delta, True

    def _reassign_move(self, pi: np.ndarray, free_phys: np.ndarray):
        if len(free_phys) == 0:
            return pi, 0.0, False
        a = int(self.rng.integers(0, self.n))
        p_new = int(self.rng.choice(free_phys))
        delta = self._reassign_delta(pi, a, p_new)
        pi_new = pi.copy()
        pi_new[a] = p_new
        return pi_new, delta, True

    # ----------------------------------------------------------- search
    def search(self, start_pi: np.ndarray, free_phys: np.ndarray | None = None,
               max_moves: int | None = None) -> tuple[np.ndarray, float]:
        """Improve a starting layout; returns (best_pi, best_cost).

        Monotone non-increasing cost (hill climbing). Terminates on move
        budget or patience exhaustion. Phase 1 = exploration (reassign moves
        allowed, large perturbations), phase 2 = exploitation (swaps only).
        """
        cfg = self.cfg
        max_moves = max_moves or cfg.max_moves
        phase1_moves = int(max_moves * (1.0 - cfg.phase2_ratio))

        pi = start_pi.copy()
        best_pi = pi.copy()
        best_cost = self.cost(pi)
        patience_left = cfg.patience

        if free_phys is not None:
            free = free_phys.copy()
        else:
            free = np.array(sorted(set(range(self.dist.shape[0])) - set(pi.tolist())),
                            dtype=np.int64)

        for move_i in range(max_moves):
            phase1 = move_i < phase1_moves

            # 嵌入定向交换（路径 B）：以结构匹配为目标的移动
            if (cfg.embedding_moves and not phase1
                    and self.rng.random() < cfg.embed_prob):
                cand, delta, gain, a, b = self._embed_swap(pi)
                if a == b:
                    patience_left -= 1
                elif delta < -1e-9 or (abs(delta) <= cfg.embedding_tol
                                       and gain > 0):
                    pi = cand
                    best_cost += delta
                    best_pi = pi.copy()
                    patience_left = cfg.patience
                else:
                    patience_left -= 1
                if patience_left <= 0:
                    break
                continue

            if phase1 and len(free) > 0 and self.rng.random() < 0.1:
                # reassign: move one logical qubit to a free physical slot
                a = int(self.rng.integers(0, self.n))
                p_new = int(self.rng.choice(free))
                delta = self._reassign_delta(pi, a, p_new)
                if delta < -1e-9:
                    p_old = int(pi[a])
                    pi[a] = p_new
                    free = np.append(free[free != p_new], p_old)
                    best_cost += delta
                    best_pi = pi.copy()
                    patience_left = cfg.patience
                else:
                    patience_left -= 1
            else:
                cand, delta, ok = self._swap_move(pi, phase1)
                if not ok:
                    patience_left -= 1
                elif delta < -1e-9:
                    pi = cand
                    best_cost += delta
                    best_pi = pi.copy()
                    patience_left = cfg.patience
                else:
                    # phase 1: chain perturbations are accept-or-discard;
                    # phase 2: first-improvement — non-improving probes consume patience
                    patience_left -= 1

            if patience_left <= 0:
                break

        return best_pi, best_cost

    def search_many(self, starts: list[np.ndarray],
                    max_moves: int | None = None) -> tuple[np.ndarray, float]:
        """Multi-start: search from each start, keep the best."""
        best_pi, best_cost = None, float("inf")
        for s in starts:
            pi, cost = self.search(s, max_moves=max_moves)
            if cost < best_cost:
                best_pi, best_cost = pi, cost
        return best_pi, best_cost


def improve_layout(graph: ProgramGraph, dist_eff: np.ndarray, start_pi: np.ndarray,
                   cfg: PostProcessConfig, rng: np.random.Generator | None = None,
                   max_moves: int | None = None) -> tuple[np.ndarray, float]:
    """Convenience wrapper: single-start adaptive local search."""
    ls = AdaptiveLocalSearch(graph, dist_eff, cfg, rng=rng)
    return ls.search(start_pi, max_moves=max_moves)
