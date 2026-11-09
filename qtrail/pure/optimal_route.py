# -*- coding: utf-8 -*-
"""精确路由求解器（A*，2026-11-11）：小线路最优 swap 轨迹自产。

用途：① 学习式路由的训练标签（专家轨迹 = 最优解路径）；
② 我们路由器的差距量化（最优上界参照）；③ 论文资产。

A* 设计：
- 状态 = (布局元组, 已执行门位图)——同一布局经不同路径可达不同
  执行进度，必须联合作为状态；
- 动作 = 耦合图上的合法 swap（代价 1）；
- 目标 = 全部 2Q 门执行完（扩展时执行闭包：依赖满足的 1Q 门立即、
  相邻 2Q 门立即——确定性闭包减小分支因子）；
- 可采纳启发式 h = max(ceil(Σ_{前沿2Q} D/|F|), max_D - 1)：
  一次 swap 使每个受影响门距离至多减 1（Σ 减少 ≤ |F|），单门距离
  每次至多减 1；max 仍可采纳 → A* 最优性有保证；
- 剪枝：自研 SABRE 多种子上界 UB（f ≥ UB 剪枝）；预算超限返回
  None（该线路放弃做标签）。
"""
from __future__ import annotations

import heapq
from functools import lru_cache

import numpy as np

from qtrail.pure.circuit import Circuit
from qtrail.pure.router import sabre_route


_DEV_CACHE = {}


def _device_data(spec):
    """(adj 集合, 邻接表, dist 矩阵) —— 按 spec 对象 id 缓存。"""
    key = id(spec)
    if key not in _DEV_CACHE:
        adj = frozenset((i, j) for i in range(spec.n)
                        for j in range(i + 1, spec.n) if spec.adj[i, j])
        nbrs = [[] for _ in range(spec.n)]
        for a in range(spec.n):
            for b in range(a + 1, spec.n):
                if spec.adj[a, b]:
                    nbrs[a].append(b)
                    nbrs[b].append(a)
        _DEV_CACHE[key] = (adj, nbrs, spec.dist.astype(np.int64))
    return _DEV_CACHE[key]


def optimal_route(circ: Circuit, spec, layout: dict, seed: int = 0,
                  budget: int = 200_000, max_swaps: int = 60) \
        -> tuple[int, list] | None:
    """A* 最优路由：返回 (最优 swap 数, swap 序列)，超预算/超阈值
    返回 None（该线路不适合做标签）。"""
    adj, nbrs, dist = _device_data(spec)
    m = len(circ.ops)
    ops = circ.ops
    deps = [set(d) for d in circ.deps()]

    # 上界：自研 SABRE 多种子取最小
    ub = max_swaps
    for s in range(4):
        try:
            _, sw, _ = sabre_route(circ, spec, dict(layout), seed=seed + s,
                                   max_swaps=max_swaps)
            ub = min(ub, sw)
        except Exception:
            pass
    if ub >= max_swaps:
        return None

    n = circ.n
    init_pos = [int(layout.get(i, i)) for i in range(n)]

    def close(exec_mask, positions):
        """执行闭包：依赖满足的 1Q 门立即执行、相邻 2Q 门立即执行。"""
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if (exec_mask >> i) & 1:
                    continue
                if deps[i] and not all((exec_mask >> j) & 1
                                       for j in deps[i]):
                    continue
                inst = ops[i]
                if inst.nq == 1:
                    exec_mask |= 1 << i
                    changed = True
                elif inst.nq == 2:
                    p1 = positions[inst.qubits[0]]
                    p2 = positions[inst.qubits[1]]
                    if (p1, p2) in adj if p1 < p2 else (p2, p1) in adj:
                        exec_mask |= 1 << i
                        changed = True
        frontier = [i for i in range(m)
                    if not ((exec_mask >> i) & 1) and ops[i].nq == 2
                    and (not deps[i] or all((exec_mask >> j) & 1
                                            for j in deps[i]))]
        return exec_mask, frontier

    def heuristic(positions, frontier):
        if not frontier:
            return 0
        total = 0
        maxd = 0
        for i in frontier:
            inst = ops[i]
            d = int(dist[positions[inst.qubits[0]],
                         positions[inst.qubits[1]]])
            total += d
            maxd = max(maxd, d)
        return max((total + len(frontier) - 1) // len(frontier), maxd - 1, 0)

    start_mask, start_fr = close(0, init_pos)
    if not start_fr:
        return 0, []
    start = (tuple(init_pos), start_mask)
    start_h = heuristic(init_pos, start_fr)
    if start_h > ub:
        return None
    open_heap = [(start_h, 0, start)]
    g_cost = {start: 0}
    came = {}  # state -> (prev_state, swap)
    expanded = 0
    while open_heap:
        f, g, state = heapq.heappop(open_heap)
        if g_cost.get(state, 1 << 30) < g:
            continue
        expanded += 1
        if expanded > budget:
            return None
        pos_t, mask = state
        positions = list(pos_t)
        mask, frontier = close(mask, positions)
        if not frontier:
            path = []
            st = state
            while st in came:
                prev, sw = came[st]
                path.append(sw)
                st = prev
            path.reverse()
            return g, path
        # 候选边：涉及前沿比特位置的邻接边（仅两端点均有逻辑比特）
        eps = set()
        for i in frontier:
            inst = ops[i]
            eps.add(positions[inst.qubits[0]])
            eps.add(positions[inst.qubits[1]])
        cand = set()
        for a in eps:
            for b in nbrs[a]:
                cand.add((a, b) if b > a else (b, a))
        for a, b in sorted(cand):
            npos = list(positions)
            moved = 0
            for k in range(n):
                if npos[k] == a:
                    npos[k] = b
                    moved += 1
                elif npos[k] == b:
                    npos[k] = a
                    moved += 1
            if moved < 1:
                continue  # 两端点均无逻辑比特的交换无意义
            ng = g + 1
            if ng > ub:
                continue
            nmask, nfr = close(mask, npos)
            nstate = (tuple(npos), nmask)
            if g_cost.get(nstate, 1 << 30) <= ng:
                continue
            h = heuristic(npos, nfr)
            if ng + h > ub:
                continue
            g_cost[nstate] = ng
            came[nstate] = (state, (a, b))
            heapq.heappush(open_heap, (ng + h, ng, nstate))
    return None


def headroom_report(circ, spec, layout, seed=0, budget=200_000,
                    max_swaps=60) -> dict:
    """我们路由器 vs 最优的差距量化。optimal=None 表示 A* 超预算。"""
    _, sw_ours, _ = sabre_route(circ, spec, dict(layout), seed=seed,
                                max_swaps=max_swaps)
    res = optimal_route(circ, spec, layout, seed=seed, budget=budget,
                        max_swaps=max_swaps)
    if res is None:
        return {"ours": sw_ours, "optimal": None}
    opt, _ = res
    return {"ours": sw_ours, "optimal": opt,
            "gap": sw_ours - opt, "ratio": sw_ours / max(opt, 1)}
