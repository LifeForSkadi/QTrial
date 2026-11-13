# -*- coding: utf-8 -*-
"""自研 VF2++ 子图嵌入（方案 B，2026-11-13 执行顺序第 6 项）。

用途：布局重标记的精确嵌入——把线路交互图 H（节点 = 2Q 门触碰的
物理位置，边 = 2Q 门对及门数）子图同构嵌入设备耦合图 G（可用节点），
并在所有嵌入中最大化 log 空间噪声分数：
    score = Σ_节点 w1q[p]·log(1−ε1q[v]) + Σ_读出 wro[p]·log(1−εro[v])
          + Σ_边 cnt·log(1−ε2q(v, v'))
（w1q = 该位置的 1Q 门数，wro = 该位置的测量数；与管线对数保真度
 同口径——补上旧 networkx 精确嵌入"只按 2Q 加权"的缺陷。）

算法（学习 qiskit VF2PostLayout 架构；公开文献）：
  - Cordella et al., IEEE TPAMI 23(12), 2001（VF2 五规则）；
  - arXiv:2209.15512（VF2++ 节点排序：最大度起点 + BFS 度数/稀有度）；
  - 搜索内增量打分 + 分支限界（乐观上界剪枝）；
  - 硬预算检查在**递归内部**（call_limit / max_solutions / wall，
    每 1024 次访问查一次墙钟）——graphstate_50 事故教训。

正确性边界：子图同构 = H 的边映射到 G 的边（非诱导；H 的非边允许
落到 G 的边上）。返回仅覆盖 H 活跃节点；全域双射由调用方补全。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class VF2Config:
    call_limit: int = 2_000_000
    max_solutions: int = 4096
    wall_limit: float = 8.0
    check_every: int = 1024
    # 候选设备节点排序策略（2026-11-13 QFT/graphstate 归因研究）：
    # "score" = 历史行为，按 log 噪声分数降序（结构盲，稀疏长环上
    # 搜索爆炸——graphstate_100 60s 无解）；
    # "warnsdorff" = 结构优先，自由邻居越少越先，分数仅作平局
    # （Hamiltonian 路径经典启发式；同实例 54 次尝试/6ms 找到 96-环）。
    cand_order: str = "warnsdorff"


def _log1m(e: float) -> float:
    return math.log(max(1.0 - float(e), 1e-12))


def embedding_score(mapping: dict, h_edges: dict, w1q: dict, wro: dict,
                    err1q, err_ro, err2q: dict, median2q: float) -> float:
    """映射 {H 节点 → 设备节点} 的 log 空间噪声分数（测试/回归用）。"""
    s = 0.0
    for u, v in mapping.items():
        s += w1q.get(u, 0) * _log1m(err1q[v])
        s += wro.get(u, 0) * _log1m(err_ro[v])
    for (a, b), cnt in h_edges.items():
        ma, mb = mapping[a], mapping[b]
        e = err2q.get((ma, mb), err2q.get((mb, ma), median2q))
        s += cnt * _log1m(e)
    return s


def _ordering(H_adj: dict, nodes: list) -> list:
    """VF2++ 风格节点排序：最大度起点 + BFS（度/邻居度/连接数降序）。"""
    deg = {u: len(H_adj[u]) for u in nodes}
    nbr_deg = {u: sum(deg[v] for v in H_adj[u]) for u in nodes}
    start = min(nodes, key=lambda u: (-deg[u], -nbr_deg[u], u))
    order = [start]
    placed = {start}
    queue = [start]
    while queue:
        u = queue.pop(0)
        nxt = [v for v in H_adj[u] if v not in placed]
        nxt.sort(key=lambda v: (-deg[v], -nbr_deg[v], v))
        for v in nxt:
            if v not in placed:
                placed.add(v)
                order.append(v)
                queue.append(v)
    # 非连通分量：继续从剩余最高度节点 BFS
    while len(order) < len(nodes):
        rest = [u for u in nodes if u not in placed]
        s = min(rest, key=lambda u: (-deg[u], -nbr_deg[u], u))
        order.append(s)
        placed.add(s)
        queue = [s]
        while queue:
            u = queue.pop(0)
            nxt = [v for v in H_adj[u] if v not in placed]
            nxt.sort(key=lambda v: (-deg[v], -nbr_deg[v], v))
            for v in nxt:
                if v not in placed:
                    placed.add(v)
                    order.append(v)
                    queue.append(v)
    return order


def find_embedding(nodes, h_edges: dict, g_adj, usable: list,
                   err1q, err_ro, err2q: dict, median2q: float,
                   w1q: dict, wro: dict,
                   config: VF2Config | None = None):
    """在 G 的可用节点上找 H 的最优子图嵌入。

    Returns: (mapping {H 节点→设备节点}, score) 或 (None, -inf)。
    """
    cfg = config or VF2Config()
    if not nodes or not h_edges:
        return None, float("-inf")

    H_adj = {u: set() for u in nodes}
    for (a, b) in h_edges:
        H_adj[a].add(b)
        H_adj[b].add(a)

    # G 侧预计算：可用节点、度数、邻居（仅可用子图）
    g_usable = [int(v) for v in usable]
    g_id = {v: i for i, v in enumerate(g_usable)}
    g_deg = {v: sum(1 for x in g_usable if g_adj[v, x]) for v in g_usable}
    g_nbr = {v: [x for x in g_usable if g_adj[v, x]] for v in g_usable}

    # 快速不可行剪枝：H 度序列必须能被 G 可用节点度序列支配
    degH = sorted((len(H_adj[u]) for u in nodes), reverse=True)
    degG = sorted((g_deg[v] for v in g_usable), reverse=True)
    if len(degH) > len(degG):
        return None, float("-inf")
    if any(dh > degG[i] for i, dh in enumerate(degH)):
        return None, float("-inf")

    order = _ordering(H_adj, list(nodes))
    # 分支限界上界：每个未匹配节点的最好节点分、每条未评分边的最大分
    node_best = {}
    for u in nodes:
        node_best[u] = max(
            w1q.get(u, 0) * _log1m(err1q[v])
            + wro.get(u, 0) * _log1m(err_ro[v]) for v in g_usable)
    node_suffix = [0.0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        node_suffix[i] = node_suffix[i + 1] + node_best[order[i]]
    min_err = min((err2q.get(e, median2q) for e in err2q), default=median2q)
    edge_best_unit = _log1m(min_err)
    total_edge_cnt = sum(h_edges.values())

    mapping = {}
    used = set()
    calls = [0]
    solutions = [0]
    t0 = time.time()
    stopped = [False]
    best = [(None, float("-inf"))]

    def dfs(idx, partial, unscored_edges):
        if stopped[0]:
            return
        calls[0] += 1
        if calls[0] % cfg.check_every == 0:
            if calls[0] > cfg.call_limit or time.time() - t0 > cfg.wall_limit:
                stopped[0] = True
                return
        if idx == len(order):
            solutions[0] += 1
            if partial > best[0][1] + 1e-12:
                best[0] = (dict(mapping), partial)
            if solutions[0] >= cfg.max_solutions:
                stopped[0] = True
            return
        # 乐观上界：部分分 + 剩余节点最好分 + 剩余边最大分
        if partial + node_suffix[idx] + unscored_edges * edge_best_unit \
                <= best[0][1] + 1e-12:
            return
        u = order[idx]
        mapped_nbrs = [m for m in H_adj[u] if m in mapping]
        unmapped_nbrs = [m for m in H_adj[u] if m not in mapping]
        cand = []
        for v in g_usable:
            if v in used or g_deg[v] < len(H_adj[u]):
                continue
            ok = True
            for m in mapped_nbrs:
                if not g_adj[v, mapping[m]]:
                    ok = False
                    break
            if not ok:
                continue
            # 前沿计数剪枝（VF2 R_in/R_out 简化）：未匹配 H 邻居数
            # 必须不超过 v 的未用可用邻居数
            free_nbrs = sum(1 for x in g_nbr[v] if x not in used)
            if len(unmapped_nbrs) > free_nbrs:
                continue
            # 增量分数（节点项 + 与已匹配邻居的边项）
            sc = (w1q.get(u, 0) * _log1m(err1q[v])
                  + wro.get(u, 0) * _log1m(err_ro[v]))
            new_scored = 0
            for m in mapped_nbrs:
                e = err2q.get((v, mapping[m]),
                              err2q.get((mapping[m], v), median2q))
                cnt = h_edges.get((u, m), h_edges.get((m, u), 1))
                sc += cnt * _log1m(e)
                new_scored += cnt
            cand.append((sc, new_scored, v, free_nbrs))
        if cfg.cand_order == "warnsdorff":
            # 结构优先（Warnsdorff）：自由邻居少 → 先尝试（最受限优先），
            # 噪声分数仅作平局。对路径/环类稀疏模式是决定性改进。
            cand.sort(key=lambda t: (t[3], -t[0], t[2]))
        else:
            cand.sort(key=lambda t: (-t[0], t[2]))
        for sc, new_scored, v, _free in cand:
            mapping[u] = v
            used.add(v)
            dfs(idx + 1, partial + sc, unscored_edges - new_scored)
            used.discard(v)
            del mapping[u]
            if stopped[0]:
                return

    dfs(0, 0.0, total_edge_cnt)
    if best[0][0] is None:
        return None, float("-inf")
    return best[0][0], best[0][1]
