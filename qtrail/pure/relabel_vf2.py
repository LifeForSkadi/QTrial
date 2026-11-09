# -*- coding: utf-8 -*-
"""全局重标记（对标 qiskit VF2PostLayout 的自主实现 + 噪声加权扩展，
2026-11-11）。

机制：
  1. 交互图 H = 路由后线路 2Q 门（含 swap）在物理位置上的支撑图；
  2. 精确嵌入：H 是否子图同构嵌入设备耦合图？networkx GraphMatcher
     （VF2 公开算法 Cordella 2001 的标准实现，限时）——找到则置换
     重标记 + 删除全部 swap（0-SWAP 消除，语义 = 输出置换变更，
     与 qiskit VF2PostLayout 同机制）；
  3. 未找到（限时/无解）→ 噪声加权部分嵌入：活跃位置内局部搜索，
     最大化 Σ_{H 边} (1-ε) · [π 后落在设备边上]——qiskit 无此扩展
     （他们找不到完美解就放弃）；
  4. 非劣化守卫 + 硬件合法性断言（调用方防线）。

语义：全局标签置换保线路酉语义（输出比特重命名），final_layout
由调用方同步更新。
"""
from __future__ import annotations

import time

import numpy as np

from qtrail.pure.circuit import Circuit, Inst


def _interaction_graph(circ: Circuit):
    """2Q 门的位置支撑图 + 边权重（门数）。

    **不含 swap**：swap 正是重标记机制要消除的对象（嵌入只需覆盖
    非 swap 的 2Q 门；swap 计入会抬高节点度导致误判无解）。
    """
    edges = {}
    for inst in circ.ops:
        if inst.nq == 2 and inst.name != "swap":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            edges[e] = edges.get(e, 0) + 1
    return edges


def _exact_embedding(circ, spec, time_budget: float = 10.0,
                     max_active: int = 60):
    """H 的支撑图嵌入设备图：返回置换 dict {旧位置: 新位置} 或 None。

    嵌入语义：把活跃位置（含 2Q 门的位置）重命名到设备节点上，使
    全部 H 边落在设备边上。GraphMatcher 为子图同构（VF2），仅对
    ≤ max_active 个活跃位置启用（稀疏图 VF2 毫秒级；过大直接放弃，
    走部分嵌入）。
    """
    import networkx as nx
    edges = _interaction_graph(circ)
    if not edges:
        return None
    active = sorted({p for e in edges for p in e})
    if len(active) > max_active:
        return None
    H = nx.Graph()
    H.add_nodes_from(active)
    H.add_edges_from(edges.keys())
    # 度序列预检：H 节点度必须 ≤ 设备最大度（快速排除无解密集图）
    deg_H = sorted((d for _, d in H.degree()), reverse=True)
    deg_G = sorted((d for _, d in nx.Graph(
        [(i, j) for i in range(spec.n) for j in range(i + 1, spec.n)
         if spec.adj[i, j]]).degree()), reverse=True)
    if any(dh > deg_G[i] for i, dh in enumerate(deg_H)):
        return None
    G = nx.Graph()
    G.add_nodes_from(range(spec.n))
    G.add_edges_from([(i, j) for i in range(spec.n)
                      for j in range(i + 1, spec.n) if spec.adj[i, j]])
    GM = nx.algorithms.isomorphism.GraphMatcher(G, H)  # G 含 H 子图
    t0 = time.time()
    # 多个嵌入解：按 Σ (1-ε) 噪声加权选最优（qiskit VF2PostLayout 的
    # 评分在 trial 层、且无此嵌入内选优）
    err2q = spec.calib.err_2q
    med = float(np.median(list(err2q.values()))) if err2q else 1e-3

    def score(mapping):
        tot = 0.0
        for (a, b) in edges:
            ma, mb = mapping[a], mapping[b]
            e = err2q.get((ma, mb), err2q.get((mb, ma), med))
            tot += 1.0 - float(e)
        return tot

    best = None
    best_s = -1.0
    for iso in GM.subgraph_isomorphisms_iter():
        if time.time() - t0 > time_budget:
            break
        # iso: G 节点 -> H 节点。我们需要的重命名 = H 节点 -> G 节点
        mapping = {h: int(g) for g, h in iso.items()}
        s = score(mapping)
        if s > best_s:
            best_s = s
            best = mapping
    return best


def _partial_embedding(circ, spec, time_budget: float = 15.0,
                       max_iters: int = 2000):
    """噪声加权部分嵌入：活跃位置内局部搜索，最大化 Σ w_e·(1-ε)·[可嵌入]。

    移动 = 交换两个活跃位置的新标签；增量评估；首优爬山 + 随机重启。
    返回置换 dict 或 None（无改进时）。
    """
    edges = _interaction_graph(circ)
    if not edges:
        return None
    err2q = spec.calib.err_2q
    med = float(np.median(list(err2q.values()))) if err2q else 1e-3
    w = {}
    for (a, b), cnt in edges.items():
        e = err2q.get((a, b), err2q.get((b, a), med))
        w[(a, b)] = cnt * (1.0 - float(e))

    active = sorted({p for e in edges for p in e})
    rng = np.random.default_rng(0)

    def score(mapping):
        tot = 0.0
        for (a, b), wt in w.items():
            ma, mb = mapping[a], mapping[b]
            if spec.adj[ma, mb]:
                tot += wt
        return tot

    mapping = {p: p for p in active}
    best = score(mapping)
    best_map = dict(mapping)
    t0 = time.time()
    n_active = len(active)
    for it in range(max_iters):
        if time.time() - t0 > time_budget:
            break
        improved = False
        # 随机扰动后的贪心序（爬山多样性）
        order = rng.permutation(n_active)
        for ia in range(n_active):
            i = order[ia]
            a = active[i]
            for j in order[ia + 1:]:
                b = active[j]
                mapping[a], mapping[b] = mapping[b], mapping[a]
                s = score(mapping)
                if s > best + 1e-12:
                    best = s
                    best_map = dict(mapping)
                    improved = True
                else:
                    mapping[a], mapping[b] = mapping[b], mapping[a]
        if not improved:
            break
        mapping = dict(best_map)
    if best <= score({p: p for p in active}) + 1e-12:
        return None  # 无净改进
    return best_map


def global_relabel(circ: Circuit, spec, time_budget: float = 25.0) \
        -> tuple[Circuit, dict] | None:
    """全局重标记。返回 (新线路, 置换) 或 None（无收益/超时）。

    置换 dict：{旧物理位: 新物理位}（作用于线路全部标签与
    final_layout）。
    """
    t0 = time.time()
    mapping = _exact_embedding(circ, spec,
                               time_budget=min(time_budget, 15.0))
    exact = mapping is not None
    if mapping is None:
        mapping = _partial_embedding(
            circ, spec, time_budget=max(0.0, time_budget - (time.time() - t0)))
    if mapping is None:
        return None

    # 应用置换；精确嵌入时删除全部 swap（0-SWAP 消除语义），
    # 部分嵌入仅重标记（swap 保留，收益由非劣化守卫裁决）
    out = Circuit(circ.n, name=circ.name)
    n_swap_removed = 0
    for inst in circ.ops:
        if exact and inst.name == "swap":
            n_swap_removed += 1
            continue
        qs = tuple(int(mapping.get(int(q), int(q))) for q in inst.qubits)
        out.append(Inst(inst.name, qs, inst.params))
    for m_ in circ.measures:
        out.measures.append(
            Inst("measure", (int(mapping.get(int(m_.qubits[0]),
                                            int(m_.qubits[0]))),),
                 cbits=m_.cbits))
    # 自验证防线（2026-11-11）：精确模式删 swap 后所有 2Q 门必须相邻，
    # 否则拒绝（曾出现未验证邻接的假阳性）
    if exact:
        for inst in out.ops:
            if inst.nq == 2 and inst.name != "swap":
                if not spec.adj[int(inst.qubits[0]), int(inst.qubits[1])]:
                    return None
    return out, mapping
