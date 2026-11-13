# -*- coding: utf-8 -*-
"""全局重标记（对标 qiskit VF2PostLayout 的自主实现 + 噪声加权扩展，
2026-11-11）。

机制：
  1. 输入 = 坐标稳定线路（post_v2.fold_only 的产物：swap 已按 arr
     坐标跟踪删除，全部门在初始布局坐标下——固定置换作用于坐标
     稳定的线路才是合法的全局重标记；对含 swap 的时变线路直接施加
     固定置换是语义错误，2026-11-11 语义测试捕获）；
  2. 交互图 H = 输入线路 2Q 门的位置支撑图；
  3. 精确嵌入：H 是否子图同构嵌入设备耦合图（可用节点排除禁用比特）？
     networkx GraphMatcher（VF2 公开算法 Cordella 2001 的标准实现，
     限时）——找到则置换重标记（0-SWAP 消除在 fold_only 上游完成）；
  4. 未找到（限时/无解）→ 噪声加权部分嵌入：活跃位置内局部搜索，
     最大化 Σ_{H 边} (1-ε) · [π 后落在设备边上]——qiskit 无此扩展
     （他们找不到完美解就放弃）；
  5. 自验证防线 + 调用方非劣化守卫。

语义：全局标签置换保线路酉语义（输出比特重命名），final_layout
由调用方同步更新（fold_only 的 arr 与 mapping 复合）。
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


def _exact_embedding_networkx(circ, spec, time_budget: float = 15.0,
                              max_active: int = 20):
    """旧精确路径（networkx GraphMatcher，≤20 活跃；A/B 与测试对照）。

    2026-11-13 第 6 项后不再是默认：无搜索内剪枝、>20 直接放弃。
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
    usable = [q for q in range(spec.n)
              if q not in set(spec.disabled_qubits)]
    G = nx.Graph()
    G.add_nodes_from(usable)
    G.add_edges_from([(i, j) for i in usable for j in usable
                      if i < j and spec.adj[i, j]])
    deg_H = sorted((d for _, d in H.degree()), reverse=True)
    deg_G = sorted((d for _, d in G.degree()), reverse=True)
    if any(dh > deg_G[i] for i, dh in enumerate(deg_H)):
        return None
    GM = nx.algorithms.isomorphism.GraphMatcher(G, H)
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
    t0 = time.time()
    for iso in GM.subgraph_isomorphisms_iter():
        if time.time() - t0 > time_budget:
            break
        mapping = {h: int(g) for g, h in iso.items()}
        s = score(mapping)
        if s > best_s:
            best_s = s
            best = mapping
    return best


def _exact_embedding(circ, spec, time_budget: float = 15.0):
    """H 的支撑图嵌入设备图：返回置换 dict {旧位置: 新位置} 或 None。

    嵌入语义：把活跃位置（含 2Q 门的位置）重命名到设备节点上，使
    全部 H 边落在设备边上。

    2026-11-13 执行顺序第 6 项：精确路径改为**自研 VF2++**
    （qtrail/pure/vf2pp.py）——VF2 五规则 + VF2++ 节点排序 + 搜索内
    增量打分（log 空间，含 1Q/读出项）+ 分支限界 + 递归内硬预算
    （call_limit / max_solutions / wall）。旧 networkx GraphMatcher
    路径只枚举、无预算内剪枝且 >20 活跃节点直接放弃；保留为测试对照
    （tests/test_vf2pp.py 的 ≤20 最优分数等价性验证）。
    开关 QTRAIL_VF2_IMPL=networkx 可切回旧路径（A/B/复现用）。

    **无节点数上限（2026-11-13 QFT/graphstate 归因修正）**：此前
    管线层硬门槛 max_active=40 是"graphstate_50 白耗预算"的过度反应
    ——graphstate_100（100 活跃、100 边、最大度 2）实际可精确嵌入
    （Qiskit SabreLayout 的 initial layout 即 0-SWAP 见证），旧门槛
    直接跳过导致 2Q 652 vs 100。现改为：
      1) 必要条件零成本可证伪：H 的边数 > 设备边数 → 必然不可嵌入；
         度序列支配在 find_embedding 内已检查；
      2) 搜索本身只受 VF2Config 硬预算（calls / solutions / 墙钟）约束，
         不再按节点数拒绝——稠密线路由条件 1 直接淘汰，稀疏线路
         （graphstate/ghz/bv 类）获得精确嵌入机会。
    候选排序改为 Warnsdorff 结构优先（见 vf2pp.VF2Config.cand_order），
    同实例从 60s 无解降至 0.01s。
    """
    import os
    if os.environ.get("QTRAIL_VF2_IMPL", "vf2pp").lower() == "networkx":
        return _exact_embedding_networkx(
            circ, spec, time_budget=time_budget, max_active=20)
    from qtrail.pure.vf2pp import VF2Config, find_embedding
    edges = _interaction_graph(circ)
    if not edges:
        return None
    active = sorted({p for e in edges for p in e})
    # 必要条件（零成本可证伪）：子图边数不能超过设备边数
    n_dev_edges = int(np.asarray(spec.adj).sum()) // 2
    if len(edges) > n_dev_edges:
        return None
    usable = [q for q in range(spec.n)
              if q not in set(spec.disabled_qubits)]
    err2q = spec.calib.err_2q
    med = float(np.median(list(err2q.values()))) if err2q else 1e-3
    # 1Q/读出项：活跃位置上的门数/测量数（评审 §3.2 的"只按 2Q
    # 加权"缺陷修复；与管线对数保真度同口径）
    w1q = {p: 0 for p in active}
    for inst in circ.ops:
        if inst.nq == 1:
            q = int(inst.qubits[0])
            if q in w1q:
                w1q[q] += 1
    wro = {p: 0 for p in active}
    for m in circ.measures:
        q = int(m.qubits[0])
        if q in wro:
            wro[q] += 1
    cfg = VF2Config(wall_limit=min(8.0, max(0.0, time_budget)))
    mapping, _score = find_embedding(
        active, edges, spec.adj, usable,
        spec.calib.err_1q, spec.calib.err_ro, err2q, med,
        w1q, wro, cfg)
    return mapping


def _partial_embedding(circ, spec, time_budget: float = 10.0,
                       max_iters: int = 5000):
    """噪声加权部分嵌入：活跃位置内局部搜索，最大化 Σ w_e·(1-ε)·[可嵌入]。

    移动 = 交换两个活跃位置的新标签；增量评估；首优爬山 + 随机重启。
    返回置换 dict 或 None（无改进时）。

    预算 = 墙钟时间（2026-11-12 用户决定改回）+ 迭代上限兜底。
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
    n_active = len(active)
    t0 = time.time()
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


def _bipartite_conflicts(nodes, edges):
    """BFS 二着色返回同色边（proto_edge_delete.py 同款，2026-11-13）。

    注意：同一条奇环上删哪条边会改变 H' 的节点标签，进而影响 VF2
    搜索成败（graphstate_105 实测：删 (8,45) 8s 内找到，删 (26,49)
    超时）——调用方应依次尝试多个冲突边。
    """
    import collections
    adj = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    color = {}
    bad = []
    for start in nodes:
        if start in color:
            continue
        color[start] = 0
        dq = collections.deque([start])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    dq.append(v)
                elif color[v] == color[u]:
                    e = (u, v) if u < v else (v, u)
                    if e not in bad:
                        bad.append(e)
    return bad


def sparse_cycle_layout(circ: Circuit, spec, time_budget: float = 8.0):
    """稀疏**非二分**低度交互图（典型：奇数环）的结构布局先验。

    动机（2026-11-13）：graphstate_105 是 105 节点奇环，7×15 设备
    二分（53/52）→ 精确嵌入数学上不可能；RL 布局 + SABRE 产生
    192 swaps（2Q 681）。本函数在**路由前**做"删一条同色边 → 路径
    嵌入"，把唯一长程门交给 SABRE 链化：实测 5 swaps、2Q 120、
    2Q 深度 23（优于 O3 的 30）。

    仅当交互图最大度 ≤2 且非二分时返回布局（pi 数组，逻辑→物理）；
    其余情况返回 None，生产行为完全不变。

    Returns: np.ndarray[int] 长度 circ.n，或 None。
    """
    from qtrail.pure.vf2pp import VF2Config, find_embedding

    edges = _interaction_graph(circ)
    if not edges:
        return None
    # 小线路不启用：布局空间小、SABRE 已接近最优；实测 toffoli_n3
    # 会出现 2Q 9→12 的轻微回归（2026-11-13 A/B），故仅 n≥10 启用。
    if circ.n < 10:
        return None
    H_adj = {}
    for a, b in edges:
        H_adj.setdefault(a, set()).add(b)
        H_adj.setdefault(b, set()).add(a)
    if max(len(v) for v in H_adj.values()) > 2:
        return None
    active = sorted(H_adj)
    conflicts = _bipartite_conflicts(active, edges.keys())
    if not conflicts:
        return None  # H 二分：交给常规布局/重标记
    usable = [q for q in range(spec.n)
              if q not in set(spec.disabled_qubits)]
    err2q = spec.calib.err_2q
    med = float(np.median(list(err2q.values()))) if err2q else 1e-3
    w1q = {p: 0 for p in active}
    for inst in circ.ops:
        if inst.nq == 1:
            q = int(inst.qubits[0])
            if q in w1q:
                w1q[q] += 1
    wro = {p: 0 for p in active}
    for m in circ.measures:
        q = int(m.qubits[0])
        if q in wro:
            wro[q] += 1
    # 删哪条同色边会改变 H' 标签、影响 VF2 搜索成败——依次尝试
    # （graphstate_105 实测：删 (8,45) 找到，删 (26,49) 超时）。
    mapping = None
    if len(conflicts) > 1:
        cands = [set(conflicts)] + [{c} for c in conflicts[:3]]
    else:
        cands = [set(conflicts)]
    per_try = max(0.5, min(8.0, time_budget))
    for bad in cands:
        h2 = {e: n for e, n in edges.items() if e not in bad}
        mapping, _score = find_embedding(
            active, h2, spec.adj, usable,
            spec.calib.err_1q, spec.calib.err_ro, err2q, med,
            w1q, wro, VF2Config(wall_limit=per_try))
        if mapping is not None:
            break
    if mapping is None:
        return None
    pi = np.full(circ.n, -1, dtype=int)
    for a, b in mapping.items():
        pi[int(a)] = int(b)
    used = set(int(x) for x in pi[pi >= 0])
    free_phys = [q for q in usable if q not in used]
    free_log = [i for i in range(circ.n) if pi[i] < 0]
    for i, q in zip(free_log, free_phys):
        pi[i] = int(q)
    if any(int(x) < 0 for x in pi):
        return None
    return pi


def _complete_permutation(mapping: dict, n: int) -> dict:
    """把活跃位置的部分置换补全为 [0, n) 上的双射。

    嵌入只给出活跃位置的映射；若未映射位置保持恒等会与映射目标
    撞车（两个逻辑比特输出身份冲突，语义损坏）——补全规则：
    **活跃映射优先**，未映射位置让出被占用的目标、分配剩余空闲
    目标（优先恒等，否则任意）。
    """
    full = {int(k): int(v) for k, v in mapping.items()}
    used = set(full.values())
    free_sources = [p for p in range(n) if p not in full]
    free_targets = [t for t in range(n) if t not in used]
    for p in free_sources:
        if p in free_targets:
            full[p] = p
            free_targets.remove(p)
    for p in free_sources:
        if p not in full:
            full[p] = free_targets.pop()
    return full


def global_relabel(circ: Circuit, spec, time_budget: float | None = None) \
        -> tuple[Circuit, dict, bool] | None:
    """全局重标记。返回 (新线路, 置换, 是否精确嵌入) 或 None。

    置换 dict：{旧物理位: 新物理位}（作用于线路全部标签与
    final_layout）。exact=False 时线路中可能残留非相邻 2Q 门
    （部分嵌入只改善边质量、不保证合法性）——调用方仅应在本组件
    为 exact 时采用，或自行链化长程门。

    time_budget 为墙钟预算（秒，默认 25；2026-11-12 用户决定从迭代
    计数改回）：精确嵌入优先 15s，余量给部分嵌入。
    """
    if time_budget is None:
        time_budget = 25.0
    t0 = time.time()
    mapping = _exact_embedding(circ, spec,
                               time_budget=min(time_budget, 15.0))
    exact = mapping is not None
    if mapping is None:
        mapping = _partial_embedding(
            circ, spec, time_budget=max(0.0, time_budget - (time.time() - t0)))
    if mapping is None:
        return None

    # 补全为全域双射（防输出身份冲突；2026-11-11 语义测试捕获）
    mapping = _complete_permutation(mapping, spec.n)

    # 应用置换；精确嵌入时删除全部 swap（0-SWAP 消除语义），
    # 部分嵌入仅重标记（swap 保留，收益由非劣化守卫裁决）
    out = Circuit(circ.n, name=circ.name)
    n_swap_removed = 0
    for inst in circ.ops:
        if exact and inst.name == "swap":
            n_swap_removed += 1
            continue
        qs = tuple(int(mapping[int(q)]) for q in inst.qubits)
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
    return out, mapping, exact
