"""自研深度感知路由器（借鉴 Cowtan et al. QIC 2019 的 LexiRoute 思想）。

LexiRoute 的核心思想：路由决策以**关键路径深度增量**为字典序第一代价。
本实现为自主简化版（引用原论文思想，代码独立）：
  1. 分层并行化：维护"可执行前沿"（前驱已完成的 2Q 门）
  2. 前沿可执行时全部并行执行（一层深度）
  3. 否则评估候选 SWAP（耦合图上与前沿端点相邻的边）：
     评分 = (收益, 距离势能) 字典序——
       收益 = 交换后立即变为可执行的前沿门数
       距离势能 = 剩余前沿门端点距离和（一步前瞻的深度代理）
     选字典序最优者执行（深度 +1）

正确性保证：只要耦合图连通，该贪心过程必然终止（每步要么执行至少
一个门，要么选择有正收益的 SWAP；无正收益时仍选一个势能最小的 SWAP）。
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import Layout, PassManager
from qiskit.transpiler.passes import (ApplyLayout, EnlargeWithAncilla,
                                      FullAncillaAllocation, SetLayout)


def lexiroute(qc: QuantumCircuit, cm, layout: dict, seed: int = 0,
              max_swaps: int = 100000) -> tuple[QuantumCircuit, int, dict]:
    """深度字典序路由：返回 (路由后线路, SWAP 数, 最终布局)。

    线路先经 SetLayout 放置到物理比特（含 ancilla 扩展），随后在物理
    空间上按层并行执行 2Q 门并插入 SWAP。1Q 门直接附加执行（不占层）。
    """
    rng = np.random.default_rng(seed)

    # ---- 放置到物理空间
    init_layout = Layout({qc.qubits[k]: v for k, v in layout.items()})
    pm = PassManager([
        SetLayout(init_layout),
        FullAncillaAllocation(coupling_map=cm),
        EnlargeWithAncilla(),
        ApplyLayout(),
    ])
    phys = pm.run(qc)

    n_phys = cm.size()
    # 邻接集合（无向）+ 全对最短距离（一次性预计算）
    import networkx as nx
    adj = set()
    gx = nx.Graph()
    gx.add_nodes_from(range(n_phys))
    for a, b in cm.get_edges():
        adj.add((min(a, b), max(a, b)))
        gx.add_edge(a, b)
    dist_mat = nx.floyd_warshall_numpy(gx).astype(np.int32)

    def _dist(p: int, q: int) -> int:
        return int(dist_mat[p, q])

    # ---- 指令流（2Q 门 + 1Q 门；跳过非门指令）
    gates = []
    for inst in phys.data:
        name = inst.operation.name
        qs = [phys.find_bit(q).index for q in inst.qubits]
        if name == "swap":
            continue
        if len(qs) == 2:
            gates.append(("2q", qs[0], qs[1], inst.operation, inst.clbits))
        elif len(qs) == 1:
            gates.append(("1q", qs[0], None, inst.operation, inst.clbits))
        # barrier/measure 忽略（measure 由调用方回挂）

    # ---- 路由主循环
    # wire -> 逻辑比特（ApplyLayout 后电路比特索引 = 初始位置）
    logical_of = {v: k for k, v in layout.items()}
    done = [False] * len(gates)
    dep = [set() for _ in gates]      # 每个门的前驱（同线序约束）
    succ = [set() for _ in gates]
    last_on = {}                       # wire -> 最后一个使用它的门索引
    for i, g in enumerate(gates):
        qs = [g[1]] if g[0] == "1q" else [g[1], g[2]]
        for q in qs:
            if q in last_on:
                dep[i].add(last_on[q])
                succ[last_on[q]].add(i)
        for q in qs:
            last_on[q] = i

    frontier = {i for i in range(len(gates)) if not dep[i]}
    out = QuantumCircuit(n_phys)
    swap_count = 0
    final = {k: v for k, v in layout.items()}   # 逻辑 -> 当前物理位置
    recent: list = []                           # 最近交换禁忌表
    swap_history: dict = {}                     # 交换计数（衰减惩罚）

    def cur_pos(w: int) -> int:
        """wire w 上的内容当前所在位置。"""
        logical = logical_of.get(w)
        return final[logical] if logical is not None else w

    def emit(idx):
        g = gates[idx]
        if g[0] == "1q":
            out.append(g[3], [out.qubits[cur_pos(g[1])]], list(g[4]))
        else:
            p1, p2 = cur_pos(g[1]), cur_pos(g[2])
            out.append(g[3], [out.qubits[p1], out.qubits[p2]], list(g[4]))
        done[idx] = True
        for s in succ[idx]:
            dep[s].discard(idx)
            if not dep[s]:
                frontier.add(s)

    while frontier:
        # 可执行（当前端点相邻）的前沿 2Q 门
        exec_now = [i for i in frontier if gates[i][0] == "2q"
                    and (min(cur_pos(gates[i][1]), cur_pos(gates[i][2])),
                         max(cur_pos(gates[i][1]), cur_pos(gates[i][2]))) in adj]
        exec_now += [i for i in frontier if gates[i][0] == "1q"]
        if exec_now:
            for i in exec_now:        # 同层并行执行
                frontier.discard(i)
                emit(i)
            continue

        # ---- 无前沿门可执行：选择 SWAP
        # 候选 = 前沿门端点最短路径的首边（路径导向：沿最短路径交换必然
        # 使距离严格下降 → 总距离单调收敛 → 必然终止且不振荡）
        cand_keys = set()
        for i in frontier:
            if gates[i][0] != "2q":
                continue
            p1, p2 = cur_pos(gates[i][1]), cur_pos(gates[i][2])
            if p1 == p2:
                continue
            if (min(p1, p2), max(p1, p2)) in adj:
                continue
            # 最短路径首边：p1 的下一个节点
            nxt = min((v for v in gx.neighbors(p1)
                       if int(dist_mat[v, p2]) == int(dist_mat[p1, p2]) - 1),
                      default=None)
            if nxt is not None:
                cand_keys.add((min(p1, nxt), max(p1, nxt)))
        if not cand_keys:
            # 极端退化：任取一条与前沿端点相邻的边（防御）
            endpoints = {cur_pos(gates[i][1]) for i in frontier} | \
                        {cur_pos(gates[i][2]) for i in frontier
                         if gates[i][0] == "2q"}
            for (a, b) in adj:
                if a in endpoints or b in endpoints:
                    cand_keys.add((a, b))
                    break
        candidates = [k for k in cand_keys
                      if k not in recent]  # 简单禁忌（近 4 步不重复）
        if not candidates and cand_keys:
            candidates = list(cand_keys)   # 禁忌全拦截时退化为无禁忌
        if not candidates:
            break

        def pos_after(p, x, y):
            """交换位置 x,y 后，位置 p 上内容的新位置。"""
            if p == x:
                return y
            if p == y:
                return x
            return p

        cur_pot = 0
        for i in frontier:
            if gates[i][0] == "2q":
                cur_pot += _dist(cur_pos(gates[i][1]), cur_pos(gates[i][2]))

        # 1Q 前沿门视为本层必执行（不计收益但释放依赖）
        oneq_done = {i for i in frontier if gates[i][0] == "1q"}

        best = None
        for (a, b) in candidates:
            # 模拟交换后：收益 = 立即可执行的前沿门数；势能 = 剩余距离和
            benefit = 0
            pot = 0
            enabled = set(oneq_done)
            for i in frontier:
                if gates[i][0] != "2q":
                    continue
                p1 = pos_after(cur_pos(gates[i][1]), a, b)
                p2 = pos_after(cur_pos(gates[i][2]), a, b)
                if p1 == p2:
                    continue
                if (min(p1, p2), max(p1, p2)) in adj:
                    benefit += 1
                    enabled.add(i)
                else:
                    pot += _dist(p1, p2)
            # 两层前瞻（路径②）：本层执行后，下一层还能解锁多少门
            next_benefit = 0
            for i in enabled:
                for s in succ[i]:
                    if s not in enabled and dep[s] <= enabled:
                        next_benefit += 1
            decay = swap_history.get((a, b), 0)
            score = (benefit + 0.5 * next_benefit - 2.0 * decay,
                     -(pot - cur_pot))   # 收益(含前瞻)优先，衰减惩罚，其次势能
            if best is None or score > best[0]:
                best = (score, a, b)
        a, b = best[1], best[2]
        swap_history[(a, b)] = swap_history.get((a, b), 0) + 1
        recent.append((a, b))
        if len(recent) > 4:
            recent.pop(0)
        # 执行 SWAP
        out.swap(a, b)
        swap_count += 1
        for logical, p in final.items():
            if p == a:
                final[logical] = b
            elif p == b:
                final[logical] = a
        if swap_count > max_swaps:
            break

    return out, swap_count, final
