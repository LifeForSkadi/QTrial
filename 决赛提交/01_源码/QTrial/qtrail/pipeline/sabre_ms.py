"""SABRE-MS 式路由器（自主实现，借鉴 2026 年 SABRE-MS 思想）。

在完整 SABRE 机制（front layer 批处理 + extended set 前瞻 + decay 防振荡）
之上，评分函数加入 makespan（调度深度）项：
    score(swap) = basic_cost + w_ext·ext_cost + λ_ms·makespan_delta − decay_penalty
    basic_cost    = 交换后前沿门端点距离和（SWAP 目标）
    ext_cost      = 交换后扩展集门端点距离和（前瞻）
    makespan_delta= 1（交换本身占一层）− 并行收益（解锁门数/前沿规模）
    λ_ms=0 还原为 SABRE 原式（SWAP 目标）；λ_ms>0 注入深度目标

论文结论（SABRE-MS）：收益来自目标函数而非机制——本实现即该结论的
自主验证与落地。
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import Layout, PassManager
from qiskit.transpiler.passes import (ApplyLayout, EnlargeWithAncilla,
                                      FullAncillaAllocation, SetLayout)


def sabre_ms_route(qc: QuantumCircuit, cm, layout: dict, seed: int = 0,
                   lam_ms: float = 0.5, w_ext: float = 0.5,
                   decay_pen: float = 0.5, max_swaps: int = 100000):
    """SABRE-MS 路由：返回 (路由后线路, SWAP 数, 最终布局)。

    机制：front layer 并行执行 + extended set + decay；
    目标：SWAP（距离和）+ λ_ms·makespan（SABRE-MS 扩展）。
    """
    rng = np.random.default_rng(seed)
    # ---- 放置到物理空间
    init = Layout({qc.qubits[k]: v for k, v in layout.items()})
    pm = PassManager([
        SetLayout(init),
        FullAncillaAllocation(coupling_map=cm),
        EnlargeWithAncilla(),
        ApplyLayout(),
    ])
    phys = pm.run(qc)

    n_phys = cm.size()
    import networkx as nx
    gx = nx.Graph()
    gx.add_nodes_from(range(n_phys))
    gx.add_edges_from([(min(a, b), max(a, b)) for a, b in cm.get_edges()])
    dist_mat = nx.floyd_warshall_numpy(gx).astype(np.int32)
    adj = set(gx.edges)

    # ---- 指令流与依赖（同线序）
    gates = []
    for inst in phys.data:
        name = inst.operation.name
        if name in ("barrier", "measure", "reset"):
            continue
        qs = [phys.find_bit(q).index for q in inst.qubits]
        if len(qs) == 2:
            gates.append(("2q", qs[0], qs[1], inst.operation, list(inst.clbits)))
        elif len(qs) == 1:
            gates.append(("1q", qs[0], None, inst.operation, list(inst.clbits)))

    dep = [set() for _ in gates]
    succ = [set() for _ in gates]
    last = {}
    for i, g in enumerate(gates):
        wires = [g[1]] if g[0] == "1q" else [g[1], g[2]]
        for w in wires:
            if w in last:
                dep[i].add(last[w])
                succ[last[w]].add(i)
            last[w] = i

    pos = {k: v for k, v in layout.items()}   # 逻辑 -> 物理
    frontier = {i for i in range(len(gates)) if not dep[i]}
    executed = set()
    out = QuantumCircuit(n_phys)
    for creg in qc.cregs:
        out.add_register(creg)
    depth = 0
    swap_count = 0
    history = {}

    def d2(p1, p2):
        return int(dist_mat[p1, p2])

    def cur(i):
        g = gates[i]
        if g[0] == "1q":
            return pos[g[1]], pos[g[1]]
        return pos[g[1]], pos[g[2]]

    def after(p, a, b):
        return b if p == a else (a if p == b else p)

    while frontier:
        # front layer：并行执行可执行门
        exec_now = [i for i in frontier
                    if gates[i][0] == "1q"
                    or (lambda p: (min(*p), max(*p)) in adj)(cur(i))]
        if exec_now:
            for i in exec_now:
                g = gates[i]
                frontier.discard(i)
                executed.add(i)
                if g[0] == "1q":
                    out.append(g[3], [out.qubits[pos[g[1]]]], g[4])
                else:
                    p1, p2 = pos[g[1]], pos[g[2]]
                    out.append(g[3], [out.qubits[p1], out.qubits[p2]], g[4])
                for s in succ[i]:
                    dep[s].discard(i)
                    if not dep[s]:
                        frontier.add(s)
            depth += 1
            continue

        # ---- 无门可执行：SABRE-MS 评分选交换
        # extended set（SABRE 定义）：前沿执行后依赖立即满足的下一层门
        allowed = frontier | executed
        ext = {s for i in frontier for s in succ[i]
               if s not in frontier and dep[s] <= allowed}
        ext = {i for i in ext if gates[i][0] == "2q"}

        # 候选 = 前沿门端点邻边（随机试验机制经实测有害，已回退）
        candidates = set()
        eps = set()
        for i in frontier:
            if gates[i][0] != "2q":
                continue
            p1, p2 = pos[gates[i][1]], pos[gates[i][2]]
            eps.add(p1)
            eps.add(p2)
        for (a, b) in adj:
            if a in eps or b in eps:
                candidates.add((a, b))

        best = None
        for (a, b) in candidates:
            # 模拟交换后的代价
            basic = 0.0
            for i in frontier:
                if gates[i][0] != "2q":
                    continue
                p1, p2 = after(pos[gates[i][1]], a, b), after(pos[gates[i][2]], a, b)
                basic += d2(p1, p2)
            ext_cost = 0.0
            for i in ext:
                p1, p2 = after(pos[gates[i][1]], a, b), after(pos[gates[i][2]], a, b)
                ext_cost += d2(p1, p2)
            # makespan 项：交换占 1 层，减去并行收益（解锁门占比）
            unlocked = sum(1 for i in frontier
                           if gates[i][0] == "2q"
                           and (lambda p: (min(*p), max(*p)) in adj)(
                               (after(pos[gates[i][1]], a, b),
                                after(pos[gates[i][2]], a, b))))
            ms_delta = 1.0 - unlocked / max(len(frontier), 1)
            penalty = history.get((a, b), 0) * decay_pen
            score = basic + w_ext * ext_cost + lam_ms * ms_delta + penalty
            if best is None or score < best[0]:
                best = (score, a, b)
        a, b = best[1], best[2]
        out.swap(a, b)
        swap_count += 1
        depth += 1
        history[(a, b)] = history.get((a, b), 0) + 1
        for logical, p in pos.items():
            if p == a:
                pos[logical] = b
            elif p == b:
                pos[logical] = a
        if swap_count > max_swaps:
            break

    return out, swap_count, pos
