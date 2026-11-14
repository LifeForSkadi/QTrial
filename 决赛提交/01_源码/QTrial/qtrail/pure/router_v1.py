"""路由器 v1 冻结版——A/B 回归门禁的逐位等价基准（scripts/ab_router_v2.py）。
勿修改本文件；对 router.py 的任何优化必须与 v1 输出逐位一致。

自研 SABRE 级路由器（qiskit-free，公开论文算法 ASPLOS'19 的自主复现）。

机制完整性：front layer 并行执行 + extended set 前瞻 + **逐门衰减权重**
（w_g = 0.5^t_g，SABRE 核心）+ 交换历史惩罚；可选 makespan 项
（SABRE-MS 思想，λ_ms=0 还原纯 SABRE 目标）。

性能：候选评分向量化（numpy 广播 [候选×前沿] 距离矩阵），50 比特级
线路秒级（2026-08-21 优化：原纯 Python 循环 qaoa_50 994s → 目标 <30s）。
正确性由态矢量等价测试保证。
"""
from __future__ import annotations

import numpy as np

from qtrail.pure.circuit import Circuit, Inst


def sabre_route(circ: Circuit, spec, layout: dict, seed: int = 0,
                lam_ms: float = 0.0, w_ext: float = 0.5,
                decay_pen: float = 0.5, max_swaps: int = 100000):
    """返回 (routed: Circuit[物理空间], swap_count, final_layout)。"""
    rng = np.random.default_rng(seed)
    ops = circ.ops
    m = len(ops)

    # 邻接布尔矩阵 + 距离矩阵（numpy）
    amat = np.zeros((spec.n, spec.n), dtype=bool)
    for a in range(spec.n):
        for b in range(a + 1, spec.n):
            if spec.adj[a, b]:
                amat[a, b] = amat[b, a] = True
    dist = spec.dist.astype(np.int64)

    # A/B 基准同款修复：防御性拷贝 deps（避免跨调用污染）
    deps = [set(d) for d in circ.deps()]
    succ = [set() for _ in range(m)]
    for i in range(m):
        for j in deps[i]:
            succ[j].add(i)

    pos = {k: int(v) for k, v in layout.items()}
    frontier = {i for i in range(m) if not deps[i]}
    executed = set()
    gate_decay = [0] * m
    history = {}
    out = Circuit(spec.n, name=circ.name + "_routed")
    swap_count = 0

    def front_arrays(fset, use_ext_weights=True):
        """前沿 2Q 门的 (q1pos, q2pos, weights) numpy 数组。"""
        idx = [i for i in fset if len(ops[i].qubits) == 2]
        if not idx:
            return None
        q1 = np.array([pos[ops[i].qubits[0]] for i in idx], dtype=np.int64)
        q2 = np.array([pos[ops[i].qubits[1]] for i in idx], dtype=np.int64)
        w = np.array([(0.5 ** gate_decay[i]) if use_ext_weights else 1.0
                      for i in idx], dtype=np.float64)
        return q1, q2, w

    while frontier:
        exec_now = [i for i in frontier
                    if len(ops[i].qubits) == 1
                    or amat[pos[ops[i].qubits[0]], pos[ops[i].qubits[1]]]]
        if exec_now:
            for i in exec_now:
                frontier.discard(i)
                executed.add(i)
                inst = ops[i]
                if inst.nq == 1:
                    out.append(Inst(inst.name, (pos[inst.qubits[0]],),
                                    inst.params))
                else:
                    p1, p2 = pos[inst.qubits[0]], pos[inst.qubits[1]]
                    out.append(Inst(inst.name, (p1, p2), inst.params))
                for s in succ[i]:
                    deps[s].discard(i)
                    if not deps[s]:
                        frontier.add(s)
            continue

        # ---- 无门可执行：向量化评分选交换
        allowed = frontier | executed
        ext = {s for i in frontier for s in succ[i]
               if s not in frontier and deps[s] <= allowed}
        ext = {i for i in ext if len(ops[i].qubits) == 2}

        f_arr = front_arrays(frontier)
        if f_arr is None:
            break
        fq1, fq2, fw = f_arr
        e_arr = front_arrays(ext)
        eq1, eq2, ew = (e_arr if e_arr is not None else
                        (np.zeros(0, dtype=np.int64),
                         np.zeros(0, dtype=np.int64),
                         np.zeros(0, dtype=np.float64)))

        eps = set(fq1.tolist()) | set(fq2.tolist())
        cand = [(a, b) for a in range(spec.n) for b in range(a + 1, spec.n)
                if amat[a, b] and (a in eps or b in eps)]
        if not cand:
            break
        ea_ = np.array([a for a, _ in cand], dtype=np.int64)
        eb_ = np.array([b for _, b in cand], dtype=np.int64)
        C = len(cand)

        def after_vec(p, ea, eb):
            """p [F] → [C, F]：交换 (ea,eb) 后的位置。"""
            p = p[None, :]
            return np.where(p == ea[:, None], eb[:, None],
                            np.where(p == eb[:, None], ea[:, None], p))

        d1 = after_vec(fq1, ea_, eb_)
        d2_ = after_vec(fq2, ea_, eb_)
        basic = (fw[None, :] * dist[d1, d2_]).sum(axis=1)
        e1 = after_vec(eq1, ea_, eb_)
        e2 = after_vec(eq2, ea_, eb_)
        ext_cost = (ew[None, :] * dist[e1, e2]).sum(axis=1)
        unlocked = amat[d1, d2_].sum(axis=1)
        ms_delta = 1.0 - unlocked / max(len(fq1), 1)
        pen = np.array([history.get((int(a), int(b)), 0) * decay_pen
                        for a, b in cand], dtype=np.float64)
        scores = basic + w_ext * ext_cost + lam_ms * ms_delta + pen
        best = int(np.argmin(scores))
        a, b = int(ea_[best]), int(eb_[best])
        out.append(Inst("swap", (a, b)))
        swap_count += 1
        history[(a, b)] = history.get((a, b), 0) + 1
        for i in list(frontier) + list(ext):
            if len(ops[i].qubits) == 2:
                p1, p2 = pos[ops[i].qubits[0]], pos[ops[i].qubits[1]]
                if p1 in (a, b) or p2 in (a, b):
                    gate_decay[i] += 1
        for logical, p in pos.items():
            if p == a:
                pos[logical] = b
            elif p == b:
                pos[logical] = a
        if swap_count > max_swaps:
            break

    for inst in circ.measures:
        out.append(Inst("measure", (pos[inst.qubits[0]],), cbits=inst.cbits))
    out.strip_ids()
    return out, swap_count, pos
