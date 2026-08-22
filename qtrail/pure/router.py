"""自研 SABRE 级路由器（qiskit-free，公开论文算法 ASPLOS'19 的自主复现）。

机制完整性：front layer 并行执行 + extended set 前瞻 + **逐门衰减权重**
（w_g = 0.5^t_g，SABRE 核心）+ 交换历史惩罚；可选 makespan 项
（SABRE-MS 思想，λ_ms=0 还原纯 SABRE 目标）。

性能：候选评分向量化（numpy 广播 [候选×前沿] 距离矩阵），50 比特级
线路秒级（2026-08-21 优化：原纯 Python 循环 qaoa_50 994s → 目标 <30s）。
2026-08-22 增量化（纯工程加速，评分函数与选择逻辑逐位等价）：
  1. 前沿/扩展集增量维护——allowed=frontier∪executed 单调增长，门执行
     时刻即 deps[s]⊆allowed 翻转时刻，exec 时更新、交换迭代零重建；
  2. 候选边由邻接表构建（排序遍历 → 与原全量扫描同序 → argmin 决胜一致）；
  3. 交换历史 numpy 矩阵（dict 等价）；4. inv 映射 O(1) 位置更新；
  5. 候选×前沿距离矩阵 Numba 融合内核（swap_kernel，无中间 [C,F]
     分配；dist==1 ⟺ 相邻替代 amat 命中，逐位一致）。
另有 2026-08-22 deps 防御性拷贝修复：路由消耗的依赖集不再污染调用方
线路对象（mapper 候选评分对同一 circ 多次调用，此前第 2 次起 DAG
残缺、评分失真）。
正确性由态矢量等价测试 + v1/v2 逐位对比保证。
"""
from __future__ import annotations

import numpy as np

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.swap_kernel import after_dist_matrix


def sabre_route(circ: Circuit, spec, layout: dict, seed: int = 0,
                lam_ms: float = 0.0, w_ext: float = 0.5,
                decay_pen: float = 0.5, max_swaps: int = 100000):
    """返回 (routed: Circuit[物理空间], swap_count, final_layout)。"""
    rng = np.random.default_rng(seed)
    ops = circ.ops
    m = len(ops)

    # 邻接布尔矩阵 + 距离矩阵 + 有序邻接表（numpy / 列表）
    amat = np.zeros((spec.n, spec.n), dtype=bool)
    nbrs = [[] for _ in range(spec.n)]
    for a in range(spec.n):
        for b in range(a + 1, spec.n):
            if spec.adj[a, b]:
                amat[a, b] = amat[b, a] = True
                nbrs[a].append(b)
                nbrs[b].append(a)
    dist = spec.dist.astype(np.int64)
    dist_flat = dist.ravel()

    # 防御性拷贝：路由过程就地消耗依赖集；mapper 候选评分会对同一 circ
    # 对象反复调用本函数，拷贝保证每次调用看到完整 DAG（2026-08-22 修复）。
    deps = [set(d) for d in circ.deps()]
    succ = [set() for _ in range(m)]
    for i in range(m):
        for j in deps[i]:
            succ[j].add(i)

    pos = {k: int(v) for k, v in layout.items()}
    inv = {v: k for k, v in pos.items()}
    frontier = {i for i in range(m) if not deps[i]}
    executed = set()
    allowed = set(frontier)
    # 扩展集增量维护：仅当 deps[s]⊆allowed 由假变真（某依赖执行）或
    # s 进入 frontier 时更新，与逐次全量重算成员逐位一致。
    ext = set()
    for i in frontier:
        for s in succ[i]:
            if s not in frontier and deps[s] <= allowed \
                    and len(ops[s].qubits) == 2:
                ext.add(s)
    gate_decay = [0] * m
    hist = np.zeros((spec.n, spec.n), dtype=np.int64)
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
                        allowed.add(s)
                        ext.discard(s)
                        # s 进入前沿 → 其后继的 deps⊆allowed 可能此刻才成立
                        for t in succ[s]:
                            if t not in frontier and deps[t] <= allowed \
                                    and len(ops[t].qubits) == 2:
                                ext.add(t)
                    elif deps[s] <= allowed and len(ops[s].qubits) == 2:
                        ext.add(s)
            continue

        # ---- 无门可执行：向量化评分选交换
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
        # 与全量扫描 (a 升序, b 升序) 同序的候选边：邻接表双向收集后排序
        pairs = set()
        for a in eps:
            for b in nbrs[a]:
                pairs.add((a, b) if b > a else (b, a))
        cand = sorted(pairs)
        if not cand:
            break
        ea_ = np.array([a for a, _ in cand], dtype=np.int64)
        eb_ = np.array([b for _, b in cand], dtype=np.int64)

        # 融合内核：直接产出 [候选×前沿] 距离矩阵（与 after_vec+花式索引
        # 逐位一致）；dist==1 ⟺ 相邻（无权 BFS 距离），等价 amat 命中
        Df = after_dist_matrix(dist_flat, spec.n, fq1, fq2, ea_, eb_)
        basic = (fw[None, :] * Df).sum(axis=1)
        unlocked = (Df == 1).sum(axis=1)
        De = after_dist_matrix(dist_flat, spec.n, eq1, eq2, ea_, eb_)
        ext_cost = (ew[None, :] * De).sum(axis=1)
        ms_delta = 1.0 - unlocked / max(len(fq1), 1)
        pen = hist[ea_, eb_] * decay_pen
        scores = basic + w_ext * ext_cost + lam_ms * ms_delta + pen
        best = int(np.argmin(scores))
        a, b = int(ea_[best]), int(eb_[best])
        out.append(Inst("swap", (a, b)))
        swap_count += 1
        hist[a, b] += 1
        for i in frontier:
            if len(ops[i].qubits) == 2:
                p1, p2 = pos[ops[i].qubits[0]], pos[ops[i].qubits[1]]
                if p1 == a or p1 == b or p2 == a or p2 == b:
                    gate_decay[i] += 1
        for i in ext:
            p1, p2 = pos[ops[i].qubits[0]], pos[ops[i].qubits[1]]
            if p1 == a or p1 == b or p2 == a or p2 == b:
                gate_decay[i] += 1
        la, lb = inv.get(a), inv.get(b)
        if la is not None:
            del inv[a]
            pos[la] = b
        if lb is not None:
            del inv[b]
            pos[lb] = a
            inv[a] = lb
        if la is not None:
            inv[b] = la
        if swap_count > max_swaps:
            break

    for inst in circ.measures:
        out.append(Inst("measure", (pos[inst.qubits[0]],), cbits=inst.cbits))
    out.strip_ids()
    return out, swap_count, pos
