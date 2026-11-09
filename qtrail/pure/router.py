"""自研 SABRE 路由器（qiskit-free，公开论文算法 ASPLOS'19 的自主复现）。

机制完整性：front layer 并行执行 + extended set 前瞻 + 交换历史惩罚；
可选 makespan 项（SABRE-MS 思想，λ_ms=0 还原纯 SABRE 目标）。

2026-11-10 评分函数对齐（对照 qiskit LightSABRE decay 启发式逐项核对）：
  1. 扩展集权重 0.5/spec.n（原 0.5 高估远期门，评分被扩展集主导）
  2. 门权重恒为 1（原 0.5^t 衰减到 0 后评分退化为平局，配合确定性
     argmin 会陷入换位循环——gf2^16 实测 10 万 swap 触顶死循环）
  3. 无界交换历史惩罚 → 有界按边禁忌（最近 tabu_window 次交换中使用过
     的物理边 ×tabu_mult，qiskit decay(0.001, 5) 的等价物；按比特惩罚
     过严会摧毁启发式信号——2026-11-10 诊断实测）
  4. 平局裁决改为种子随机（qiskit SabreSwap 的 seed 即为此用）
  5. attempt_limit=10n 防卡死逃逸（连续 10n 次无门执行的交换后随机换边）
  6. trials 参数：多种子路由取 swap 最少者（qiskit 预设 O3 = 20 trials）
  7. lookahead 前瞻（默认 0.5，SABRE Alg.3）：恒收益 15-20%（gf2^16
     1606→1371、gf2^32 6932→6477、dtc_100 5656→4488）
  8. 噪声感知路由（noise_w，默认关）：距离×边误差会破坏跳数优化
     （swap +20-60%），保留为实验参数
诊断数据：gf2^7_mult 1425→271 swap；gf2^16 死循环消除；clifford_100
71388→~12k。

性能：候选评分向量化（numpy 广播 [候选×前沿] 距离矩阵），50 比特级
线路秒级（2026-08-21 优化：原纯 Python 循环 qaoa_50 994s → 60s，见
技术报告 §7.2）。
2026-08-22 增量化（纯工程加速，评分函数与选择逻辑逐位等价）：
  1. 前沿/扩展集增量维护——allowed=frontier∪executed 单调增长，门执行
     时刻即 deps[s]⊆allowed 翻转时刻，exec 时更新、交换迭代零重建；
  2. 候选边由邻接表构建（排序遍历 → 与原全量扫描同序 → argmin 决胜一致）；
  3. inv 映射 O(1) 位置更新；
  4. 候选×前沿距离矩阵 Numba 融合内核（swap_kernel，无中间 [C,F]
     分配；dist==1 ⟺ 相邻替代 amat 命中）。
另有 2026-08-22 deps 防御性拷贝修复：路由消耗的依赖集不再污染调用方
线路对象（mapper 候选评分对同一 circ 多次调用，此前第 2 次起 DAG
残缺、评分失真）。
正确性由态矢量等价测试 + 硬件合法性测试保证。
"""
from __future__ import annotations

import numpy as np

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.swap_kernel import after_dist_matrix


def sabre_route(circ: Circuit, spec, layout: dict, seed: int = 0,
                lam_ms: float = 0.0, w_ext: float | None = None,
                decay_pen: float = 0.0, tabu_window: int = 0,
                tabu_mult: float = 1000.0, tabu_decay: float = 0.0,
                tabu_scope: str = "edge", tabu_count: bool = False,
                lookahead_w: float = 0.5, noise_w: float = 0.0,
                noise_tie: float = 0.0,
                max_swaps: int = 100000, trials: int = 1):
    """返回 (routed: Circuit[物理空间], swap_count, final_layout)。

    w_ext=None → 0.5/spec.n（qiskit decay 口径）；decay_pen 为历史遗留
    参数，已由有界禁忌替代（传入即忽略）。默认 tabu_window=0（无禁忌）：
    2026-11-10 诊断 9 线路 ×4 配置验证无禁忌总量最优且无单条劣化，
    恒权评分 + 种子随机平局 + 10n 逃逸已保证无换位死循环。禁忌模式
    （可选开启，qiskit decay(0.001,5) 的近似，按需启用）：
      tabu_count=False + tabu_decay=0：窗口内使用过的边/比特 ×tabu_mult
      tabu_decay>0：惩罚随步龄指数衰减（软禁忌）
      tabu_count=True：窗口内使用次数 k → ×tabu_mult^k
    lookahead_w>0：前瞻启发式（SABRE 论文 Algorithm 3）：score =
      H(s) + lookahead_w·min_{s2≠s} H(s∘s2)（qiskit 2.5 已无前向模拟
      lookahead——本实现差异化）。
    noise_w>0：噪声感知距离缩放（实验参数，默认关：破坏跳数优化）。
    noise_tie>0：噪声感知平局裁决（实验参数，默认关——实测无论容差
      1.0 还是精确同分均劣化换数：dtc_100 4488→7838，低误差边形成
      路由陷阱；保留供后续研究，勿作默认）。
    trials>1 时多种子路由取 swap 最少者（种子 = seed..seed+trials-1）。
    """
    if trials > 1:
        best = None
        for k in range(trials):
            routed, swaps, fin = sabre_route(
                circ, spec, layout, seed=seed + k, lam_ms=lam_ms,
                w_ext=w_ext, decay_pen=decay_pen, tabu_window=tabu_window,
                tabu_mult=tabu_mult, tabu_decay=tabu_decay,
                tabu_scope=tabu_scope, tabu_count=tabu_count,
                lookahead_w=lookahead_w,
                max_swaps=max_swaps, trials=1)
            if best is None or swaps < best[1]:
                best = (routed, swaps, fin)
        return best

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
    if noise_w > 0.0:
        # 噪声感知路由：距离按边校准误差缩放 dist·(1+noise_w·ε)——
        # 同跳数下优先低误差边（qiskit SabreSwap 噪声盲，差异化机制）
        err2q = spec.calib.err_2q
        med = float(np.median(list(err2q.values()))) if err2q else 1e-3
        scale = np.ones((spec.n, spec.n), dtype=np.float64)
        for (i, j), e in err2q.items():
            scale[i, j] = scale[j, i] = 1.0 + noise_w * float(e)
        dist = dist.astype(np.float64) * scale
    dist_flat = dist.ravel()
    edge_err = None
    if noise_tie > 0.0:
        # 噪声感知平局裁决用：逐边校准误差（缺失边取中位）
        err2q = spec.calib.err_2q
        med = float(np.median(list(err2q.values()))) if err2q else 1e-3
        edge_err = np.full(spec.n * spec.n, med, dtype=np.float64)
        for (i, j), e in err2q.items():
            edge_err[i * spec.n + j] = edge_err[j * spec.n + i] = float(e)

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
    out = Circuit(spec.n, name=circ.name + "_routed")
    swap_count = 0
    wext = w_ext if w_ext is not None else 0.5 / spec.n
    # 禁忌表：物理边/比特上次被选为交换的步号（-1 = 从未）
    last_edge = np.full(spec.n * spec.n, -1, dtype=np.int64)
    last_qubit = np.full(spec.n, -1, dtype=np.int64)
    recent_edges = []
    since_exec = 0
    attempt_limit = 10 * circ.n

    def front_arrays(fset):
        """集合内 2Q 门的 (q1pos, q2pos) numpy 数组。"""
        idx = [i for i in fset if len(ops[i].qubits) == 2]
        if not idx:
            return None
        q1 = np.array([pos[ops[i].qubits[0]] for i in idx], dtype=np.int64)
        q2 = np.array([pos[ops[i].qubits[1]] for i in idx], dtype=np.int64)
        return q1, q2

    while frontier:
        exec_now = [i for i in frontier
                    if len(ops[i].qubits) == 1
                    or amat[pos[ops[i].qubits[0]], pos[ops[i].qubits[1]]]]
        if exec_now:
            since_exec = 0
            # 按线路原始序执行（= DAG 拓扑序；qiskit 语义）——set 任意序
            # 会打散前沿形状、劣化后续交换决策（2026-11-10 诊断）
            for i in sorted(exec_now):
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
        fq1, fq2 = f_arr
        e_arr = front_arrays(ext)
        eq1, eq2 = (e_arr if e_arr is not None else
                    (np.zeros(0, dtype=np.int64),
                     np.zeros(0, dtype=np.int64)))

        eps = set(fq1.tolist()) | set(fq2.tolist())
        if eq1.size:  # 扩展集位置同样参与候选（qiskit：候选边取 F∪E 邻域）
            eps |= set(eq1.tolist()) | set(eq2.tolist())
        # 候选边：前沿门触碰位置的邻接边（排序遍历 → 全量扫描同序）
        pairs = set()
        for a in eps:
            for b in nbrs[a]:
                pairs.add((a, b) if b > a else (b, a))
        cand = sorted(pairs)
        if not cand:
            break
        ea_ = np.array([a for a, _ in cand], dtype=np.int64)
        eb_ = np.array([b for _, b in cand], dtype=np.int64)

        # 融合内核：直接产出 [候选×前沿] 距离矩阵（dist==1 ⟺ 相邻）
        Df = after_dist_matrix(dist_flat, spec.n, fq1, fq2, ea_, eb_)
        basic = Df.sum(axis=1)
        unlocked = (Df == 1).sum(axis=1)
        De = after_dist_matrix(dist_flat, spec.n, eq1, eq2, ea_, eb_)
        ext_cost = De.sum(axis=1)
        ms_delta = 1.0 - unlocked / max(len(fq1), 1)
        if lookahead_w > 0.0:
            # 前瞻（SABRE Algorithm 3 / qiskit lookahead）：每候选 s 先
            # 换位得到前沿新位置，再枚举后续交换 s2（排除自身）取最优
            # 代价，加权加入评分——惩罚"第一步就把路走死"的候选
            P1 = np.where(fq1[None, :] == ea_[:, None], eb_[:, None],
                          np.where(fq1[None, :] == eb_[:, None],
                                   ea_[:, None], fq1[None, :]))
            P2 = np.where(fq2[None, :] == ea_[:, None], eb_[:, None],
                          np.where(fq2[None, :] == eb_[:, None],
                                   ea_[:, None], fq2[None, :]))
            h2 = np.full(len(cand), np.inf)
            for c in range(len(cand)):
                D2 = after_dist_matrix(dist_flat, spec.n, P1[c], P2[c],
                                       ea_, eb_)
                v = D2.sum(axis=1).astype(np.float64)
                v[c] = np.inf  # 排除自身（两次同边换位=恒等，无信息）
                h2[c] = v.min()
            scores = basic + wext * ext_cost + lookahead_w * h2
        else:
            scores = basic + wext * ext_cost
        # 有界禁忌（对标 qiskit decay(0.001, 5)）：重复使用刚交换过的
        # 边/比特 ×tabu_mult；count 模式按窗口内使用次数 k 取 tabu_mult^k
        if tabu_window > 0 and swap_count > 0:
            if tabu_count:
                mult = np.ones(len(cand))
                for (ra, rb) in recent_edges:
                    hit = (ea_ == ra) & (eb_ == rb)
                    if hit.any():
                        mult[hit] *= tabu_mult
                scores = scores * mult
            elif tabu_scope == "both":
                # 组合：按比特温和平滑（×3 半衰）+ 按边硬窗口（×tabu_mult）
                aq = swap_count - np.maximum(last_qubit[ea_], last_qubit[eb_])
                scores = scores * (1.0 + 2.0 * (0.5 ** aq.astype(np.float64)))
                ae = swap_count - last_edge[ea_ * spec.n + eb_]
                touch = ae <= tabu_window
                if touch.any():
                    scores = scores * np.where(touch, tabu_mult, 1.0)
            else:
                if tabu_scope == "edge":
                    age = swap_count - last_edge[ea_ * spec.n + eb_]
                else:
                    age = swap_count - np.maximum(last_qubit[ea_],
                                                  last_qubit[eb_])
                if tabu_decay > 0.0:
                    mult = 1.0 + (tabu_mult - 1.0) * (
                        tabu_decay ** age.astype(np.float64))
                    scores = scores * mult
                else:
                    touch = age <= tabu_window
                    if touch.any():
                        scores = scores * np.where(touch, tabu_mult, 1.0)
        scores = scores + lam_ms * ms_delta

        if since_exec >= attempt_limit:
            # 防卡死逃逸：连续 10n 次无门执行的交换后随机换边
            pick = int(rng.integers(0, len(cand)))
        else:
            # 平局裁决：评分在最优+noise_tie 容差内的候选按边误差升序
            # （噪声感知，qiskit 无此机制）；误差同分时种子随机
            sbest = scores.min()
            tied = np.where(scores <= sbest + noise_tie)[0]
            if noise_tie > 0.0 and edge_err is not None and len(tied) > 1:
                errs = edge_err[ea_[tied] * spec.n + eb_[tied]]
                ebest = errs.min()
                tied = tied[errs <= ebest + 1e-12]
            pick = int(tied[rng.integers(0, len(tied))])
        a, b = int(ea_[pick]), int(eb_[pick])
        out.append(Inst("swap", (a, b)))
        swap_count += 1
        since_exec += 1
        last_edge[a * spec.n + b] = swap_count
        last_qubit[a] = swap_count
        last_qubit[b] = swap_count
        recent_edges.append((a, b))
        if len(recent_edges) > tabu_window:
            recent_edges.pop(0)
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
