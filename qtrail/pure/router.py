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

_DEBUG_LAYER = False


def sabre_route(circ: Circuit, spec, layout: dict, seed: int = 0,
                lam_ms: float = 0.0, w_ext: float | None = 0.0,
                decay_pen: float = 0.0, tabu_window: int = 0,
                tabu_mult: float = 1000.0, tabu_decay: float = 0.0,
                tabu_scope: str = "edge", tabu_count: bool = False,
                lookahead_w: float = 0.0, w_layer: float | None = None,
                cand_scope: str = "front",
                noise_w: float = 0.0,
                noise_tie: float = 0.0,
                decay_inc: float = 0.0, decay_reset: int = 5,
                scorer=None, scorer_sample: bool = False,
                scorer_temp: float = 1.0, scorer_stats: dict | None = None,
                max_swaps: int = 100000, trials: int = 1,
                mix_rollout: bool = False, pcg_tie: bool = True):
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
      lookahead——本实现差异化，rollout 前瞻）。
    w_layer：qiskit lookahead 层权重（2026-11-13 对齐 rust 源码语义）：
      第 1 层 = 把 front 层所有 2Q 门视为已满足后拓扑暴露的下一批 2Q
      门集（deps ⊆ allowed∪frontier），score += w_layer·Σ_{门∈层1}
      交换后距离和。None → 0.5/spec.n（qiskit O3 默认；实测该层权重
      ~0.005 单独贡献 38% swap 减少）；0.0 显式关闭。与 rollout 前瞻
      （lookahead_w）独立可叠加。
    cand_scope：候选交换来源。"front" = 仅 front 层活跃比特的邻边
      （qiskit 2.5 rust 语义）；"fext" = front∪ext 活跃比特（旧语义）。
    noise_w>0：噪声感知距离缩放（实验参数，默认关：破坏跳数优化）。
    noise_tie>0：噪声感知平局裁决（实验参数，默认关——实测无论容差
      1.0 还是精确同分均劣化换数：dtc_100 4488→7838，低误差边形成
      路由陷阱；保留供后续研究，勿作默认）。
    trials>1 时多种子路由取 swap 最少者（种子 = seed..seed+trials-1）。

    scorer（学习式评分器，默认 None）：传入 RouteScorer 时评分改由
    MLP 给出（特征 = 距离分量/步龄/边误差/度/窗口使用次数，见
    rl_score.py）；scorer_sample=True 时 softmax 采样并把 logp/熵累
    计进 scorer_stats（REINFORCE 训练用），否则贪心 argmax。
    """
    if trials > 1:
        # 多种子并行（2026-11-12：numba/numpy 释放 GIL，线程池即可
        # 近似并行；对标 qiskit rust 的并行 trials）
        from concurrent.futures import ThreadPoolExecutor
        kwargs = dict(lam_ms=lam_ms, w_ext=w_ext, decay_pen=decay_pen,
                      tabu_window=tabu_window, tabu_mult=tabu_mult,
                      tabu_decay=tabu_decay, tabu_scope=tabu_scope,
                      tabu_count=tabu_count, lookahead_w=lookahead_w,
                      w_layer=w_layer, cand_scope=cand_scope,
                      noise_w=noise_w, noise_tie=noise_tie, scorer=scorer,
                      scorer_sample=scorer_sample, scorer_temp=scorer_temp,
                      scorer_stats=scorer_stats, max_swaps=max_swaps,
                      trials=1, pcg_tie=pcg_tie)

        def _trial(k):
            kw = kwargs
            if mix_rollout and k % 2 == 1:
                # 混合机制（2026-11-13）：奇号种子用 rollout 前瞻
                # （dtc 类线路有效：前向模拟预见 basic 看不到的信息，
                # 实测 5656→4488）；偶号用 lookahead 层（hwb/gf2^7 类
                # 有效：知情平局裁决）。取 swap 最少者——两机制互补。
                kw = dict(kwargs, lookahead_w=0.5, w_layer=0.0)
            return sabre_route(circ, spec, dict(layout), seed=seed + k,
                               **kw)

        with ThreadPoolExecutor(max_workers=min(trials, 8)) as ex:
            results = list(ex.map(_trial, range(trials)))
        best = min(results, key=lambda t: t[1])
        return best

    rng = np.random.default_rng(seed)
    rng_pcg = None
    if pcg_tie:
        from qtrail.pure.pcg import Pcg64Mcg
        rng_pcg = Pcg64Mcg(seed)
    ops = circ.ops
    m = len(ops)
    # ops 数组化（2026-11-12 机制加速）：qubits/门类型转 numpy——
    # 主循环与依赖构建的 Python 属性访问 → 数组索引
    _q0 = np.empty(m, dtype=np.int64)
    _q1 = np.empty(m, dtype=np.int64)
    _nq = np.empty(m, dtype=np.int64)
    for i, inst in enumerate(ops):
        _nq[i] = len(inst.qubits)
        _q0[i] = int(inst.qubits[0])
        _q1[i] = int(inst.qubits[1]) if _nq[i] == 2 else -1

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
    if noise_tie > 0.0 or scorer is not None:
        # 逐边校准误差（缺失边取中位）：噪声平局裁决 / 评分器特征用
        err2q = spec.calib.err_2q
        med = float(np.median(list(err2q.values()))) if err2q else 1e-3
        edge_err = np.full(spec.n * spec.n, med, dtype=np.float64)
        for (i, j), e in err2q.items():
            edge_err[i * spec.n + j] = edge_err[j * spec.n + i] = float(e)

    # 防御性拷贝：路由过程就地消耗依赖集；mapper 候选评分会对同一 circ
    # 对象反复调用本函数，拷贝保证每次调用看到完整 DAG（2026-08-22 修复）。
    # 依赖重建（2026-11-13，qiskit SabreDAG 精确语义）：每门的依赖 =
    # 同比特的【前一个 2Q 门】（1Q 门附着在 2Q 门上，不构成依赖、不被
    # 依赖——qiskit 的 try_route 对 1Q 门总立即成功）。原始
    # Circuit.deps() 是"同比特最后触碰"链（1Q 门参与），导致 lookahead
    # 层 matching 性被破坏——hwb10 上 L1 层 6429 门（16 比特）、层分量
    # ~76 压过 basic ~3-7、层失效（平凡布局 swap 28828 vs qiskit 20004）。
    # 链合并（2026-11-13，qiskit SabreDAG 链节点语义）：qasm 紧邻且
    # 同比特对的连续 2Q 门合并为链——链头执行即整链执行（同比特对 ⇒
    # 同时可路由），链成员不进入依赖/层结构。逐门处理会让 lookahead
    # 第 1 层沦为 front 的同比特复制（层分量对平局候选常数、wl=50
    # 零效果），qiskit 的层 = 下一批链（新比特对位置）——dtc 类 2x
    # 差距的机制（dag.rs prev.indices.push 定位）。
    chain_next = [-1] * m
    chain_prev = [-1] * m
    # 链判定（2026-11-13 二修）：qiskit Single 前驱语义——2Q 门 g2 的
    # 两个比特的"前一个 2Q"相同且同对 ⇒ 并入链（**穿过中间 1Q 门**，
    # 此前仅 qasm 紧邻——dtc 类 1Q 门密集、链全空，第 0 步 front 与
    # qiskit 分歧的对拍实锤）。链合并后链头执行即整链执行（同对 ⇒
    # 同时可路由）。
    # 链判定 O(m)（2026-11-12 修复）：原实现对每门回溯
    # range(i-1,-1,-1) 找前驱——O(m²)，hwb12 51 万门的依赖构建
    # 隐藏成本。改用 last_2q 前向跟踪（与 deps 段同款）。
    last2q_by_q = {}
    for i in range(m):
        if _nq[i] != 2 or ops[i].name == "swap":
            continue
        a, b = _q0[i], _q1[i]
        pa = last2q_by_q.get(a)
        pb = last2q_by_q.get(b)
        if (pa is not None and pa == pb
                and {_q0[pa], _q1[pa]} == {a, b}):
            # 沿链尾延伸（pa 可能是链成员——回溯版语义：挂在前驱
            # 之后；直接挂链头会覆盖已有链成员、破坏链）
            tail = pa
            while chain_next[tail] != -1:
                tail = chain_next[tail]
            chain_next[tail] = i
            chain_prev[i] = tail
            continue
        last2q_by_q[a] = i
        last2q_by_q[b] = i

    deps = [set() for _ in range(m)]
    last_2q = {}
    for i in range(m):
        if chain_prev[i] != -1:
            continue          # 链成员：并入链头，不参与依赖结构
                              # （last_2q 保持链头——链外后继 deps
                              # =链头，链头执行后满足；1Q 门的
                              # owner 判定靠区间位置限制）
        for q in ops[i].qubits:
            j = last_2q.get(int(q))
            if j is not None:
                deps[i].add(j)
        if len(ops[i].qubits) == 2:
            for q in ops[i].qubits:
                last_2q[int(q)] = i
    # 链区间内的 1Q 门并入链（qiskit 附着语义）：deps ⊆ 本链 2Q 集
    chain_1q_owner = [-1] * m
    for head in range(m):
        if chain_prev[head] != -1 or chain_next[head] == -1:
            continue
        chain_set = set()
        j = head
        while j != -1:
            chain_set.add(j)
            j = chain_next[j]
        tail = head
        while chain_next[tail] != -1:
            tail = chain_next[tail]
        for k in range(head + 1, tail + 1):
            if len(ops[k].qubits) == 1 and deps[k] <= chain_set:
                chain_1q_owner[k] = head

    succ = [set() for _ in range(m)]
    for i in range(m):
        for j in deps[i]:
            succ[j].add(i)

    pos = {k: int(v) for k, v in layout.items()}
    inv = {v: k for k, v in pos.items()}
    frontier = {i for i in range(m)
                if not deps[i] and chain_prev[i] == -1
                and chain_1q_owner[i] == -1}
    executed = set()
    allowed = set(frontier)
    # 扩展集增量维护：仅当 deps[s]⊆allowed 由假变真（某依赖执行）或
    # s 进入 frontier 时更新，与逐次全量重算成员逐位一致。
    ext = set()
    # lookahead 层（qiskit rust 语义）：SabreDAG 的依赖是【2Q 门之间】
    # （1Q 门被穿过，不构成节点），故第 1 层 = {2Q 门 s : 所有 2Q 前驱
    # ∈ allowed∪frontier, s∉frontier}。增量维护 cnt_af[s] =
    # |2Q 前驱∩(allowed∪frontier)|——x 进入 allowed∪frontier 的时刻
    # 即 x 进入 frontier（执行不改该并集成员），故 cnt 只在【2Q 门】
    # 进入 frontier 时 +1。swap 不改变层结构（qiskit apply_swap 语义）。
    l1 = set()
    cnt_af = [0] * m
    n2q_deps = [sum(1 for j in deps[i] if len(ops[j].qubits) == 2)
                for i in range(m)]
    for i in frontier:
        for s in succ[i]:
            if s not in frontier and deps[s] <= allowed \
                    and chain_prev[s] == -1 \
                    and len(ops[s].qubits) == 2:
                ext.add(s)
            if len(ops[i].qubits) == 2:
                cnt_af[s] += 1
    if w_layer != 0.0:
        for s in range(m):
            if (s not in frontier and s not in executed
                    and chain_prev[s] == -1
                    and len(ops[s].qubits) == 2
                    and cnt_af[s] == n2q_deps[s]):
                l1.add(s)
    out = Circuit(spec.n, name=circ.name + "_routed")
    swap_count = 0
    batch = []  # 自上次进展以来的交换批（逃逸回卷用）
    wext = w_ext if w_ext is not None else 0.5 / spec.n
    wl = w_layer if w_layer is not None else 0.5 / spec.n
    # 禁忌表：物理边/比特上次被选为交换的步号（-1 = 从未）
    last_edge = np.full(spec.n * spec.n, -1, dtype=np.int64)
    last_qubit = np.full(spec.n, -1, dtype=np.int64)
    recent_edges = []
    edge_hist = np.zeros(spec.n * spec.n, dtype=np.float32)
    deg = np.array([len(nb) for nb in nbrs], dtype=np.float32)
    # decay 启发式（2026-11-12，对齐 qiskit heuristic.rs 的 DecayHeuristic：
    # 每比特乘数 1.0 起、每次交换 +increment、每 decay_reset 次交换或每次
    # 路由进展时全体重置；评分整体 × max(乘数)。极轻微扰动——打破对称
    # 平局而非惩罚。此前 ×1000 禁忌实验失败即因尺度错了 6 个数量级）
    decay = np.ones(spec.n, dtype=np.float64)
    decay_steps = 0
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
        if _DEBUG_LAYER:
            print(f"[dbg-loop] |frontier|={len(frontier)} |ext|={len(ext)} "
                  f"|l1|={len(l1)} executed={len(executed)}", flush=True)
            for fi in list(frontier):
                if len(ops[fi].qubits) == 2:
                    for s in succ[fi]:
                        if len(ops[s].qubits) == 2 and s not in frontier \
                                and s not in executed:
                            print(f"  [dbg-succ] 门{s} deps={len(deps[s])} "
                                  f"cnt={cnt_af[s]} 在L1?={s in l1}",
                                  flush=True)
        exec_now = [i for i in frontier
                    if len(ops[i].qubits) == 1
                    or amat[pos[ops[i].qubits[0]], pos[ops[i].qubits[1]]]]
        if exec_now:
            since_exec = 0
            batch.clear()
            decay.fill(1.0)      # 进展点：decay 窗口重置（qiskit 语义）
            decay_steps = 0
            # 按线路原始序执行（= DAG 拓扑序；qiskit 语义）——set 任意序
            # 会打散前沿形状、劣化后续交换决策（2026-11-10 诊断）
            for i in sorted(exec_now):
                frontier.discard(i)
                executed.add(i)
                # 链头执行 = 整链按序输出（含链区间内并入的
                # 1Q 门——qiskit indices 附着语义）
                chain_set = set()
                j = i
                while j != -1:
                    chain_set.add(j)
                    j = chain_next[j]
                tail = i
                while chain_next[tail] != -1:
                    tail = chain_next[tail]
                for k in range(i, tail + 1):
                    inst = ops[k]
                    if k == i or k in chain_set or chain_1q_owner[k] == i:
                        if k != i:
                            executed.add(k)
                        if inst.nq == 1:
                            out.append(Inst(inst.name,
                                            (pos[inst.qubits[0]],),
                                            inst.params))
                        else:
                            p1 = pos[inst.qubits[0]]
                            p2 = pos[inst.qubits[1]]
                            out.append(Inst(inst.name, (p1, p2),
                                            inst.params))

                for s in succ[i]:
                    deps[s].discard(i)
                    if not deps[s] and chain_1q_owner[s] == -1:
                        frontier.add(s)
                        allowed.add(s)
                        ext.discard(s)
                        l1.discard(s)
                        # s 进入前沿 → 其后继的 deps⊆allowed 可能此刻才成立
                        for t in succ[s]:
                            if t not in frontier and deps[t] <= allowed \
                                    and chain_prev[t] == -1 \
                                    and len(ops[t].qubits) == 2:
                                ext.add(t)
                            # s 进入 allowed∪frontier → lookahead 层计数
                            # （qiskit 语义：仅 2Q 门构成依赖链）
                            if w_layer != 0.0 and len(ops[s].qubits) == 2:
                                cnt_af[t] += 1
                                if (t not in frontier and t not in executed
                                        and chain_prev[t] == -1
                                        and len(ops[t].qubits) == 2
                                        and cnt_af[t] == n2q_deps[t]):
                                    l1.add(t)
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
        if cand_scope == "fext" and eq1.size:
            # 扩展集位置同样参与候选（旧语义）
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

        # 绝对距离和评分（经典 SABRE；qiskit LightSABRE 的边际评分
        # 复现实验 2026-11-12 未获益——其 lookahead 层的额外机制
        # 未能从源码反推，保留我们的双步前瞻差异化）
        Df = after_dist_matrix(dist_flat, spec.n, fq1, fq2, ea_, eb_)
        basic = Df.sum(axis=1)
        unlocked = (Df == 1).sum(axis=1)
        De = after_dist_matrix(dist_flat, spec.n, eq1, eq2, ea_, eb_)
        ext_cost = De.sum(axis=1)
        ms_delta = 1.0 - unlocked / max(len(fq1), 1)
        # lookahead 层评分（qiskit rust 语义）：第 1 层门集的交换后距离和
        l1_cost = None
        if l1:
            lq1 = np.array([pos[ops[i].qubits[0]] for i in l1],
                           dtype=np.int64)
            lq2 = np.array([pos[ops[i].qubits[1]] for i in l1],
                           dtype=np.int64)
            l1_cost = after_dist_matrix(dist_flat, spec.n, lq1, lq2,
                                        ea_, eb_).sum(axis=1)
        if _DEBUG_LAYER and len(cand) > 0:
            print(f"[dbg] |front|={len(fq1)} |ext|={len(eq1)} |l1|={len(l1)} "
                  f"basic=({basic.min():.1f},{basic.max():.1f}) "
                  + (f"l1_raw=({l1_cost.min():.1f},{l1_cost.max():.1f}) "
                     if l1_cost is not None else "l1_EMPTY"),
                  flush=True)
        h2 = None
        if lookahead_w > 0.0 or scorer is not None:
            # 前瞻（SABRE Algorithm 3）：每候选 s 先换位得到前沿新位置，
            # 再枚举后续交换 s2（排除自身）取最优代价——惩罚"第一步就
            # 把路走死"的候选（scorer 模式作为评分器特征）
            P1 = np.where(fq1[None, :] == ea_[:, None], eb_[:, None],
                          np.where(fq1[None, :] == eb_[:, None],
                                   ea_[:, None], fq1[None, :]))
            P2 = np.where(fq2[None, :] == ea_[:, None], eb_[:, None],
                          np.where(fq2[None, :] == eb_[:, None],
                                   ea_[:, None], fq2[None, :]))
            # 前瞻批量内核（2026-11-12）：替代 Python 循环
            # （20 次 numba 边界调用/门 → 1 次）
            from qtrail.pure.swap_kernel import h2_lookahead_batch
            h2 = h2_lookahead_batch(ea_, eb_, P1, P2, dist_flat,
                                    spec.n)
        scores = basic + wext * ext_cost \
            + (lookahead_w * h2 if h2 is not None else 0.0) \
            + (wl * l1_cost if l1_cost is not None else 0.0)
        if decay_inc > 0.0:
            # decay：整体 × max(乘数)（qiskit DecayHeuristic 语义）
            scores = scores * np.maximum(decay[ea_], decay[eb_])

        if since_exec >= attempt_limit:
            # 防卡死逃逸（qiskit 语义，2026-11-12 对齐）：
            # 回卷当前交换批到上次进展点，贪心把距离最小的前沿门
            # 的比特沿最短路径拉到相邻（force_enable_closest_node）
            for (ra, rb) in reversed(batch):
                out.ops.pop()
                swap_count -= 1
                la, lb = inv.get(ra), inv.get(rb)
                if la is not None:
                    inv.pop(ra, None)
                    pos[la] = rb
                if lb is not None:
                    inv.pop(rb, None)
                    pos[lb] = ra
                if la is not None:
                    inv[rb] = la
                if lb is not None:
                    inv[ra] = lb
            batch.clear()
            since_exec = 0
            # 最近门贪心拉近：沿 BFS 最短路径逐边移动
            best_gate = None
            best_d = 10 ** 9
            for i in frontier:
                if len(ops[i].qubits) == 2:
                    d = int(dist[pos[ops[i].qubits[0]], pos[ops[i].qubits[1]]])
                    if d < best_d:
                        best_d = d
                        best_gate = i
            if best_gate is not None and best_d > 1:
                q0 = pos[ops[best_gate].qubits[0]]
                q1 = pos[ops[best_gate].qubits[1]]
                # BFS 最短路径（q0 → q1）
                prev = {q0: None}
                from collections import deque
                dq = deque([q0])
                while dq:
                    u = dq.popleft()
                    if u == q1:
                        break
                    for v in nbrs[u]:
                        if v not in prev:
                            prev[v] = u
                            dq.append(v)
                if q1 in prev:
                    path = []
                    u = q1
                    while u is not None:
                        path.append(u)
                        u = prev[u]
                    path.reverse()
                    # 两端向中间 split 推进（qiskit force_enable_closest_
                    # node 语义）：左右两侧 swap 可并行，深度减半
                    split = len(path) // 2
                    moves = [(path[i], path[i + 1]) for i in range(split)]
                    for i in range(split - 1):
                        e = len(path) - 1 - i
                        moves.append((path[e], path[e - 1]))
                    for a, b in moves:
                        out.append(Inst("swap", (a, b)))
                        swap_count += 1
                        batch.append((a, b))
                        la, lb = inv.get(a), inv.get(b)
                        if la is not None:
                            inv.pop(a, None)
                            pos[la] = b
                        if lb is not None:
                            inv.pop(b, None)
                            pos[lb] = a
                        if la is not None:
                            inv[b] = la
                        if lb is not None:
                            inv[a] = lb
            continue
        elif scorer is not None:
            # 学习式评分（rl_score.RouteScorer）：MLP 特征打分
            # （距离分量/步龄/边误差/度/窗口使用次数），采样或贪心
            import torch
            nF = max(len(fq1), 1)
            nE = max(len(eq1), 1)
            F = np.empty((len(cand), 9), dtype=np.float32)
            F[:, 0] = basic / nF
            F[:, 1] = ext_cost / nE
            F[:, 2] = h2 / nF
            F[:, 3] = unlocked / nF
            F[:, 4] = np.minimum(
                (swap_count - last_edge[ea_ * spec.n + eb_]) / 20.0, 1.0)
            F[:, 5] = np.minimum((swap_count - np.maximum(
                last_qubit[ea_], last_qubit[eb_])) / 20.0, 1.0)
            F[:, 6] = edge_err[ea_ * spec.n + eb_]
            F[:, 7] = (deg[ea_] + deg[eb_]) / 10.0
            F[:, 8] = np.minimum(edge_hist[ea_ * spec.n + eb_] / 5.0, 1.0)
            st = scorer(torch.from_numpy(F),
                        torch.from_numpy(scores).float())
            if scorer_sample:
                pick, logp, ent = scorer.sample_pick(st, scorer_temp)
                if scorer_stats is not None:
                    scorer_stats["logp"] += float(logp.detach())
                    scorer_stats["entropy"] += float(ent.detach())
                    # 保留计算图的逐步累加（REINFORCE 反向传播）
                    scorer_stats["logp_t"] = \
                        scorer_stats.get("logp_t", 0.0) + logp
                    scorer_stats["ent_t"] = \
                        scorer_stats.get("ent_t", 0.0) + ent
            else:
                pick = scorer.greedy_pick(st)
        else:
            # 有界禁忌（对标 qiskit decay(0.001, 5)）：重复使用刚交换
            # 过的边/比特 ×tabu_mult；count 模式按窗口使用次数取幂
            if tabu_window > 0 and swap_count > 0:
                if tabu_count:
                    mult = np.ones(len(cand))
                    for (ra, rb) in recent_edges:
                        hit = (ea_ == ra) & (eb_ == rb)
                        if hit.any():
                            mult[hit] *= tabu_mult
                    scores = scores * mult
                elif tabu_scope == "both":
                    # 组合：按比特温和平滑（×3 半衰）+ 按边硬窗口
                    aq = swap_count - np.maximum(last_qubit[ea_],
                                                 last_qubit[eb_])
                    scores = scores * (1.0 + 2.0 * (
                        0.5 ** aq.astype(np.float64)))
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
            # 平局裁决：qiskit best_epsilon=1e-10 容差（rust 语义）——
            # 距离和是整数，存在大量精确平局；lookahead 层分量（~1e-8
            # 量级）在容差内打破平局定向选择（知情平局裁决，qiskit
            # lookahead 层实测 38% 收益的来源）。noise_tie 保留为噪声
            # 感知扩展（默认 0 关闭）。
            sbest = scores.min()
            tied = np.where(scores <= sbest + max(1e-10, noise_tie))[0]
            if _DEBUG_LAYER and len(tied) > 1:
                if l1_cost is not None:
                    lr = (l1_cost[tied].min(), l1_cost[tied].max())
                else:
                    lr = None
                print(f"[dbg-tie] n_tied={len(tied)} n_cand={len(cand)} "
                      f"basic_tied=({basic[tied].min()},{basic[tied].max()}) "
                      + (f"l1_tied={lr}" if lr else "l1_EMPTY"),
                      flush=True)
            if noise_tie > 0.0 and edge_err is not None and len(tied) > 1:
                errs = edge_err[ea_[tied] * spec.n + eb_[tied]]
                ebest = errs.min()
                tied = tied[errs <= ebest + 1e-12]
            if pcg_tie and rng_pcg is not None:
                pick = int(tied[rng_pcg.gen_range(len(tied))])
            else:
                pick = int(tied[rng.integers(0, len(tied))])
        a, b = int(ea_[pick]), int(eb_[pick])
        out.append(Inst("swap", (a, b)))
        swap_count += 1
        since_exec += 1
        batch.append((a, b))
        last_edge[a * spec.n + b] = swap_count
        last_qubit[a] = swap_count
        last_qubit[b] = swap_count
        recent_edges.append((a, b))
        edge_hist[a * spec.n + b] += 1.0
        if len(recent_edges) > tabu_window:
            oa, ob = recent_edges.pop(0)
            edge_hist[oa * spec.n + ob] -= 1.0
        if decay_inc > 0.0:
            decay_steps += 1
            if decay_steps >= decay_reset:
                decay.fill(1.0)
                decay_steps = 0
            else:
                decay[a] += decay_inc
                decay[b] += decay_inc
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
            # 2026-11-13 审查代理 P1 修复：触顶不再静默截断（丢弃
            # 剩余门产生语义错误的半线路——hwb12 51 万门只发射
            # 前缀的根因），改为显式拒绝交付。
            raise RuntimeError(
                f"sabre_route swap 触顶 {max_swaps}——线路需要更多"
                f"路由预算，调用方须提高 max_swaps 或降级线路")

    for inst in circ.measures:
        out.append(Inst("measure", (pos[inst.qubits[0]],), cbits=inst.cbits))
    out.strip_ids()
    return out, swap_count, pos
