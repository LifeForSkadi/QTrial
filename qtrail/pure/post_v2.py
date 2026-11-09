# -*- coding: utf-8 -*-
"""v5 后处理栈（新管线，与 post.py 并存；原 v4 行为保持不变）。

在 v4（post.py：连通性感知推挤 + 同向 cx 消解 + 尾部吸收）之上新增：

  B 组件：1Q 游程优化（对标 qiskit Optimize1qGates）——
    rz/sx/x 同比特游程矩阵乘积合并为 u3 再合成（相位剥离 + θ=π 分支，
    数值坑已在 wip/improvements-v6 踩过并修复）；
    H 块奇偶消解（cx→H-CZ-H 分解产生的相邻 H 块，H²=I 精确成立，
    偶数个相邻 H 块整体删除、奇数保留一个）；
    分解后 1Q 优化不动点循环（合并→再合成→rz 合并→H 消解，3 轮上限）。

  A2 组件：2Q 块重合成（下一阶段实现，见 collect_resynthesize_2q_blocks
    占位说明）——SWAP 消化的核心机制。

正确性：全部变换保语义（1Q 游程合并是矩阵精确运算；H 消解是精确恒等式），
由态矢量等价测试 + A/B 门禁（v4 vs v5 逐位可解释差异）+ 硬件合法性测试保证。
"""
from __future__ import annotations

import math

import numpy as np

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.post import (H_GATE, _u3_to_rzsx, absorb_trailing,
                              cancel_cx, merge_adjacent_rz, post_route,
                              push_swaps)

# v4 行为保持：推挤/消解/吸收与 post.py 完全相同（连通性感知）
post_route_v2 = post_route
push_swaps_v2 = push_swaps
cancel_cx_v2 = cancel_cx
absorb_trailing_v2 = absorb_trailing


# ---------------------------------------------------------------- B：1Q 优化
def _oneq_matrix(name: str, params: tuple):
    """rz/sx/x/u3/u1 → 2×2 矩阵（平台约定：RZ(a)=diag(e^{−ia/2}, e^{ia/2})）。"""
    if name == "rz":
        h = params[0] / 2.0
        return np.diag([np.exp(-1j * h), np.exp(1j * h)])
    if name == "sx":
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
    if name == "x":
        return np.array([[0.0, 1.0], [1.0, 0.0]])
    if name == "u1":
        return np.diag([1.0, np.exp(1j * params[0])])
    if name == "u3":
        th, ph, la = params
        c, s = math.cos(th / 2), math.sin(th / 2)
        return np.array([[c, -np.exp(1j * la) * s],
                         [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]],
                        dtype=complex)
    raise ValueError(name)


def _mat_to_u3_params(U) -> tuple:
    """SU(2) 矩阵 → u3(θ,φ,λ) 参数（含全局相位剥离的精确提取）。

    一般 SU(2)：U = e^{iα}·u3(θ,φ,λ)，α 为 a 的辐角（u3 规范形
    a=cos(θ/2) 为实）；e^{iα} 是 1Q 门的全局相位、物理无关，剥离后
    θ = 2·atan2(|b|,|a|)，φ = arg(c)−α，λ = arg(−b)−α。
    θ=π 时 d=a=0 分支取 α=0；θ>π/2 时经 d 求 α（a 幅角数值不稳）。
    """
    a, b = complex(U[0, 0]), complex(U[0, 1])
    c, d = complex(U[1, 0]), complex(U[1, 1])
    th = 2.0 * math.atan2(abs(b), abs(a))
    if abs(b) < 1e-12:  # 对角：纯 rz
        return (th, 0.0, (math.atan2(d.imag, d.real)
                          - math.atan2(a.imag, a.real)) % (2 * math.pi))
    if abs(a) < 1e-12:  # θ=π：d=a=0，规范取 α=0
        return (th, math.atan2(c.imag, c.real) % (2 * math.pi),
                math.atan2((-b).imag, (-b).real) % (2 * math.pi))
    if abs(a) >= abs(b):
        alpha = math.atan2(a.imag, a.real)
    else:  # θ>π/2：经 d 求 α
        alpha = (math.atan2(c.imag, c.real) + math.atan2((-b).imag, (-b).real)
                 - math.atan2(d.imag, d.real))
    ph = (math.atan2(c.imag, c.real) - alpha) % (2 * math.pi)
    la = (math.atan2((-b).imag, (-b).real) - alpha) % (2 * math.pi)
    return (th, ph, la)


def merge_1q_runs(out_ops: list) -> int:
    """相邻同比特 1Q 门（rz/sx/x）游程合并为单一 u3（矩阵精确）。

    机制对标 qiskit Optimize1qGates：H·u1·H 等游程坍缩为一个 U3，
    ZXZ 再合成后 1Q 门数从 7+ 降到 ≤5。返回合并的游程数。
    """
    merged = 0
    out: list = []
    i = 0
    n = len(out_ops)
    while i < n:
        inst = out_ops[i]
        if inst.name in ("rz", "sx", "x"):
            q = inst.qubits[0]
            j = i
            U = np.eye(2, dtype=complex)
            while j < n and out_ops[j].name in ("rz", "sx", "x") \
                    and out_ops[j].qubits[0] == q:
                U = _oneq_matrix(out_ops[j].name, out_ops[j].params) @ U
                j += 1
            if j - i > 1:
                # 剥离全局相位（sx 行列式为 i，游程乘积 det≠1；1Q 门的
                # 全局相位即全系统全局相位，物理无关、剥离精确）
                U = U / np.sqrt(np.linalg.det(U))
                params = _mat_to_u3_params(U)
                # 收益检查（2026-11-09 自审计修复，v6 移植缺陷）：只有
                # 合成后的门数少于原游程才合并——短游程（2-4 门）合成
                # ZSXXZ 5 门反而膨胀（qft5 实测 +10 门）；长游程（>5 门）
                # 与 H 形游程的收益保留
                if len(_u3_to_rzsx(*params)) < (j - i):
                    out.append(Inst("u3", (q,), params))
                    merged += 1
                else:
                    out.extend(out_ops[i:j])
            else:
                out.append(inst)
            i = j
        else:
            out.append(inst)
            i += 1
    if merged:
        out_ops[:] = out
    return merged


def _is_h_block(ops, i):
    return (ops[i].name == "rz" and ops[i + 1].name == "sx"
            and ops[i + 2].name == "rz"
            and abs(ops[i].params[0] - H_GATE[0]) < 1e-9
            and abs(ops[i + 2].params[0] - H_GATE[0]) < 1e-9)


def _cancel_adjacent_h(out_ops: list) -> None:
    """消除相邻 H 块（rz(π/2),sx,rz(π/2) 三元组）的恒等对：H²=I 精确成立。

    cx→H-CZ-H 分解使共享目标比特的相邻 cx 产生相邻 H 块；同比特上
    偶数个相邻 H 块整体为恒等（奇数个保留一个）。这是 qiskit 优化级
    1Q 门数约减的主要机制（实测 3.65 1Q/cz vs 朴素 7+）。
    """
    i = 0
    while i < len(out_ops) - 2:
        if _is_h_block(out_ops, i):
            q = out_ops[i].qubits[0]
            k, j = 0, i
            while j < len(out_ops) - 2 and _is_h_block(out_ops, j) \
                    and out_ops[j].qubits[0] == q:
                k += 1
                j += 3
            if k >= 2:
                del out_ops[i:j]
                if k % 2 == 1:
                    out_ops[i:i] = [Inst("rz", (q,), (H_GATE[0],)),
                                    Inst("sx", (q,)),
                                    Inst("rz", (q,), (H_GATE[0],))]
                i += 3 if k % 2 == 1 else 0
            else:
                i = j
        else:
            i += 1


# ------------------------------------------------- A2：2Q 块重合成
def _block_unitary(ops_slice, pair):
    """块内门（含 1Q）→ 4×4 酉矩阵（按线路顺序乘积，作用于 pair 上）。"""
    U = np.eye(4, dtype=complex)
    a, b = pair
    for inst in ops_slice:
        if inst.nq == 1:
            m1 = _oneq_matrix(inst.name, inst.params)
            q = int(inst.qubits[0])
            U = (np.kron(m1, np.eye(2)) if q == a
                 else np.kron(np.eye(2), m1)) @ U
        else:
            name = inst.name
            if name == "cx":
                ctl = int(inst.qubits[0])
                m = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                              [0, 0, 1, 0]], dtype=complex)
                if ctl != a:  # 控制位是 b：翻转方向
                    m = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
                                  [0, 1, 0, 0]], dtype=complex)
            elif name == "cz":
                m = np.eye(4, dtype=complex)
                m[3, 3] = -1
            else:  # swap
                m = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                              [0, 0, 0, 1]], dtype=complex)
            U = m @ U
    return U


def collect_resynthesize_2q_blocks(circ: Circuit, spec,
                                    max_blocks: int | None = None) -> Circuit:
    """A2 组件：2Q 块收集 + 重合成（SWAP 消化）。

    流程：collect_2q_blocks 收集同比特对的连续 2Q 门块（含中间 1Q）→
    块酉 → KAK 初值 + torch Adam 拟合 21 参数（3 CX + 12 单比特）→
    收敛验证（overlap > 1-1e-9）→ 替换原块（否则保留原门）。

    块内 2Q 门均作用于同一对（a, b），且 (a, b) 在耦合图上（收集时
    已检查）→ 合成产物硬件合法。噪声感知选优（KAK 相位自由度）作为
    后续增强（当前先保证正确性与收益）。
    """
    from qtrail.pure.synthesis import (block_params_to_insts,
                                       collect_2q_blocks, fit_block_torch,
                                       kak_decompose, _init_3cx_params,
                                       _su2_euler, _cached_3cx_params)

    ops = circ.ops
    blocks = collect_2q_blocks(circ, spec)
    if max_blocks is not None:
        blocks = blocks[:max_blocks]
    if not blocks:
        return circ
    # 从后往前替换（保持索引有效）
    for start, end, pair, n_swap, n_2q in reversed(blocks):
        U = _block_unitary(ops[start:end], pair)
        if abs(np.linalg.det(U)) < 1e-10:
            continue
        U = U / np.linalg.det(U) ** 0.25  # SU(4) 规范化
        # 恒等块（2Q 门相互抵消，如 cx·cx、swap·swap）：整体删除
        ov_i = abs(np.trace(U)) / 4.0
        if ov_i > 1 - 1e-9:
            del ops[start:end]
            continue
        try:
            dec = kak_decompose(U, iters=20)  # 初值器只需"接近"（~6ms）
            init = np.zeros(21)
            for idx, K in enumerate((dec["K1"], dec["K2"])):
                th, ph, la = _su2_euler(K)
                init[3 * idx:3 * idx + 3] = [th, ph, la]
            for idx, K in enumerate((dec["K3"], dec["K4"])):
                th, ph, la = _su2_euler(K)
                init[14 + 3 * idx:14 + 3 * idx + 3] = [th, ph, la]
            # 3-CX 中间参数走缓存（相同 Cartan 系数的块复用，耗时优化）
            init[6:14] = _cached_3cx_params(np.asarray(dec["k"]))
            t, ov = fit_block_torch(U, init)
        except Exception:
            continue
        if ov < 1 - 1e-9:
            continue
        new_insts = block_params_to_insts(t, pair)
        ops[start:end] = new_insts
    circ._deps = None
    return circ


# ------------------------------------------------- C1：噪声感知后布局重标记
def noise_aware_relabel(circ: Circuit, spec, final_layout: dict,
                        max_passes: int = 30) -> int:
    """C1 组件：路由后重标记（对标 VF2PostLayout 的自主弱化版）。

    移动 = 全局标签置换的单步：物理位 p → 空位 p'（线路中所有标签为
    p 的门同时改为 p'——全局置换保语义）。合法性：每个涉及 p 的 2Q 门
    （含 swap）改后端点必须相邻；接受条件：p 相关边的校准误差和下降。
    纯标签置换不改门序列/门数/深度，只改"线路落在哪些耦合器上"。

    返回接受的移动次数。线路与 final_layout 原地更新。
    """
    ops = circ.ops
    disabled = set(spec.disabled_qubits)
    err2q = spec.calib.err_2q
    median2q = float(np.median(list(err2q.values()))) if err2q else 1e-3

    def edge_err(a, b):
        if a > b:
            a, b = b, a
        return err2q.get((a, b), err2q.get((b, a), median2q))

    # 预计算：邻居表与代价（一次 O(L) 扫描；移动后增量更新）
    nbrs = {}
    used = set()
    for inst in ops:
        if inst.nq == 2:
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            nbrs.setdefault(a, set()).add(b)
            nbrs.setdefault(b, set()).add(a)
            used.add(a)
            used.add(b)

    def cost_of(p):
        return sum(edge_err(p, x) for x in nbrs.get(p, ()))

    n_moved = 0
    for _pass in range(max_passes):
        improved = False
        order = sorted(used, key=cost_of, reverse=True)
        for p in order:
            p_nbrs = nbrs.get(p, set())
            if not p_nbrs:
                continue  # 无 2Q 门的孤立位移动无收益
            c_old = sum(edge_err(p, x) for x in p_nbrs)
            best = None
            for p_new in range(spec.n):
                if p_new in disabled or p_new in used:
                    continue
                # 合法性：所有 (p, x) → (p_new, x) 必须相邻
                if not all(spec.adj[p_new, x] for x in p_nbrs):
                    continue
                c_new = sum(edge_err(p_new, x) for x in p_nbrs)
                if c_new < c_old - 1e-12 and (best is None
                                              or c_new < best[0]):
                    best = (c_new, p_new)
            if best is not None:
                p_new = best[1]
                for inst in ops:
                    inst.qubits = tuple(p_new if int(x) == p else int(x)
                                        for x in inst.qubits)
                # final_layout 数值与线路标签同步
                for k, v in final_layout.items():
                    if v == p:
                        final_layout[k] = p_new
                # 增量更新邻居表：p 的邻居把 p 换成 p_new
                for x in p_nbrs:
                    nbrs[x].discard(p)
                    nbrs[x].add(p_new)
                nbrs[p_new] = nbrs.pop(p)
                used.discard(p)
                used.add(p_new)
                n_moved += 1
                improved = True
        if not improved:
            break
    circ._deps = None
    return n_moved


# ------------------------------------------------- L3：学习式重标记（复用 QAPolicy）
def learned_relabel(policy, spec, circ: Circuit, final_layout: dict,
                    cfg, dev="cpu", seed=0, noise_lambda=0.5,
                    n_rl_keep=3) -> int:
    """L3 组件：学习式重标记（复用已训练的 QAPolicy，零重训练）。

    机制：把路由后线路视为"新逻辑线路"（物理位标签 = 新逻辑比特），
    用 QAPolicy 多起点解码生成候选重标记（解码学过的噪声感知自动
    注入）；过滤出全部 2Q 门（含 swap）相邻的合法候选；与恒等/C1
    当前产物按 est_fidelity 仲裁（非劣化）。返回接受时应用的移动
    次数（0 = 未改善，线路不变）。

    候选 π 即标签置换（q → π[q]），合法性 = ∀2Q 门 (π[a],π[b]) 相邻。
    """
    from qtrail.problems import build_program_graph
    from qtrail.pure.metrics import estimate_fidelity
    from qtrail.search.decoding import multistart_decode

    def fid_of_pi(pi):
        """候选置换下的保真度（不修改线路：标签视图计算）。"""
        f = 1.0
        err2q = spec.calib.err_2q
        median2q = float(np.median(list(err2q.values()))) if err2q else 1e-3
        for inst in circ.ops:
            qs = [int(pi[int(q)]) for q in inst.qubits]
            if inst.nq == 2:
                a, b = qs[0], qs[1]
                if a > b:
                    a, b = b, a
                e = err2q.get((a, b), err2q.get((b, a), median2q))
                f *= max(1.0 - float(e), 1e-12)
            else:
                f *= max(1.0 - float(spec.calib.err_1q[qs[0]]), 1e-12)
        for inst in circ.measures:
            q = int(pi[int(inst.qubits[0])])
            f *= max(1.0 - float(spec.calib.err_ro[q]), 1e-12)
        return float(f)

    def legal(pi):
        for inst in circ.ops:
            if inst.nq == 2:
                a = int(pi[int(inst.qubits[0])])
                b = int(pi[int(inst.qubits[1])])
                if not spec.adj[a, b]:
                    return False
        return True

    def apply_pi(pi):
        moved = 0
        for inst in circ.ops:
            new_qs = tuple(int(pi[int(q)]) for q in inst.qubits)
            if new_qs != inst.qubits:
                moved += 1
            inst.qubits = new_qs
        for k, v in final_layout.items():
            final_layout[k] = int(pi[int(v)])
        return moved

    ops = [(i.name, tuple(int(q) for q in i.qubits)) for i in circ.ops]
    graph = build_program_graph(spec.n, ops, circuit_id="relabel")
    fid_now = estimate_fidelity(circ, spec.calib)

    candidates = [np.arange(spec.n, dtype=np.int64)]  # 恒等（当前标签）
    try:
        starts = multistart_decode(policy, graph, spec, cfg.decode,
                                   noise_lambda=noise_lambda, dev=dev,
                                   rng=np.random.default_rng(seed))
        kept = 0
        for s in starts:
            if legal(s) and not any(np.array_equal(s, c)
                                    for c in candidates):
                candidates.append(s)
                kept += 1
                if kept >= n_rl_keep:
                    break
    except Exception:
        pass

    best = None
    for pi in candidates:
        fid_pi = fid_of_pi(pi)
        if fid_pi > fid_now + 1e-12 and (best is None or fid_pi > best[0]):
            best = (fid_pi, pi)
    if best is None:
        return 0
    return apply_pi(best[1])


# ------------------------------------------------- A1：置换折叠 + LNN 重链化
def fold_and_relink(circ: Circuit, spec, max_relink: int = 800,
                    d_max: int = 6):
    """A1 组件：swap 置换折叠 + 长程 2Q 门 LNN 链重链化。

    机制（对标 qiskit VF2PostLayout + 线性函数合成的自主实现）：
      1. 正序单遍折叠：swap 并入标签置换 arr（门 q → arr[q]，swap 删除）
      2. 折叠后的 2Q 门若不相邻 → KMS 链分解（4(d-1) 门，噪声感知选路）
      3. 逐 swap 净收益判定（2026-11-10 修复全折叠的长链爆炸）：吸收
         swap 前前瞻其影响集（后续含 a/b 的 2Q 门），链化距离 ≤ d_max
         且总链成本 < 3 才吸收，否则保留 swap（局部保守、语义正确）
    语义等价（置换折叠是代数恒等）+ 硬件合法（链门在路径相邻对上）。

    返回 (new_circuit, arr)：arr = 折叠置换（未采用时为 None）。
    """
    from qtrail.pure.lnn import noise_aware_chain

    ops = circ.ops
    n_swap = sum(1 for i in ops if i.name == "swap")
    if n_swap == 0:
        return circ, None
    n_2q_orig = sum(1 for i in ops if i.nq == 2)
    orig_cost = n_2q_orig + 3 * n_swap  # 原展开口径（swap→3cx）

    # 折叠 + 链化（逐 swap 判定）
    arr = np.arange(spec.n, dtype=np.int64)
    out = []
    n_relink = 0
    i = 0
    n = len(ops)
    while i < n:
        inst = ops[i]
        if inst.name == "swap":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            # 前瞻影响集：后续含 a/b 的 2Q 门（重标记后的链成本）
            cost = 0.0
            ok = True
            for j in range(i + 1, n):
                g = ops[j]
                if g.name == "swap":
                    # 后续 swap 会独立处理（可能被折叠），不计入成本
                    continue
                if g.nq != 2 or not (a in g.qubits or b in g.qubits):
                    continue
                # 重标记后（当前 arr 下）的门对
                q0 = int(arr[int(g.qubits[0])])
                q1 = int(arr[int(g.qubits[1])])
                if a == int(g.qubits[0]):
                    q0 = int(arr[b])
                elif a == int(g.qubits[1]):
                    q1 = int(arr[b])
                if b == int(g.qubits[0]):
                    q0 = int(arr[a])
                elif b == int(g.qubits[1]):
                    q1 = int(arr[a])
                if spec.adj[q0, q1]:
                    continue  # 重标记后仍相邻：免费
                if n_relink >= max_relink:
                    ok = False
                    break
                # 长程：链成本（路径距离需查——用 dist 矩阵快速查）
                d = int(spec.dist[q0, q1])
                if d > d_max:
                    ok = False
                    break
                cost += 4 * (d - 1) - 1  # 链门 - 原 cx
                if cost >= 3:
                    ok = False
                    break
            if not ok:
                # 不吸收：保留 swap（按当前 arr 映射标签）
                out.append(Inst("swap", (int(arr[a]), int(arr[b]))))
                i += 1
                continue
            # 吸收：置换并入 arr
            arr[a], arr[b] = arr[b], arr[a]
            i += 1
            continue
        if inst.nq == 1:
            out.append(Inst(inst.name, (int(arr[int(inst.qubits[0])]),),
                            inst.params))
            i += 1
            continue
        # 2Q 门（cx/cz）：重标记后检查相邻性
        q0, q1 = int(arr[int(inst.qubits[0])]), int(arr[int(inst.qubits[1])])
        if spec.adj[q0, q1]:
            out.append(Inst(inst.name, (q0, q1), inst.params))
            i += 1
            continue
        gates, _ = noise_aware_chain(spec, q0, q1)
        if gates is None:
            return circ, None
        n_relink += 1
        if inst.name == "cx":
            for gc, gt in gates:
                out.append(Inst("cx", (int(gc), int(gt))))
        else:  # cz(a,b) = H(b)·cx(a,b)·H(b)
            for gname, gp in (("rz", (math.pi / 2,)), ("sx", ()),
                              ("rz", (math.pi / 2,))):
                out.append(Inst(gname, (int(q1),), gp))
            for gc, gt in gates:
                out.append(Inst("cx", (int(gc), int(gt))))
            for gname, gp in (("rz", (math.pi / 2,)), ("sx", ()),
                              ("rz", (math.pi / 2,))):
                out.append(Inst(gname, (int(q1),), gp))
        i += 1
    # 测量重标记
    new_circ = Circuit(spec.n, name=circ.name)
    new_circ.ops = out
    for m in circ.measures:
        new_circ.measures.append(
            Inst("measure", (int(arr[int(m.qubits[0])]),), cbits=m.cbits))
    n_2q_new = sum(1 for i in out if i.nq == 2)
    if n_2q_new >= orig_cost:
        return circ, None  # 不劣化：折叠无收益（arr=None 表示未采用）
    return new_circ, arr


# ------------------------------------------------- A2：可交换消解（rz 穿 cz）
def commute_rz_through_cz(out_ops: list, max_rounds: int = 10) -> int:
    """A2 组件：rz 单向穿过 cz 的可交换消解（2026-11-10）。

    rz 与 cz 对易（cz 为对角门）：把 cz 之间的 rz 单向推向线路末端
    聚集，随后相邻 rz 合并 + H 块奇偶消解——1Q 门数进一步压缩
    （对标 qiskit CommutativeCancellation 的自主实现）。

    单向推挤（rz 只向后穿 cz）保证收敛；返回移动次数。
    """
    moved_total = 0
    for _round in range(max_rounds):
        changed = False
        i = 0
        n = len(out_ops)
        while i < n - 1:
            a, b = out_ops[i], out_ops[i + 1]
            if (a.name == "rz" and b.name == "cz"
                    and int(a.qubits[0]) in (int(b.qubits[0]),
                                             int(b.qubits[1]))):
                out_ops[i], out_ops[i + 1] = b, a   # rz 后移
                changed = True
                moved_total += 1
                i += 2
            else:
                i += 1
        # 相邻 rz 合并（对 out_ops 直接操作）
        _merge_adjacent_rz_ops(out_ops)
        _cancel_adjacent_h(out_ops)
        if not changed:
            break
    return moved_total


def _merge_adjacent_rz_ops(out_ops):
    """相邻同比特 rz 角度合并（就地；不动点循环）。"""
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out_ops) - 1:
            a, b = out_ops[i], out_ops[i + 1]
            if (a.name == "rz" and b.name == "rz"
                    and a.qubits == b.qubits):
                out_ops[i] = Inst("rz", a.qubits,
                                  (a.params[0] + b.params[0],))
                del out_ops[i + 1]
                changed = True
            i += 1


# ------------------------------------------- A3：CX 通勤对消（CommutativeCancellation 的 cx 版）
def _cx_commutes(a: Inst, b: Inst) -> bool:
    """cx(a,b) 与 op 是否对易：比特不相交恒对易；同为 cx 时同控/同目标
    对易；其余（含交叉控制 cx(a,b)/cx(b,a)、cz 共享比特、1Q 共享比特）
    均不对易。"""
    qa = a.qubits
    if set(qa).isdisjoint(b.qubits):
        return True
    if b.name != "cx":
        return False
    qb = b.qubits
    return qa[0] == qb[0] or qa[1] == qb[1]


def commute_cancel_cx(out_ops: list, window: int = 12,
                      max_rounds: int = 6) -> int:
    """A3 组件：窗口内 cx(a,b)···cx(a,b) 对消（中间门全部与之对易时）。

    对标 qiskit CommutativeCancellation（cx 自逆门的通勤对消）——
    折叠+链化后同一路径边上来自不同链的相邻 cx 是主要受益者。
    只做消解、不做交换重排（单调，必然终止）。返回消解门数。
    """
    removed = 0
    for _ in range(max_rounds):
        round_removed = 0
        i = 0
        n = len(out_ops)
        while i < n:
            a = out_ops[i]
            if a.name != "cx":
                i += 1
                continue
            jmax = min(i + window, n)
            for j in range(i + 1, jmax):
                b = out_ops[j]
                if b.name == "cx" and b.qubits == a.qubits \
                        and all(_cx_commutes(a, out_ops[k])
                                for k in range(i + 1, j)):
                    del out_ops[j]
                    del out_ops[i]
                    n -= 2
                    round_removed += 2
                    break
            else:
                i += 1
        removed += round_removed
        if round_removed == 0:
            break
    return removed


# ------------------------------------------------------------ v5 分解入口
def decompose_to_platform_v2(circ: Circuit, expand_swap: bool = True,
                             cancel_cx: bool = True,
                             fixpoint: bool = True) -> Circuit:
    """v5 平台基分解：v4 分解（cx→H-CZ-H、swap→3cx、u3→rzsx）+ 1Q 优化
    不动点循环（对标 qiskit Optimize1qGates + 交换消除）。

    与 v4（post.py decompose_to_platform）的差异：分解前多一轮 cx
    通勤对消（A3，对标 qiskit CommutativeCancellation）+ 分解后 1Q
    优化不动点（B 组件）——均语义等价、门数减少。
    cancel_cx/fixpoint 为消融开关（默认开启）。
    """
    from qtrail.pure.post import decompose_to_platform as _v4_decompose
    # A3：cx 通勤对消（对标 qiskit CommutativeCancellation）——在副本上
    # 运行，不动调用方线路（fold/链化后同一路径边上的相邻 cx 成对出现）
    work = Circuit(circ.n, name=circ.name)
    work.ops = list(circ.ops)
    work.measures = list(circ.measures)
    if cancel_cx:
        commute_cancel_cx(work.ops)
    out = _v4_decompose(work, expand_swap=expand_swap)
    if not fixpoint:
        return out
    # 1Q 优化不动点循环：游程合并为 u3 → ZXZ 再合成 → rz 穿 cz 消解
    # → rz 合并 → H 块奇偶消解，重复至稳定（4 轮上限）
    for _ in range(4):
        changed = False
        if merge_1q_runs(out.ops) > 0:
            new_ops = []
            for inst in out.ops:
                if inst.name == "u3":
                    for gname, gparams in _u3_to_rzsx(*inst.params):
                        new_ops.append(Inst(gname, inst.qubits, gparams))
                else:
                    new_ops.append(inst)
            out.ops = new_ops
            changed = True
        merge_adjacent_rz(out)
        before = len(out.ops)
        commute_rz_through_cz(out.ops)   # A2：可交换消解
        _cancel_adjacent_h(out.ops)
        changed = changed or len(out.ops) != before
        if not changed:
            break
    return out
