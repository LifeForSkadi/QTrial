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
from qtrail.pure.post import (H_GATE, _u3_to_rzsx, _u3_to_rzsx_min,
                             absorb_trailing,
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
                # 恒等游程整体删除（x·x、sx⁴、H² 类；对标
                # RemoveIdentityEquivalent）
                if abs(np.trace(U)) / 2 > 1 - 1e-9:
                    merged += 1
                else:
                    params = _mat_to_u3_params(U)
                    # 收益检查（2026-11-09 自审计修复，v6 移植缺陷）：
                    # 只有合成后的门数少于原游程才合并——短游程（2-4
                    # 门）合成 ZSXXZ 5 门反而膨胀（qft5 实测 +10 门）；
                    # 长游程（>5 门）与 H 形游程的收益保留
                    if len(_u3_to_rzsx_min(*params)) < (j - i):
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


def slide_1q_through(out_ops: list, max_rounds: int = 6) -> int:
    """B7 组件：1Q 门跨门通勤滑动（对标 Optimize1qGates 的核心语义）。

    rz/sx/x 与"比特不相交"的任意门对易：把 1Q 门滑过不相交的 2Q
    门，使同一比特的 1Q 门聚拢成相邻游程——随后 merge_1q_runs 可
    合并合成。单调：只滑动（每轮净移动 ≥1 才继续），绝不凭空重排
    2Q 门。返回滑动次数。
    """
    moved_total = 0
    for _ in range(max_rounds):
        moved = 0
        i = 0
        n = len(out_ops)
        while i < n - 1:
            a, b = out_ops[i], out_ops[i + 1]
            if a.name in ("rz", "sx", "x") and a.nq == 1 \
                    and b.nq >= 1 and int(a.qubits[0]) not in b.qubits:
                # a 与 b 比特不相交 → 对易，a 右移
                out_ops[i], out_ops[i + 1] = b, a
                moved += 1
                i += 1
            else:
                i += 1
        moved_total += moved
        if moved == 0:
            break
    return moved_total


def _strip_identity_1q(out_ops: list) -> int:
    """B4 组件：恒等 1Q 门移除（对标 RemoveIdentityEquivalent）。

    rz 角度模 2π 为 0 的门删除；相邻同比特 x·x = I 对消；4 个相邻
    sx = I（sx⁴ 恒等，sx 自逆）。返回移除门数。
    """
    removed = 0
    n = len(out_ops)
    i = 0
    while i < n:
        inst = out_ops[i]
        if inst.name == "rz":
            th = inst.params[0] % (2 * math.pi)
            if abs(th) < 1e-9 or abs(th - 2 * math.pi) < 1e-9:
                del out_ops[i]
                n -= 1
                removed += 1
                continue
        if inst.name == "x" and i + 1 < n and out_ops[i + 1].name == "x" \
                and out_ops[i + 1].qubits == inst.qubits:
            del out_ops[i:i + 2]
            n -= 2
            removed += 2
            continue
        if inst.name == "sx" and i + 3 < n \
                and all(out_ops[i + k].name == "sx"
                        and out_ops[i + k].qubits == inst.qubits
                        for k in range(1, 4)):
            del out_ops[i:i + 4]
            n -= 4
            removed += 4
            continue
        i += 1
    return removed


def _absorb_h(out_ops: list) -> int:
    """B5 组件：H 吸收（对标 Optimize1qGates 的 H·rz·H = rx 语义）。

    恒等式：rz(θ)·H = H·rx(θ)、H·rz(θ) = rx(θ)·H。合并方向取
    "向 rx 坍缩"：H,rz(θ),H → rx(θ)（7 门 → ZXZ 3 门）。实现为
    窗口重写：H 块 (rz(π/2),sx,rz(π/2)) 与其相邻 rz 合并为
    rx(θ) = rz(π/2),sx,rz(θ)（ZXZ 形），再把新 rz 并入相邻游程。
    只消解不凭空重排——逐窗口扫描，单调（每步净减 ≥2 门）。
    """
    n_changed = 0
    i = 0
    n = len(out_ops)
    while i < n:
        # 模式：H,rz(θ),H → rx(θ)（rz 夹在两个 H 块之间）
        if (i + 4 < n and _is_h_block(out_ops, i)
                and out_ops[i + 3].name == "rz"
                and out_ops[i + 3].qubits[0] == out_ops[i].qubits[0]
                and _is_h_block(out_ops, i + 4)):
            q = out_ops[i].qubits[0]
            th = out_ops[i + 3].params[0]
            # rx(θ) = rz(π/2), sx, rz(θ)
            out_ops[i:i + 7] = [Inst("rz", (q,), (H_GATE[0],)),
                                Inst("sx", (q,)),
                                Inst("rz", (q,), (th,))]
            n -= 4
            n_changed += 1
            i += 3
            continue
        i += 1
    return n_changed


def _cancel_adjacent_cz(out_ops: list) -> int:
    """B6 组件：cz 相邻对消（cz 全对易、cz·cz = I）。

    对标 CommutativeCancellation 的 cz 部分：补全 cx·cx 消解链
    （H 对消 → cz 对消 → H 对消）。窗口内同对 cz 且中间全为对易门
    （cz/对角 1Q）时成对消除。
    """
    removed = 0
    i = 0
    n = len(out_ops)
    while i < n:
        a = out_ops[i]
        if a.name != "cz":
            i += 1
            continue
        jmax = min(i + 8, n)
        for j in range(i + 1, jmax):
            b = out_ops[j]
            if b.name == "cz" and b.qubits == a.qubits \
                    and all(out_ops[k].name in ("cz", "rz")
                            for k in range(i + 1, j)):
                del out_ops[j]
                del out_ops[i]
                n -= 2
                removed += 2
                break
        else:
            i += 1
    return removed


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


# ------------------------------------------- A4：纯坐标折叠（fold_only，无链化）
def fold_only(circ: Circuit):
    """A4 组件：纯置换折叠——arr 坐标跟踪、删除全部 swap、门按 arr
    重标记，**不做链化、不做逐 swap 判定**（恒正确：代数恒等式）。

    产物 = 原线路在初始布局坐标下的表示（长程门保持长程、非法性由
    后续组件处理）。为全局重标记（relabel_vf2.global_relabel）提供
    语义正确的前置：固定置换只能作用于坐标稳定的线路。

    返回 (new_circ, arr)：arr = 最终坐标置换（fold 同款语义）。
    """
    ops = circ.ops
    arr = np.arange(circ.n, dtype=np.int64)
    out = []
    for inst in ops:
        if inst.name == "swap":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            arr[a], arr[b] = arr[b], arr[a]
            continue
        qs = tuple(int(arr[int(q)]) for q in inst.qubits)
        out.append(Inst(inst.name, qs, inst.params))
    new_circ = Circuit(circ.n, name=circ.name)
    new_circ.ops = out
    for m_ in circ.measures:
        new_circ.measures.append(
            Inst("measure", (int(arr[int(m_.qubits[0])]),), cbits=m_.cbits))
    return new_circ, arr


# ------------------------------------------- A5：长程门链化（独立 pass）
def chain_long_gates(circ: Circuit, spec, d_max: int | None = None):
    """A5 组件：非相邻 2Q 门 LNN 链化（噪声感知选路；cz 用 H 包裹）。

    从 fold_and_relink 抽出的独立链化（fold 语义：链矩阵已验证）。
    默认全部链化（d_max=None）：残留长程门会造成硬件非法，而链化
    膨胀由调用方非劣化开关兜底（v4 回退永远合法）。返回 (新线路,
    链化数)。
    """
    from qtrail.pure.lnn import noise_aware_chain
    out = []
    n_chained = 0
    for inst in circ.ops:
        if inst.nq != 2 or inst.name == "swap":
            out.append(inst)
            continue
        q0, q1 = int(inst.qubits[0]), int(inst.qubits[1])
        if spec.adj[q0, q1]:
            out.append(inst)
            continue
        if d_max is not None and int(spec.dist[q0, q1]) > d_max:
            out.append(inst)
            continue
        gates, _ = noise_aware_chain(spec, q0, q1)
        if gates is None:
            out.append(inst)
            continue
        n_chained += 1
        if inst.name == "cx":
            for gc, gt in gates:
                out.append(Inst("cx", (int(gc), int(gt))))
        else:  # cz(a,b) = H(b)·cx 链·H(b)
            for gname, gp in (("rz", (math.pi / 2,)), ("sx", ()),
                              ("rz", (math.pi / 2,))):
                out.append(Inst(gname, (q1,), gp))
            for gc, gt in gates:
                out.append(Inst("cx", (int(gc), int(gt))))
            for gname, gp in (("rz", (math.pi / 2,)), ("sx", ()),
                              ("rz", (math.pi / 2,))):
                out.append(Inst(gname, (q1,), gp))
    new_circ = Circuit(circ.n, name=circ.name)
    new_circ.ops = out
    new_circ.measures = list(circ.measures)
    return new_circ, n_chained


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
                    for gname, gparams in _u3_to_rzsx_min(*inst.params):
                        new_ops.append(Inst(gname, inst.qubits, gparams))
                else:
                    new_ops.append(inst)
            out.ops = new_ops
            changed = True
        merge_adjacent_rz(out)
        before = len(out.ops)
        slide_1q_through(out.ops)        # B7：1Q 跨门通勤聚拢
        commute_rz_through_cz(out.ops)   # A2：可交换消解
        _absorb_h(out.ops)               # B5：H·rz·H → rx（H 吸收）
        _cancel_adjacent_cz(out.ops)     # B6：cz·cz = I（补全 cx·cx 消解链）
        _cancel_adjacent_h(out.ops)
        _strip_identity_1q(out.ops)      # B4：恒等门移除
        changed = changed or len(out.ops) != before
        if not changed:
            break
    return out
