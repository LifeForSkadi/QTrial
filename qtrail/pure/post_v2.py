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
def _rz_mat(alpha: float) -> np.ndarray:
    """SU(2) 的 rz(α) 矩阵（K 因子相位扫描用，2026-11-13）。"""
    return np.diag([np.exp(-1j * alpha / 2), np.exp(1j * alpha / 2)])


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


def _to_cz_basis(insts: list) -> list:
    """3cx 合成序列 → 3cz 基（2026-11-13，qiskit tqp 输出结构的对齐）。

    cx = H_b·cz·H_b 展开，H_b 与相邻 1Q 矩阵乘合并；cz 之间的段
    按比特合并成 u3（对角时输出 rz）——中间旋转在 cz 基为 rz-only
    （H·ry(θ)·H = rz(θ) 到全局相位，数值验证），每块 8 个 u3 的
    参数冗余降为 4 u3 + 4 rz（qiskit 每块 11 门 vs 我们 28 门）。
    """
    # 1) 展开 cx → H_b, cz, H_b
    flat = []
    for inst in insts:
        if inst.name == "cx":
            a, b = inst.qubits
            h = [Inst("rz", (b,), (H_GATE[0],)),
                 Inst("sx", (b,)),
                 Inst("rz", (b,), (H_GATE[0],))]
            flat.extend(h)
            flat.append(Inst("cz", (a, b)))
            flat.extend(h)
        else:
            flat.append(inst)
    # 2) 按 cz 切段，段内按比特矩阵乘合并（对角 → rz，否则 u3）
    out = []
    seg = []

    def flush():
        nonlocal seg
        if not seg:
            return
        by_q = {}
        for inst in seg:
            q = inst.qubits[0]
            by_q.setdefault(q, np.eye(2, dtype=complex))
            by_q[q] = _oneq_matrix(inst.name, inst.params) @ by_q[q]
        for q, U in by_q.items():
            U = U / np.sqrt(np.linalg.det(U))
            if abs(np.trace(U)) / 2 > 1 - 1e-9:
                continue  # 恒等
            th, ph, la = _mat_to_u3_params(U)
            # 对角检测（rz-only）
            off = abs(U[0, 1])
            if off < 1e-9:
                ang = np.angle(U[0, 0] / U[1, 1])
                out.append(Inst("rz", (q,), (float(ang),)))
            else:
                out.append(Inst("u3", (q,), (th, ph, la)))
        seg = []

    for inst in flat:
        if inst.name == "cz":
            flush()
            out.append(inst)
        else:
            seg.append(inst)
    flush()
    return out


def _platformize_insts(insts: list) -> list:
    """合成输出的 u3/u1/cx → rz/sx/cz 平台门（2026-11-13）。

    直接最优分解（u3→B8 合成器最少门、u1→rz、cx→H-CZ-H），替代
    v4 再翻译的双重分解冗余——qiskit tqp 合成输出即平台门
    （块内 11 门 vs 我们的 28 门）。
    """
    out = []
    for inst in insts:
        if inst.name == "u3":
            for g, p in _u3_to_rzsx_min(*inst.params):
                out.append(Inst(g, inst.qubits, p))
        elif inst.name == "u1":
            out.append(Inst("rz", inst.qubits, inst.params))
        elif inst.name == "cx":
            a, b = inst.qubits
            h = [Inst("rz", (b,), (H_GATE[0],)),
                 Inst("sx", (b,)),
                 Inst("rz", (b,), (H_GATE[0],))]
            out.extend(h)
            out.append(Inst("cz", (a, b)))
            out.extend(h)
        else:
            out.append(inst)
    return out


def _merge_u3_chain(insts: list) -> list:
    """合成输出的 u3/u1 链合并（2026-11-13）：同比特连续的 u3/u1 →
    矩阵乘积合成单 u3（精确、全局相位剥离）；u1（rz）穿过 cx 控制位
    与后续 u3 合并：[u1(q), cx(q,r), u3(q)] → [cx(q,r), U(q)]——
    qiskit euler 合成器的等价机制（rz 与 cx 控制位对易）。
    """
    out = []
    i = 0
    n = len(insts)
    while i < n:
        inst = insts[i]
        if inst.name in ("u3", "u1") and inst.nq == 1:
            q = inst.qubits[0]
            j = i
            U = np.eye(2, dtype=complex)
            while j < n:
                nj = insts[j]
                if nj.name in ("u3", "u1") and nj.qubits[0] == q:
                    U = _oneq_matrix(nj.name, nj.params) @ U
                    j += 1
                else:
                    break
            # 穿 cx 控制位的模式：[U(q), cx(q,r), u3(q)] →
            # [cx(q,r), U'(q)]（U' = U·u3）
            if (j < n and insts[j].name == "cx"
                    and insts[j].qubits[0] == q and j + 1 < n
                    and insts[j + 1].name in ("u3", "u1")
                    and insts[j + 1].qubits[0] == q):
                cx_inst = insts[j]
                U = _oneq_matrix(insts[j + 1].name,
                                 insts[j + 1].params) @ U
                U = U / np.sqrt(np.linalg.det(U))
                if abs(np.trace(U)) / 2 > 1 - 1e-9:
                    out.append(cx_inst)      # U 恒等：只留 cx
                else:
                    out.append(cx_inst)
                    out.append(Inst("u3", (q,), _mat_to_u3_params(U)))
                i = j + 2
                continue
            if j - i > 1:
                U = U / np.sqrt(np.linalg.det(U))
                if abs(np.trace(U)) / 2 > 1 - 1e-9:
                    pass  # 恒等：删除
                else:
                    out.append(Inst("u3", (q,), _mat_to_u3_params(U)))
            else:
                out.append(inst)
            i = j
        else:
            out.append(inst)
            i += 1
    return out


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


def merge_u3_runs(out_ops: list) -> int:
    """分解前的 u3 级游程合并（2026-11-12，qv 类 1Q 缺口机制）。

    数学依据（预注册见 pending_decisions 补充三十六）：
    1. 不相交支撑对易（定理级）：u3(q) 可滑过不触碰 q 的 cx/swap，
       bubble 聚拢同比特 u3 游程
    2. U3·U3 = e^{iδ}·U3'（U(2) 封闭 / ZXZ 欧拉，N&C Thm 4.1）；
       e^{iδ}I_q = e^{iδ}I_total 是全系统全局相位，精确可丢
       （_mat_to_u3_params 的 α 剥离即此）
    3. 与台账 #11/#13/#15 的关系：那些是分解后 rz/sx 层操作
       （被 P7 sx穿cz 锁死）；本 pass 在分解前 u3 层，锁不存在

    单调：只滑动（不动 2Q 结构）+ 只合并（每对净 −1 门）。
    返回合并的 u3 对数。
    """
    # 1) bubble 滑动聚拢（固定轮数，右移；每轮把每个 u3 移到其
    # 最右阻塞位）。只滑过非 u3 门：异比特 u3 无需互相穿越
    # （对方自会滑走），也避免"两 u3 互不相交→无限互换"的
    # 无界 ping-pong（while-moved 版实测死循环）。
    for _ in range(6):
        i = 0
        n = len(out_ops)
        while i < n - 1:
            a, b = out_ops[i], out_ops[i + 1]
            if a.name == "u3" and b.name != "u3" and b.nq >= 1 \
                    and int(a.qubits[0]) not in b.qubits:
                out_ops[i], out_ops[i + 1] = b, a
            i += 1
    # 2) 相邻同比特 u3 对合并（矩阵精确乘，相位剥离）+ 收益检查：
    # 只合并分解后净减门的对——两个零角 u3（各 1-3 门）合并成
    # 通用 u3（5 门）是负收益（qv32 实测 +656 门），跳过
    from qtrail.pure.post import _u3_to_rzsx_min as _rzsx_min
    merged = 0
    i = 0
    n = len(out_ops)
    while i < n - 1:
        a, b = out_ops[i], out_ops[i + 1]
        if a.name == "u3" and b.name == "u3" \
                and a.qubits == b.qubits:
            U = _oneq_matrix("u3", b.params) @ _oneq_matrix("u3", a.params)
            new_params = _mat_to_u3_params(U)
            if len(_rzsx_min(*new_params)) \
                    < len(_rzsx_min(*a.params)) + len(_rzsx_min(*b.params)):
                out_ops[i] = Inst("u3", a.qubits, new_params)
                del out_ops[i + 1]
                n -= 1
                merged += 1
            else:
                i += 1
        else:
            i += 1
    return merged


def merge_1q_runs_v2(out_ops: list, slide_rounds: int = 6) -> int:
    """分解前完整 1Q run 合并 v2（2026-11-12，补充三十七预注册）。

    输入 = _expand_keep_u3 产物（cx→H-CZ-H、swap→3cx 已展开、
    u3 保持未分解）。机制：
    1. bubble 滑动全部 1Q 门（u3/u1/rz/sx/x 统一矩阵处理）跨
       不相交门（定理级：不相交支撑对易）——固定轮数，O(m)
    2. 同比特相邻 run 矩阵乘 → 单 u3（U(2) 封闭，相位剥离精确，
       收益检查：分解后门数 < run 长度才替换）
    调用方随后 u3→rz/sx 分解，【不做跨 cz 通勤】——台账 #18：
    跨 cz 通勤把 5 门 u3 撕成不可再合并的碎片（sx 被 P7 锁死）。

    复杂度：O(slide_rounds·m) + O(m)，无超多项式项。
    返回合并的 run 数。
    """
    _1Q = ("u3", "u1", "rz", "sx", "x")
    # 1) 滑动聚拢（只滑过非 1Q 门——异比特 1Q 无需互相穿越）
    for _ in range(slide_rounds):
        i = 0
        n = len(out_ops)
        while i < n - 1:
            a, b = out_ops[i], out_ops[i + 1]
            if a.name in _1Q and b.name not in _1Q and b.nq >= 1 \
                    and int(a.qubits[0]) not in b.qubits:
                out_ops[i], out_ops[i + 1] = b, a
            i += 1
    # 2) 同比特 run 矩阵乘合并
    from qtrail.pure.post import _u3_to_rzsx_min as _rzsx_min
    merged = 0
    i = 0
    n = len(out_ops)
    while i < n:
        inst = out_ops[i]
        if inst.name in _1Q:
            q = inst.qubits[0]
            j = i
            U = np.eye(2, dtype=complex)
            while j < n and out_ops[j].name in _1Q \
                    and out_ops[j].qubits[0] == q:
                U = _oneq_matrix(out_ops[j].name, out_ops[j].params) @ U
                j += 1
            if j - i > 1:
                # 全局相位剥离（1Q 门相位 = 全系统全局相位，精确）
                U = U / np.sqrt(np.linalg.det(U))
                if abs(np.trace(U)) / 2 > 1 - 1e-9:
                    del out_ops[i:j]          # 恒等 run 整体删除
                    n -= (j - i)
                    merged += 1
                    continue
                params = _mat_to_u3_params(U)
                if len(_rzsx_min(*params)) < (j - i):
                    out_ops[i] = Inst("u3", (q,), params)
                    del out_ops[i + 1:j]
                    n -= (j - i - 1)
                    merged += 1
            i += 1
        else:
            i += 1
    return merged


def _expand_keep_u3(ops: list, expand_swap: bool = True) -> list:
    """cx→H-CZ-H、swap→3cx 展开，u3 保持不分解（u3 级合并的前提：
    分解后 sx 被 cz 锁死 P7，合并窗口永久关闭）。O(m)。"""
    out = []
    for inst in ops:
        if inst.name == "cx":
            a, b = inst.qubits       # 控制 a、目标 b：H 作用于目标
            out.append(Inst("rz", (b,), (H_GATE[0],)))
            out.append(Inst("sx", (b,)))
            out.append(Inst("rz", (b,), (H_GATE[0],)))
            out.append(Inst("cz", (a, b)))
            out.append(Inst("rz", (b,), (H_GATE[0],)))
            out.append(Inst("sx", (b,)))
            out.append(Inst("rz", (b,), (H_GATE[0],)))
        elif inst.name == "swap" and expand_swap:
            a, b = inst.qubits
            for ctl, tgt in ((a, b), (b, a), (a, b)):
                out.append(Inst("rz", (tgt,), (H_GATE[0],)))
                out.append(Inst("sx", (tgt,)))
                out.append(Inst("rz", (tgt,), (H_GATE[0],)))
                out.append(Inst("cz", (ctl, tgt)))
                out.append(Inst("rz", (tgt,), (H_GATE[0],)))
                out.append(Inst("sx", (tgt,)))
                out.append(Inst("rz", (tgt,), (H_GATE[0],)))
        else:
            out.append(inst)
    return out


def decompose_1q_v2(ops: list) -> int:
    """v2 收尾：合并后的 u3 → rz/sx 最优分解 + 段内 rz 合并 +
    恒等移除。仅段内、无跨 cz 通勤（台账 #18 纪律）。O(m)。
    返回移除/合并门数。"""
    from qtrail.pure.post import _u3_to_rzsx_min as _rzsx_min
    out = []
    n_removed = 0
    for inst in ops:
        if inst.name == "u3":
            gates = _rzsx_min(*inst.params)
            for name, params in gates:
                out.append(Inst(name, inst.qubits, params))
        else:
            out.append(inst)
    ops[:] = out
    from qtrail.pure.post import merge_adjacent_rz as _mar
    n_removed += len(ops)
    _mar_from_ops(ops)
    n_removed -= len(ops)
    n_removed += _strip_identity_1q(ops)
    return n_removed


def _mar_from_ops(ops: list) -> int:
    """merge_adjacent_rz 的 ops 列表版（v2 收尾用，O(n) 单遍）。"""
    out = []
    i = 0
    n = len(ops)
    merged = 0
    while i < n:
        inst = ops[i]
        if inst.name == "rz" and i + 1 < n \
                and ops[i + 1].name == "rz" \
                and ops[i + 1].qubits == inst.qubits:
            th = (inst.params[0] + ops[i + 1].params[0]) % (2 * math.pi)
            out.append(Inst("rz", inst.qubits, (th,)))
            i += 2
            merged += 1
        else:
            out.append(inst)
            i += 1
    ops[:] = out
    return merged


# ------------------------------------------------- 线性分解（补充四十预注册）
_U3_H = (math.pi / 2, 0.0, math.pi)  # H = u3(π/2, 0, π)（精确，det=−1）


def _expand_to_u3cz(ops: list, expand_swap: bool = True) -> list:
    """cx → u3H·cz·u3H、swap → 3×(cx 展开)，1Q 门（u3/u1/rz/sx/x）
    原样保留。O(m)。

    关键设计（台账 #19 失败的根因修复）：H 用单 u3(π/2,0,π) 表示
    （cx = H_b·cz·H_b 精确恒等式，det 相位由 run 合并的全局相位
    剥离统一处理）——#19 把 H 展开成 rz·sx·rz 3 门后，独立 H 块
    （3 门 run）不满足收益检查（3 门 ≮ 3）无合并空间；单 u3 表示
    下 [u3, H] 2 元素 run 合并为 5 门即有收益，H 块吸收在 u3 层
    完成（P7 的 sx 锁不存在于 u3 层——台账 #11/#18/#19 的锁结构）。
    """
    out = []
    for inst in ops:
        if inst.name == "cx":
            a, b = inst.qubits
            out.append(Inst("u3", (b,), _U3_H))
            out.append(Inst("cz", (a, b)))
            out.append(Inst("u3", (b,), _U3_H))
        elif inst.name == "swap" and expand_swap:
            a, b = inst.qubits
            for ctl, tgt in ((a, b), (b, a), (a, b)):
                out.append(Inst("u3", (tgt,), _U3_H))
                out.append(Inst("cz", (ctl, tgt)))
                out.append(Inst("u3", (tgt,), _U3_H))
        elif inst.name == "h":   # h → 精确 u3 表示（防御：生产输入
            out.append(Inst("u3", inst.qubits, _U3_H))  # 无 h，语义测试含）
        else:
            out.append(inst)
    return out


def _merge_runs_u3cz(ops: list) -> int:
    """cz 分段内逐比特 1Q run 矩阵乘合并（u3 级，含 H 块吸收）。

    机制（线性语义，单遍无循环）：
    1. 以 cz（及任何多比特门）为段边界；段内 1Q 门按比特分组
       （跨比特门张量积结构天然对易，分组重排语义无损——定理级）
    2. 每比特 run 矩阵乘 U = g_k···g_1；U/√det(U) 全局相位剥离
       （1Q 恒等的相位 = 全系统全局相位，精确可丢——与
       merge_1q_runs 同一约定）；恒等 run 整体删除
    3. 收益检查：合成后门数 < 各门门数和才替换（短 run 不膨胀，
       qft 教训）——独立 H 块（3 门）保持 rz·sx·rz 形态；
       合并产物统一输出 u3（对角产物 _mat_to_u3_params 返回
       θ=0 → 基翻译时收为单 rz）
    4. 无跨 cz 通勤、无循环——台账 #6/#18 的不动点振荡
       （rz 穿走 ↔ merge 合成回）机制不存在；跨 cz 移动只允许
       cz 段内（= 无移动）
    O(m)。返回合并的 run 数。
    """
    from qtrail.pure.post import _u3_to_rzsx_min as _rzsx_min

    def gcount(inst):
        return len(_rzsx_min(*inst.params)) if inst.name == "u3" else 1

    merged = 0
    out: list = []
    i = 0
    n = len(ops)
    while i < n:
        inst = ops[i]
        if inst.nq == 1:
            # 收集段：连续 1Q 门（多比特门为边界）
            seg = []
            while i < n and ops[i].nq == 1:
                seg.append(ops[i])
                i += 1
            # 按比特分组（保持首现顺序——输出确定性）
            by_q: dict = {}
            order: list = []
            for g in seg:
                q = g.qubits[0]
                if q not in by_q:
                    by_q[q] = [g]
                    order.append(q)
                else:
                    by_q[q].append(g)
            for q in order:
                gs = by_q[q]
                if len(gs) == 1:
                    out.append(gs[0])
                    continue
                U = np.eye(2, dtype=complex)
                for g in gs:
                    U = _oneq_matrix(g.name, g.params) @ U
                U = U / np.sqrt(np.linalg.det(U))
                if abs(np.trace(U)) / 2 > 1 - 1e-9:
                    merged += 1          # 恒等 run 整体删除
                    continue
                params = _mat_to_u3_params(U)
                if len(_rzsx_min(*params)) < sum(gcount(g) for g in gs):
                    out.append(Inst("u3", (q,), params))
                    merged += 1
                else:
                    out.extend(gs)
        else:
            out.append(inst)
            i += 1
    if merged:
        ops[:] = out
    return merged


def _u3_level_to_platform(ops: list) -> list:
    """u3 → _u3_to_rzsx_min（B8 最优分解）、u1 → rz，其余原样。O(m)。"""
    from qtrail.pure.post import _u3_to_rzsx_min as _rzsx_min
    out = []
    for inst in ops:
        if inst.name == "u3":
            for g, p in _rzsx_min(*inst.params):
                out.append(Inst(g, inst.qubits, p))
        elif inst.name == "u1":
            out.append(Inst("rz", inst.qubits, inst.params))
        else:
            out.append(inst)
    return out


def _translate_v4(ops: list, expand_swap: bool = True) -> list:
    """平台基翻译（列表版）：cx→H·cz·H（H = rz(π/2)·sx·rz(π/2)
    边置式，qiskit 翻译同款）、swap→[sx⊗sx, cz]×3、u3→B8 最优
    分解、u1→rz、h→H 块、rz/sx/x 原样。O(m)。

    swap 形式（2026-11-12 对拍解剖，qv32 判据达标的关键）：
    SWAP = cz·(sx⊗sx)·cz·(sx⊗sx)·cz·(sx⊗sx) 精确（矩阵验证
    重叠 1.0、qiskit Operator 复核）——6 个 1Q + 3 cz vs
    cx 链式 18 个 1Q + 3 cz：789 swap × 12 门 ≈ qiskit 翻译
    少 9,195 门的主体（此前 gap 2,616 的根源）。
    """
    from qtrail.pure.post import H_GATE as _HG, _u3_to_rzsx_min as _rzsx_min

    def _h(q):
        return (Inst("rz", (q,), (_HG[0],)), Inst("sx", (q,)),
                Inst("rz", (q,), (_HG[0],)))

    out = []
    for inst in ops:
        if inst.name == "cx":
            a, b = inst.qubits
            out.extend(_h(b))
            out.append(Inst("cz", (a, b)))
            out.extend(_h(b))
        elif inst.name == "swap" and expand_swap:
            a, b = inst.qubits
            for _ in range(3):
                out.append(Inst("sx", (a,)))
                out.append(Inst("sx", (b,)))
                out.append(Inst("cz", (a, b)))
        elif inst.name == "u3":
            for g, p in _rzsx_min(*inst.params):
                out.append(Inst(g, inst.qubits, p))
        elif inst.name == "u1":
            out.append(Inst("rz", inst.qubits, inst.params))
        elif inst.name == "h":
            out.extend(_h(inst.qubits[0]))
        else:
            out.append(inst)
    return out


def _merge_runs_cross(ops: list, max_passes: int = 3) -> int:
    """rz/sx/x 层游程合并 + 对角跨 cz 全深度收集（线性、单调）。

    机制依据（2026-11-12 diag_linear_vs_qiskit 实测解剖）：qiskit
    O3 后处理在同一输入上的输出结构（sx≈每 H 块 1 个、4.5k 单 rz
    游程）证明其 Optimize1qGates 语义在翻译后 rz/sx 层运行：
    ① 逐比特游程收集：cz 只对其自身比特是边界（qiskit
       CommutationAnalysis 语义——不相交支撑对易，定理级）；
       游程矩阵乘合并（min 序列严格少于原长才替换）
    ② 对角累加器（全深度、单遍）：每比特的 rz 可穿过全部 cz
       （对角×对角对易，定理级）——累加器 A 吸收各游程的前导
       rz、尾 rz 移交新累加器；H 块 [rz(π/2),sx,rz(π/2)] 的
       两个 rz 各归一侧链、剩 sx 单门（3→1 门/块）；H 块链的
       rz 聚成单 rz（8×π/2 ≡ 0 → 恒等直接消失）。
       逐边界规则版（右穿/左穿/双合并）只兑现 13% 收益——[sx,rz]
       尾 rz 并入邻 H 块被 4<4 阻塞；累加器版是 qiskit 的
       collect 语义本身。

    单调性：合并只在严格减门时替换；累加器吸收使每 rz 净 −1 门
    或不变（rz(α)+rz(β)=rz(α+β) 精确）。无振荡循环（台账 #6/#18
    的振荡源于非单调的纯移动 pass + 循环回看架构；本 pass 单遍
    不回看，不动点即终止）。O(max_passes·m)。
    返回净减门数。
    """
    from qtrail.pure.post import _u3_to_rzsx_min as _rzsx_min

    def merge_run(gs):
        """run 的最少门形态：矩阵乘 → 恒等删除 / min 序列
        （严格少于原长才替换，否则原样）。"""
        if len(gs) <= 1:
            return gs
        U = np.eye(2, dtype=complex)
        for g in gs:
            U = _oneq_matrix(g.name, g.params) @ U
        U = U / np.sqrt(np.linalg.det(U))
        if abs(np.trace(U)) / 2 > 1 - 1e-9:
            return []
        seq = []
        for g, p in _rzsx_min(*_mat_to_u3_params(U)):
            seq.append(Inst(g, gs[0].qubits, p))
        return seq if len(seq) < len(gs) else gs

    def rz_valid(a):
        return a is not None and abs(a % (2 * math.pi)) > 1e-9

    total = 0
    for _pass in range(max_passes):
        # ---- 逐比特序列化：seq[q] = [(run, 边界序号, 是否 cz 边界)]
        #      cz 只关闭自身比特的 pending（不相交比特滑过——游程
        #      跨 cz 延长）；非 cz 多比特门 = 全员硬边界
        seq: dict = {}
        pending: dict = {}
        k = 0
        for inst in ops:
            if inst.nq == 2:
                if inst.name == "cz":
                    for q in inst.qubits:
                        if q in pending:
                            seq.setdefault(q, []).append(
                                (pending.pop(q), k, True))
                else:
                    for q in list(pending):
                        seq.setdefault(q, []).append(
                            (pending.pop(q), k, False))
                k += 1
            else:
                pending.setdefault(int(inst.qubits[0]), []).append(inst)
        for q, gs in pending.items():
            seq.setdefault(q, []).append((gs, k, False))
        # ---- 每比特：段内合并 + 对角累加收集（全深度单遍；
        #      非 cz 硬边界不可穿——A 必须在硬边界前 flush、
        #      尾 rz 不剥）
        # 注意（2026-11-12 负结果）：不得把 rz(π) 从游程中段剥离——
        # Z·sx ≠ −sx·Z（矩阵直验 + qiskit 通勤检查器实测均不对易），
        # 生产紧凑子空间验证 fid=0.0 捕获。正确机制 = merge_run 的
        # 矩阵乘（sx·rz(π)·sx = i·rz(π) 三合一自动坍缩）。
        before = sum(len(r) for s in seq.values() for r, _b, _c in s)
        for q, s in seq.items():
            new_s = []
            A = None
            prev_c, prev_b = True, None
            for run, b, c in s:
                # 段内合并
                if len(run) > 1:
                    run = merge_run(run)
                if not run:
                    prev_c, prev_b = c, b
                    continue
                # 硬边界前 flush（A 不得穿非 cz 门）
                if not prev_c and rz_valid(A):
                    new_s.append(([Inst("rz", (q,), (A % (2 * math.pi),))],
                                  prev_b, False))
                    A = None
                if len(run) == 1 and run[0].name == "rz":
                    # 全对角（单 rz）：进累加器（rz 跨 cz 精确对易）；
                    # 后随硬边界则立即落地
                    A = run[0].params[0] if A is None else \
                        (A + run[0].params[0]) % (2 * math.pi)
                    if not c:
                        if rz_valid(A):
                            new_s.append(([Inst("rz", (q,),
                                                (A % (2 * math.pi),))],
                                          b, c))
                        A = None
                else:
                    # 剥前导 rz → A
                    if run[0].name == "rz":
                        A = run[0].params[0] if A is None else \
                            (A + run[0].params[0]) % (2 * math.pi)
                        run = run[1:]
                    # 剥尾 rz → 新累加器（后随硬边界不剥）
                    A_next = None
                    if c and run and run[-1].name == "rz":
                        A_next = run[-1].params[0]
                        run = run[:-1]
                    # flush A（置于本 run 边界前）+ run 主体
                    if rz_valid(A):
                        new_s.append(([Inst("rz", (q,),
                                            (A % (2 * math.pi),))], b, c))
                    if run:
                        new_s.append((run, b, c))
                    A = A_next
                prev_c, prev_b = c, b
            if rz_valid(A):
                new_s.append(([Inst("rz", (q,), (A % (2 * math.pi),))],
                              k, False))
            seq[q] = new_s
        after = sum(len(r) for s in seq.values() for r, _b, _c in s)
        if after >= before:
            break
        # ---- 重建 ops：游程按边界序号归位（同一边界各比特游程的
        #      发射序 = 首现序——跨比特 1Q 门张量积结构对易，
        #      任意确定序语义等价）
        by_cz: dict = {}
        for q, s in seq.items():
            for run, b, _c in s:
                if run:
                    by_cz.setdefault(b, []).extend(run)
        new_ops = []
        k = 0
        for inst in ops:
            if inst.nq == 2:
                new_ops.extend(by_cz.pop(k, []))
                new_ops.append(inst)
                k += 1
        new_ops.extend(by_cz.pop(k, []))
        total += len(ops) - len(new_ops)
        ops[:] = new_ops
    return total


def decompose_linear(circ: Circuit, expand_swap: bool = True,
                     cancel_cx: bool = True, spec=None,
                     a2_rounds: int = 2) -> Circuit:
    """线性分解（2026-11-12 补充四十预注册：qiskit pass 顺序架构）。

    阶段（全部 O(m)·固定轮数·严格增益，无不动点循环）：
    1. CommutativeCancellation(cx 层)：A3 窗口通勤对消
    2. 基翻译（v4 同款：cx→H-CZ-H、swap→3cx、u3→ZSXXZ、u1→rz）
    3. Optimize1qGates 语义（_merge_runs_cross）：段内矩阵乘合并 +
       边界 rz 跨 cz 对角通行（含 H 块 3→1 门剥壳）——实测解剖
       （diag_linear_vs_qiskit）证明 qiskit 该机制在翻译后层运行，
       非 u3 层（第一版 u3 级段内合并三条全败、+16%）
    4. A2 块重合成（固定 a2_rounds 轮）：sx 对合并 + x 穿 cz +
       cz 聚拢（qiskit O1 同款单调机制）→ A2 解析合成 → 再翻译
       + 再合并
    5. 最小收尾：x 穿 cz + cz 全局对消 + 相邻 rz 合并 + 恒等移除
       ——无跨 cz rz 通勤循环（#6 振荡机制不存在：本路径所有操作
       严格增益或不动，单遍不回看）

    预注册判据（pending_decisions 补充四十）：qv32 1Q ≤16,500、
    depth ≤3600、合法性+logf 不劣化、heis/qft/hwb12 回归不劣化、
    O(m) 预算。不达标即回退（linear=False 默认关）。
    """
    work = Circuit(circ.n, name=circ.name)
    work.ops = list(circ.ops)
    work.measures = list(circ.measures)
    if cancel_cx:
        commute_cancel_cx(work.ops)
    ops = _translate_v4(work.ops, expand_swap=expand_swap)
    _merge_runs_cross(ops)
    for _ in range(a2_rounds if spec is not None else 0):
        _merge_sx_pairs(ops)
        slide_x_through_cz(ops, max_rounds=12)
        _cluster_cz_segments(ops)
        c = Circuit(circ.n, name=circ.name)
        c.ops = ops
        c.measures = work.measures
        c = collect_resynthesize_2q_blocks(c, spec, max_blocks=60)
        ops = _translate_v4(c.ops, expand_swap=False)
        _merge_runs_cross(ops)
    slide_x_through_cz(ops, max_rounds=12)
    _cancel_cz_global(ops)
    _mar_from_ops(ops)
    _strip_identity_1q(ops)
    work.ops = ops
    work._deps = None
    return work


def _is_h_block(ops, i):
    q = ops[i].qubits[0]
    return (ops[i].name == "rz" and ops[i + 1].name == "sx"
            and ops[i + 2].name == "rz"
            and ops[i + 1].qubits[0] == q
            and ops[i + 2].qubits[0] == q
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
        # 模式：H,rz(θ),H → rx(θ)（rz 夹在两个 H 块之间）。
        # 恒等式：H·rz(θ)·H = rx(θ) = U3(θ, −π/2, π/2)（模全局相位）；
        # 旧实现的 3 门替换 [rz(π/2),sx,rz(θ)] 只是"半个 H + rz"——
        # 语义错误（2026-11-13 经 qaoa12 fid=0.25 复现定位，修正为
        # 经 B8 合成器生成完整 rx(θ) 分解）。
        if (i + 6 < n and _is_h_block(out_ops, i)
                and out_ops[i + 3].name == "rz"
                and out_ops[i + 3].qubits[0] == out_ops[i].qubits[0]
                and _is_h_block(out_ops, i + 4)):
            q = out_ops[i].qubits[0]
            th = out_ops[i + 3].params[0]
            from qtrail.pure.post import _u3_to_rzsx_min
            rx_gates = _u3_to_rzsx_min(th, -math.pi / 2, math.pi / 2)
            repl = [Inst(g, (q,), p) for g, p in rx_gates]
            out_ops[i:i + 7] = repl
            n += len(repl) - 7
            n_changed += 1
            i += len(repl)
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


def _cancel_cz_global(out_ops: list) -> int:
    """B6+ 全局 cz 对消（对标 qiskit CommutativeCancellation 的全线路语义）。

    cz 与 cz/rz 全对易、与比特不相交的 sx/x 对易——线路按"通行段"
    划分（段边界 = 与段内 cz 共享比特的 sx/x/h 门），段内 cz 可自由
    移动：按比特对统计奇偶，偶数全消、奇数保留 1 个（置于段尾，
    保持非 cz 门原位）。O(n) 单遍。返回移除门数。
    """
    removed = 0
    out = []
    seg = []          # 当前通行段
    seg_cz_bits = set()
    seg_sx_bits = set()

    def flush():
        nonlocal removed
        if not seg:
            return
        cnt = {}
        keep_order = []
        for inst in seg:
            if inst.name == "cz":
                key = tuple(sorted(inst.qubits))
                if key not in cnt:
                    cnt[key] = 0
                    keep_order.append(key)
                cnt[key] += 1
        keep = [k for k in keep_order if cnt[k] % 2 == 1]
        noncz = [inst for inst in seg if inst.name != "cz"]
        removed += len(seg) - len(noncz) - len(keep)
        out.extend(noncz)
        out.extend(Inst("cz", k) for k in keep)
        seg.clear()
        seg_cz_bits.clear()
        seg_sx_bits.clear()

    for inst in out_ops:
        if inst.name in ("sx", "x", "h") and inst.nq == 1 \
                and int(inst.qubits[0]) in seg_cz_bits:
            flush()
            out.append(inst)
        elif inst.name in ("cz", "rz"):
            # cz 与段内不相交 sx/x 若共享比特则不可穿过 → 切断
            if inst.name == "cz" and (
                    set(int(q) for q in inst.qubits) & seg_sx_bits):
                flush()
            seg.append(inst)
            if inst.name == "cz":
                seg_cz_bits.update(int(q) for q in inst.qubits)
        elif inst.name in ("sx", "x") and inst.nq == 1 \
                and int(inst.qubits[0]) not in seg_cz_bits:
            seg.append(inst)   # 不相交：可穿过
            seg_sx_bits.add(int(inst.qubits[0]))
        else:
            flush()
            out.append(inst)
    flush()
    out_ops[:] = out
    return removed


def _cluster_cz_segments(out_ops: list) -> int:
    """同对 cz 对易聚拢（2026-11-13，qiskit O1 同款机制）。

    cz 与 cz/rz 全对易、与不相交 sx/x 对易——在通行段内把同对 cz
    重排成连续块（非 cz 门保持相对顺序）。聚拢后 A2 块收集能吃到
    完整 12-cz 块（heisenberg 类）——qiskit 的 O1 产物即此形态
    （(3,4)×12 连续）→ tqp 一步合成 3 cz。返回移动次数。
    """
    moved = 0
    out = []
    seg = []
    seg_cz_bits = set()
    seg_sx_bits = set()

    def flush():
        nonlocal moved
        if not seg:
            return
        # 段内 cz 按对聚类（cz 全对易，重排合法）：记录每对出现顺序
        order = []
        seen = set()
        for inst in seg:
            if inst.name == "cz":
                key = tuple(sorted(inst.qubits))
                if key not in seen:
                    seen.add(key)
                    order.append(key)
        groups = {k: [] for k in order}
        noncz = []
        for inst in seg:
            if inst.name == "cz":
                groups[tuple(sorted(inst.qubits))].append(inst)
            else:
                noncz.append(inst)
        # 重建：非 cz 门原位（相对序保持），cz 分组连续置于段尾
        out.extend(noncz)
        for k in order:
            out.extend(groups[k])
        seg.clear()
        seg_cz_bits.clear()
        seg_sx_bits.clear()

    for inst in out_ops:
        if inst.name in ("sx", "x", "h") and inst.nq == 1 \
                and int(inst.qubits[0]) in seg_cz_bits:
            flush()
            out.append(inst)
        elif inst.name in ("cz", "rz"):
            if inst.name == "cz" and (
                    set(int(q) for q in inst.qubits) & seg_sx_bits):
                flush()
            seg.append(inst)
            if inst.name == "cz":
                seg_cz_bits.update(int(q) for q in inst.qubits)
        elif inst.name in ("sx", "x") and inst.nq == 1 \
                and int(inst.qubits[0]) not in seg_cz_bits:
            seg.append(inst)
            seg_sx_bits.add(int(inst.qubits[0]))
        else:
            flush()
            out.append(inst)
    flush()
    moved = len(out_ops)
    out_ops[:] = out
    return moved - len(out)


def _sort_segment_1q(out_ops: list) -> int:
    """段内 1Q 通勤重排（2026-11-13）：cz 之间的 1Q 段按比特稳定分组
    （不同比特的 1Q 门对易、同比特相对顺序保持）。分组后 sx 对相邻
    可合并为 x（再穿 cz 打通聚拢）。返回重排次数。
    """
    moved = 0
    seg = []
    out = []

    def flush():
        nonlocal moved
        if not seg:
            return
        grouped = sorted(seg, key=lambda inst: int(inst.qubits[0]))
        if grouped != seg:
            moved += 1
        out.extend(grouped)
        seg.clear()

    for inst in out_ops:
        if inst.name == "cz":
            flush()
            out.append(inst)
        else:
            seg.append(inst)
    flush()
    if moved:
        out_ops[:] = out
    return moved


def _merge_sx_pairs(out_ops: list) -> int:
    """sx·sx → x（精确恒等式，2026-11-13）：1Q 段 {rz,rz,sx,sx} 合成
    后残留的 sx 对合并为 x——x 可穿 cz（带对面 rz(π)），打通聚拢
    通道（heisenberg 类 222 个 {rz2,sx2} 段阻挡同对 cz 聚拢的根因）。
    """
    removed = 0
    i = 0
    n = len(out_ops)
    while i < n - 1:
        a, b = out_ops[i], out_ops[i + 1]
        if a.name == "sx" and b.name == "sx" \
                and a.qubits == b.qubits:
            out_ops[i] = Inst("x", a.qubits)
            del out_ops[i + 1]
            n -= 1
            removed += 1
        i += 1
    return removed


def _preslide_1q(out_ops: list, max_rounds: int = 8) -> int:
    """预收集解碎片（2026-11-13）：任意 1Q 门（u1/u3/rz/sx/x）滑过
    比特不相交的门——同对 2Q 游程中外比特 1Q 门被移出，块收集才能
    吃到完整游程（qiskit O3 CommutativeCancellation 的等价前置；
    Collect2qBlocks 的 "uninterrupted" 语义下必须先行解碎片）。

    单调右滑、同比特相对序保持（不相交对易为精确恒等）。
    """
    moved_total = 0
    for _ in range(max_rounds):
        moved = 0
        i = 0
        n = len(out_ops)
        while i < n - 1:
            a, b = out_ops[i], out_ops[i + 1]
            if (a.nq == 1 and a.name in ("u1", "u3", "rz", "sx", "x")
                    and int(a.qubits[0]) not in b.qubits):
                out_ops[i], out_ops[i + 1] = b, a
                moved += 1
                i += 1
            else:
                i += 1
        moved_total += moved
        if moved == 0:
            break
    return moved_total


def slide_x_through_cz(out_ops: list, max_rounds: int = 4) -> int:
    """x 穿 cz（2026-11-13，qiskit CommutationChecker 标准规则）：
    X(q) 在 cz(q,r) 上 → 交换 + 对面插入 rz(π)。恒等式
    X_a·CZ = CZ·X_a·rz_b(π)·i（全局相位 i，fidelity 口径无影响）。
    返回移动次数。
    """
    moved = 0
    for _ in range(max_rounds):
        changed = False
        i = 0
        n = len(out_ops)
        while i < n - 1:
            a, b = out_ops[i], out_ops[i + 1]
            if (a.name == "x" and b.name == "cz" and a.nq == 1
                    and int(a.qubits[0]) in (int(b.qubits[0]),
                                             int(b.qubits[1]))):
                q = int(a.qubits[0])
                other = (int(b.qubits[1]) if q == int(b.qubits[0])
                         else int(b.qubits[0]))
                out_ops[i], out_ops[i + 1] = b, a
                out_ops.insert(i + 2, Inst("rz", (other,),
                                           (math.pi,)))
                n += 1
                moved += 1
                changed = True
                i += 3
            else:
                i += 1
        if not changed:
            break
    return moved


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
                                    max_blocks: int | None = None,
                                    lock_emissions: bool = False,
                                    use_tuccil: bool = False,
                                    min_gate: int = 5,
                                    gate_guard: bool = False) -> Circuit:
    """A2 组件：2Q 块收集 + 重合成（SWAP 消化）。

    流程：collect_2q_blocks 收集同比特对的连续 2Q 门块（含中间 1Q）→
    块酉 → KAK 初值 + torch Adam 拟合 21 参数（3 CX + 12 单比特）→
    收敛验证（overlap > 1-1e-9）→ 替换原块（否则保留原门）。

    块内 2Q 门均作用于同一对（a, b），且 (a, b) 在耦合图上（收集时
    已检查）→ 合成产物硬件合法。噪声感知选优（KAK 相位自由度）作为
    后续增强（当前先保证正确性与收益）。

    lock_emissions（2026-11-13 Tuccil 循环保护）：True 时给 Tuccil
    3-CZ 模板成功产物的每个门打 tu_locked 标记——后续轮次收集把
    它们当硬块边界（防不动点吸收，见 collect_2q_blocks）。
    """
    from qtrail.pure.synthesis import (block_params_to_insts,
                                       collect_2q_blocks, fit_block_torch,
                                       kak_decompose, _init_3cx_params,
                                       _su2_euler, _cached_3cx_params)

    ops = circ.ops
    blocks = collect_2q_blocks(circ, spec, min_gate=min_gate)
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
        if ov_i > 1 - 1e-7:
            del ops[start:end]
            continue
        try:
            # 解析合成（2026-11-13，qiskit Weyl/KAK 闭式解——
            # 替代 torch 拟合，0.9ms/块 vs 300-500 步 backward）
            from qtrail.pure.synthesis import synth_block_analytic
            new_insts = synth_block_analytic(U, pair)
        except Exception:
            continue
        # 精确性守卫（解析解应恒过；保险验证重叠）
        U2 = _block_unitary(new_insts, pair)
        ov = abs(np.vdot(U2.reshape(-1), U.reshape(-1))) / 4.0
        if ov < 1 - 1e-7:
            continue
        tu_ok = False
        if not use_tuccil:
            ops[start:end] = new_insts
            continue
        # Tuccil 3-CZ 模板精修（2026-11-13）：3cx 解转 cz 基初值 →
        # 多起点拟合选最少门（8 因子可对角化——1Q 差距 1.66x 的
        # 模板级修复：3cx 固定 12 单位 vs Tuccil 对角解 ~6 单位）
        try:
            from qtrail.pure.synthesis import fit_block_tuccil
            from qtrail.pure.post import _u3_to_rzsx_min
            czseq = _to_cz_basis(new_insts)
            init24 = np.zeros(24)
            u3s = [i for i in czseq if i.name == 'u3']
            rzs = [i for i in czseq if i.name == 'rz']
            if len(u3s) >= 4:
                init24[0:6] = u3s[0].params + u3s[1].params
                init24[18:24] = u3s[-2].params + u3s[-1].params
            for j, r in enumerate(rzs[:4]):
                init24[6 + 3 * j] = r.params[0]

            def n1q_of_tu(tv):
                return sum(len(_u3_to_rzsx_min(
                    float(tv[3 * j]), float(tv[3 * j + 1]),
                    float(tv[3 * j + 2]))) for j in range(8))

            rng_t = np.random.default_rng(1)
            best_n = None
            best_tu = None
            for s in range(12):
                init_s = (init24 + rng_t.uniform(-1.5, 1.5, 24)
                          if s > 0 else init24)
                # 分阶段 lr：粗调快收敛到邻域 + 精修（实测收敛率
                # 8/8，单 lr 5/8）
                t_s, ov_s = fit_block_tuccil(U, init_s, steps=300,
                                             lr=0.2)
                if ov_s >= 1 - 1e-7:
                    t_s, ov_s = fit_block_tuccil(U, t_s, steps=500,
                                                 lr=0.02)
                if ov_s < 1 - 1e-7:
                    continue
                n_s = n1q_of_tu(t_s)
                if best_n is None or n_s < best_n:
                    best_n, best_tu = n_s, t_s
            if best_tu is None:
                raise RuntimeError('tuccil no convergence')
            # 输出：8 因子最优分解 + 3 cz（直接平台门）
            tu_out = []
            for j in range(8):
                a, b = pair
                q = a if j in (0, 2, 4, 6) else b
                for g, gp in _u3_to_rzsx_min(
                        float(best_tu[3 * j]), float(best_tu[3 * j + 1]),
                        float(best_tu[3 * j + 2])):
                    tu_out.append(Inst(g, (q,), gp))
                if j in (1, 3, 5):
                    tu_out.append(Inst('cz', (a, b)))
            # 块级择优（2026-11-13）：Tuccil 与通用 3cx 解按平台化后
            # 1Q 数取小者——12 起点可能落次优局部极值，择优保证
            # 严格不劣化（理论：Tuccil 对角化是 1Q 下界方向，但拟合
            # 收敛无全局保证）
            n_generic = sum(1 for _i in _platformize_insts(new_insts)
                            if _i.nq == 1)
            n_tu = sum(1 for _i in tu_out if _i.nq == 1)
            if n_tu < n_generic:
                new_insts = tu_out
                tu_ok = True
        except Exception:
            pass
        # 门数守卫（gate_guard，2026-11-13）：替换仅当平台化产物
        # 严格少于原块——min_gate 放宽到 3 时，原块是已优化的
        # 3-cz 块（~32 1Q），通用 12-u3 解（~40）必须被拒；Tuccil
        # 8 因子（~20）才能通过。择优基准 = 原块而非通用解。
        if gate_guard:
            if len(_platformize_insts(new_insts)) >= len(
                    _platformize_insts(list(ops[start:end]))):
                continue
        ops[start:end] = new_insts
        if lock_emissions and tu_ok:
            for _inst in new_insts:
                _inst.tu_locked = True
    circ._deps = None
    return circ


# ------------------------------------------- A2-full：同对块全量 KAK（v6 组件）
def collect_resynthesize_2q_blocks_full(circ: Circuit, spec,
                                        max_blocks: int = 150) -> Circuit:
    """A2 全量版（v6 管线组件，对标 qiskit 2.x TwoQubitPeephole 的
    实际机制——rust 源码确认：同对 2Q 游程收集 + 酉合成 + 误差加权
    择优替换，而非 NCT 模板）。

    与 A2（阈值 3s+g≥5）的差异：每个同对游程都尝试 KAK 重合成，
    但**逐块保真度守卫**（合成产物在该对边上的误差积 ≤ 原块误差积
    才替换——误差加权语义）+ 块数上限防护（hwb 类耗时）。
    """
    from qtrail.pure.synthesis import (block_params_to_insts,
                                       collect_2q_blocks, fit_block_torch,
                                       kak_decompose, _init_3cx_params,
                                       _su2_euler, _cached_3cx_params)

    ops = circ.ops
    blocks = collect_2q_blocks(circ, spec)
    if not blocks:
        return circ
    if max_blocks is not None and len(blocks) > max_blocks:
        blocks = blocks[:max_blocks]
    err2q = spec.calib.err_2q
    med = float(np.median(list(err2q.values()))) if err2q else 1e-3

    def block_error(start, end, pair):
        a, b = pair
        e = err2q.get((a, b), err2q.get((b, a), med))
        return (1.0 - float(e)) ** sum(
            1 for i in range(start, end) if ops[i].nq == 2)

    n_replaced = 0
    for start, end, pair, n_swap, n_2q in reversed(blocks):
        if n_2q < 2:
            continue
        U = _block_unitary(ops[start:end], pair)
        if abs(np.linalg.det(U)) < 1e-10:
            continue
        U = U / np.linalg.det(U) ** 0.25  # SU(4) 规范化
        # 恒等块：整体删除
        if abs(np.trace(U)) / 4 > 1 - 1e-7:
            del ops[start:end]
            n_replaced += 1
            continue
        orig_err = block_error(start, end, pair)
        try:
            from qtrail.pure.synthesis import synth_block_analytic
            new_insts = synth_block_analytic(U, pair)
        except Exception:
            continue
        U2 = _block_unitary(new_insts, pair)
        ov = abs(np.vdot(U2.reshape(-1), U.reshape(-1))) / 4.0
        if ov < 1 - 1e-7:
            continue
        # 误差加权守卫：合成产物（3 CX + 12 1Q）的 2Q 误差积必须
        # 不劣于原块（1Q 误差由后续 B 组件统一处理）
        new_2q_err = (1.0 - float(err2q.get(pair, err2q.get(
            (pair[1], pair[0]), med)))) ** 3
        if new_2q_err >= orig_err - 1e-12 and len(new_insts) < (end - start):
            ops[start:end] = new_insts
            n_replaced += 1
    if n_replaced:
        circ._deps = None
    return circ


# ------------------------------------------------- C1：噪声感知后布局重标记
def noise_aware_relabel(circ: Circuit, spec, final_layout: dict,
                        max_passes: int = 30, layout: dict | None = None) -> int:
    """C1 组件：路由后重标记（对标 VF2PostLayout 的自主弱化版）。

    移动 = 全局标签置换的单步：物理位 p → 空位 p'（线路中所有标签为
    p 的门同时改为 p'——全局置换保语义）。合法性：每个涉及 p 的 2Q 门
    （含 swap）改后端点必须相邻；接受条件：p 相关边的校准误差和下降。
    纯标签置换不改门序列/门数/深度，只改"线路落在哪些耦合器上"。

    layout：初始布局（输入嵌入约定），随同一置换同步更新——交付
    结果的 layout/final_layout 坐标必须自洽（2026-11-12 审查发现：
    重标记只更新 final_layout 时，输入按 layout 嵌入即语义错位）。

    返回接受的移动次数。线路与 final_layout（及 layout）原地更新。
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
    used = set()          # 参与 2Q 门的位置（可移动候选）
    occupied = set()      # 全部被占用的物理位（移动目标的禁止集）
    for inst in ops:
        for q in inst.qubits:
            occupied.add(int(q))
        if inst.nq == 2:
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            nbrs.setdefault(a, set()).add(b)
            nbrs.setdefault(b, set()).add(a)
            used.add(a)
            used.add(b)
    # P0 修复（2026-11-12 审查代理捕获）：仅有 1Q 门/测量/布局端点的
    # 比特也必须计入占用——此前 used 只由 2Q 门填充，C1 会把一个
    # 逻辑比特挪到"只有 1Q 门的另一逻辑比特"上（双射破坏：两条线
    # 合并、final_layout 撞车；复现 final_layout={0:2,1:1,2:2}）。
    for m in circ.measures:
        occupied.add(int(m.qubits[0]))
    occupied.update(int(v) for v in final_layout.values())
    if layout is not None:
        occupied.update(int(v) for v in layout.values())

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
                if p_new in disabled or p_new in occupied:
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
                # 审查 P1-1 修复（2026-11-13）：measure 随标签置换
                # 同步（此前只动 ops 与 final_layout，读回错位）
                for m in circ.measures:
                    if int(m.qubits[0]) == p:
                        m.qubits = (p_new,)
                # final_layout 数值与线路标签同步
                for k, v in final_layout.items():
                    if v == p:
                        final_layout[k] = p_new
                # layout（输入嵌入）随同一置换同步（2026-11-12 审查修复）
                if layout is not None:
                    for k, v in layout.items():
                        if v == p:
                            layout[k] = p_new
                # 增量更新邻居表：p 的邻居把 p 换成 p_new
                for x in p_nbrs:
                    nbrs[x].discard(p)
                    nbrs[x].add(p_new)
                nbrs[p_new] = nbrs.pop(p)
                used.discard(p)
                used.add(p_new)
                occupied.discard(p)
                occupied.add(p_new)
                n_moved += 1
                improved = True
        if not improved:
            break
    circ._deps = None
    return n_moved


# ------------------------------------------------- L3：学习式重标记（复用 QAPolicy）
def learned_relabel(policy, spec, circ: Circuit, final_layout: dict,
                    cfg, dev="cpu", seed=0, noise_lambda=0.5,
                    n_rl_keep=3, layout: dict | None = None) -> int:
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
        """候选置换下的对数保真度 Σlog(1−ε)（不修改线路：标签视图
        计算）。2026-11-12 审查 P0-3：乘积在大型线路下溢，比较口径
        统一为 log。"""
        ssum = 0.0
        err2q = spec.calib.err_2q
        median2q = float(np.median(list(err2q.values()))) if err2q else 1e-3
        for inst in circ.ops:
            qs = [int(pi[int(q)]) for q in inst.qubits]
            if inst.nq == 2:
                a, b = qs[0], qs[1]
                if a > b:
                    a, b = b, a
                e = err2q.get((a, b), err2q.get((b, a), median2q))
                ssum += math.log(max(1.0 - float(e), 1e-12))
            else:
                ssum += math.log(max(1.0 - float(spec.calib.err_1q[qs[0]]),
                                    1e-12))
        for inst in circ.measures:
            q = int(pi[int(inst.qubits[0])])
            ssum += math.log(max(1.0 - float(spec.calib.err_ro[q]), 1e-12))
        return float(ssum)

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
        # 审查 P1-1 修复（2026-11-13）：measure 随标签置换同步
        for m in circ.measures:
            m.qubits = (int(pi[int(m.qubits[0])]),)
        for k, v in final_layout.items():
            final_layout[k] = int(pi[int(v)])
        if layout is not None:
            for k, v in layout.items():
                layout[k] = int(pi[int(v)])
        return moved

    ops = [(i.name, tuple(int(q) for q in i.qubits)) for i in circ.ops]
    graph = build_program_graph(spec.n, ops, circuit_id="relabel")
    from qtrail.pure.metrics import estimate_log_fidelity
    fid_now = estimate_log_fidelity(circ, spec.calib)

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
        # 2026-11-12 审查 P0-3：log 口径容差 1e-9（乘积口径的
        # 1e-12 绝对容差在下溢区失去意义）
        if fid_pi > fid_now + 1e-9 and (best is None or fid_pi > best[0]):
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
         且总链成本 < 3 才吸收；否则**放弃整个折叠**（返回 None，
         由调用方其他候选兜底）——2026-11-12 审查 P0 修复：保留
         swap 与内容初始坐标约定互斥（见分支注释）
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
                # 【P0 修复 2026-11-12，学术审查代理捕获】此前"保留
                # swap"分支发射 swap(arr[a],arr[b]) 并同步交换 arr——
                # 在内容初始坐标约定下（输出门标签 = arr[物理位]），
                # 物理 swap 本身就是纯重标记（只该更新 arr），再发射
                # 一个 swap = 应用两次，产物与 routed 不等价（随机
                # routed 电路 37/40 失败，管线中被选中即错）。
                # 该分支无法局部修复（保留 swap 需要物理坐标记账，
                # 与输出的内容坐标约定互斥）——正确做法 = 放弃折叠，
                # 由调用方其他候选兜底（fold_only+VF2 / fold_keep /
                # 裸路由，均语义已验证）。
                return circ, None
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


def commute_rz_through_cz_rev(out_ops: list, max_rounds: int = 10) -> int:
    """rz 向左穿过 cz 的反向推挤（正向 commute 的镜像，2026-11-13）。

    rz 与 cz 对易（cz 对角）：把 cz 右侧的 rz 向左推进相邻段，使
    cz 分隔的 1Q 段能从两侧收拢 rz 参与 u3 合成（对标 qiskit 通勤
    收集的双向语义）。返回移动次数。
    """
    moved_total = 0
    for _round in range(max_rounds):
        changed = False
        i = 0
        n = len(out_ops)
        while i < n - 1:
            a, b = out_ops[i], out_ops[i + 1]
            if (a.name == "cz" and b.name == "rz"
                    and int(b.qubits[0]) in (int(a.qubits[0]),
                                             int(a.qubits[1]))):
                out_ops[i], out_ops[i + 1] = b, a   # rz 前移
                changed = True
                moved_total += 1
                i += 2
            else:
                i += 1
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
def fold_keep_coords(circ: Circuit, final_layout: dict,
                     init_layout: dict | None = None):
    """A4+ 组件：swap 吸收 + 最终坐标重标记（2026-11-13，审查代理
    P0 修复版——qiskit VF2PostLayout/ApplyLayout 语义的正确实现）。

    审查发现（2026-11-13）："删 swap + 门不动"只保留发射时刻坐标，
    同一逻辑比特的多个门标签随时间漂移——交付判据要求每个逻辑
    比特固定在最终坐标。正确语义 = 重放 swap 维护 content[p]
    （物理位 p 的当前逻辑比特），门坐标 p → fin[content[p]]
    （最终布局坐标）。

    关键：content 初始 = init_layout 的逆（初始布局时物理位上的
    逻辑比特；缺省恒等），重放全部 swap 后收敛到 final_layout
    的逆（自洽）。审查代理玩具实验 [x(0),swap(0,1),cx(1,2)] +
    fl={0:1,1:0,2:2}：旧版 fid=0、本版 fid=1.0。

    返回 (new_circ, fin)：fin[逻辑 i] = 最终物理位（= final_layout）。
    """
    fl = dict(final_layout)
    init = dict(init_layout) if init_layout else {i: i
                                                  for i in range(circ.n)}
    content = {p: -1 for p in range(circ.n)}
    for log, phys in init.items():
        content[int(phys)] = int(log)
    out = []
    for inst in circ.ops:
        if inst.name == "swap":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            content[a], content[b] = content[b], content[a]
            continue
        qs = tuple(int(fl.get(content[int(q)], content[int(q)]))
                   for q in inst.qubits)
        out.append(Inst(inst.name, qs, inst.params))
    new_circ = Circuit(circ.n, name=circ.name)
    new_circ.ops = out
    # measure 原样（路由器已在最终坐标发射 measure——审查代理确认）
    for m_ in circ.measures:
        new_circ.measures.append(
            Inst("measure", m_.qubits, cbits=m_.cbits))
    return new_circ, fl


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
                             fixpoint: bool = True,
                             merge_u3: bool = False,
                             merge_u3_v2: bool = False,
                             spec=None, resynth_every: int = 0,
                             lock_emissions: bool = False,
                             final_sweep: bool = True,
                             final_tuccil: bool = False,
                             final_min_gate: int = 5,
                             final_gate_guard: bool = False,
                             max_rounds: int = 12,
                             h_first: bool = False,
                             linear: bool = False) -> Circuit:
    """v5 平台基分解：v4 分解（cx→H-CZ-H、swap→3cx、u3→rzsx）+ 1Q 优化
    不动点循环（对标 qiskit Optimize1qGates + 交换消除）。

    与 v4（post.py decompose_to_platform）的差异：分解前多一轮 cx
    通勤对消（A3，对标 qiskit CommutativeCancellation）+ 分解后 1Q
    优化不动点（B 组件）——均语义等价、门数减少。
    cancel_cx/fixpoint 为消融开关（默认开启）。

    linear=True（2026-11-12 补充四十预注册）：走 decompose_linear
    线性架构（qiskit pass 顺序：cx 通勤对消 → u3 级 run 合并含
    H 块吸收 → 基翻译 → 固定轮数 A2 → 最小收尾，无不动点循环）。
    与 fixpoint 路径互斥（其余 fixpoint 参数忽略）。
    """
    if linear:
        return decompose_linear(circ, expand_swap=expand_swap,
                                cancel_cx=cancel_cx, spec=spec)
    from qtrail.pure.post import decompose_to_platform as _v4_decompose
    # A3：cx 通勤对消（对标 qiskit CommutativeCancellation）——在副本上
    # 运行，不动调用方线路（fold/链化后同一路径边上的相邻 cx 成对出现）
    work = Circuit(circ.n, name=circ.name)
    work.ops = list(circ.ops)
    work.measures = list(circ.measures)
    if cancel_cx:
        commute_cancel_cx(work.ops)
    if merge_u3:
        # u3 级游程合并（2026-11-12，qv 类机制，补充三十六预注册）：
        # 分解前把同比特连续 u3 乘成单 u3——分解后 sx 被 cz 锁死
        # （P7），此窗口是唯一合并点
        merge_u3_runs(work.ops)
    if merge_u3_v2:
        # v2（补充三十七预注册）：展开保留 u3 → 全类 1Q 滑动 →
        # 矩阵级 run 合并 → 分解 + 段内收尾。无跨 cz 通勤
        # （台账 #18 纪律）——O(m) 预算
        work2 = Circuit(circ.n, name=circ.name)
        work2.ops = _expand_keep_u3(work.ops, expand_swap=expand_swap)
        work2.measures = list(circ.measures)
        merge_1q_runs_v2(work2.ops)
        decompose_1q_v2(work2.ops)
        work2.strip_ids()
        return work2
    out = _v4_decompose(work, expand_swap=expand_swap)
    if not fixpoint:
        return out
    # 1Q 优化不动点循环：游程合并为 u3 → ZXZ 再合成 → rz 穿 cz 消解
    # → rz 合并 → H 块奇偶消解，重复至稳定（12 轮上限；实测 hwb10
    # 4→12 轮 1Q −19% 且 20 轮仍在缓降——提前退出兜底墙钟）
    for rnd in range(max_rounds):
        changed = False
        _sort_segment_1q(out.ops)   # 段内重排：同比特游程相邻，
                                    # merge 才能吃跨比特段的长游程
        if h_first:
            # H 优先组 v2（实验参数，2026-11-13）：仅 H 组件——
            # 滑动聚拢 + H·H 对消 + H·rz·H → rx（qiskit
            # CommutativeCancellation 的 H 自逆对消语义）。
            # 不含 commute（台账 #6：深层推挤坏合成——h_first v1
            # 失败原因）。恒等式：B4/B5/C3，全精确。
            slide_1q_through(out.ops)
            _cancel_adjacent_h(out.ops)
            _absorb_h(out.ops)
            _strip_identity_1q(out.ops)
        _merge_sx_pairs(out.ops)    # sx·sx = x（每轮消化）
        slide_x_through_cz(out.ops, max_rounds=4)  # x 穿 cz 打通段边界
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
        # A2 双向段内聚集：每轮只穿 1 层 cz（rz 进入相邻段即被后续
        # merge 合成消耗——qiskit 通勤收集语义；深层推挤会把 rz 与
        # sx 分离、合成失效，实测 hwb10 1Q 更差）
        commute_rz_through_cz(out.ops, max_rounds=1)
        commute_rz_through_cz_rev(out.ops, max_rounds=1)
        _absorb_h(out.ops)               # B5：H·rz·H → rx（H 吸收）
        _cancel_cz_global(out.ops)       # B6+：cz 全局对消（全线路通行段）
        _cancel_adjacent_h(out.ops)
        _strip_identity_1q(out.ops)      # B4：恒等门移除
        changed = changed or len(out.ops) != before
        # A2 块重合成进循环（2026-11-13，对标 qiskit O2 的
        # TwoQubitPeepholeOptimization + FixedPoint 迭代）：合成产生
        # 新 1Q → 下轮 1Q 优化移走 → 2Q 聚拢 → 更大块再合成。
        # 每 resynth_every 轮一次（成本控制：torch 拟合 ~10s/块）
        if (resynth_every > 0 and spec is not None
                and (rnd + 1) % resynth_every == 0):
            # 段内 1Q 重排 → sx 对合并 → x 穿 cz 打通段边界 →
            # 同对 cz 聚拢（qiskit O1 同款机制）→ A2 吃完整 12-cz 块
            _sort_segment_1q(out.ops)
            _merge_sx_pairs(out.ops)
            slide_x_through_cz(out.ops, max_rounds=12)
            _cluster_cz_segments(out.ops)
            out2 = collect_resynthesize_2q_blocks(out, spec, max_blocks=40,
                                                 lock_emissions=lock_emissions)
            # A2 合成器输出 cx/u1 非平台门——立即再翻译（否则循环
            # 早停后残留非平台门，2026-11-13）
            # 直接平台化（2026-11-13）：合成输出 u3/u1/cx → rz/sx/cz
            # 最优分解，替代 v4 再翻译的双重分解冗余
            out2.ops = _platformize_insts(out2.ops)
            out2._deps = None
            changed = changed or len(out2.ops) != len(out.ops)
            out = out2
        if not changed:
            break
    # 最后一轮 A2（2026-11-13）：循环收敛后把剩余同对块再合成一次
    if resynth_every > 0 and spec is not None:
        _cluster_cz_segments(out.ops)
        out = collect_resynthesize_2q_blocks(out, spec, max_blocks=60,
                                             lock_emissions=lock_emissions,
                                             use_tuccil=final_tuccil,
                                             min_gate=final_min_gate,
                                             gate_guard=final_gate_guard)
        # 最终平台化 + 1Q 收尾（2026-11-13 修复）：最后 A2 输出的
        # u3/u1/cx 是非平台门——此前直接返回（口径虚高：u3 按 1 个
        # 1Q 门计误差，真机不可执行）。平台化后补 2 轮 1Q 吸收
        # （u3→rzsx 与 cx→H-CZ-H 的新生游程合并）。
        # final_sweep=False（消融开关）：验证收尾是否为 Tuccil
        # 吸收者。
        out.ops = _platformize_insts(out.ops)
        out._deps = None
        for _ in range(2 if final_sweep else 0):
            _sort_segment_1q(out.ops)
            _merge_sx_pairs(out.ops)
            slide_x_through_cz(out.ops, max_rounds=12)
            if merge_1q_runs(out.ops) > 0:
                new_ops = []
                for inst in out.ops:
                    if inst.name == "u3":
                        for gname, gparams in _u3_to_rzsx_min(*inst.params):
                            new_ops.append(Inst(gname, inst.qubits, gparams))
                    else:
                        new_ops.append(inst)
                out.ops = new_ops
            merge_adjacent_rz(out)
            slide_1q_through(out.ops)
            commute_rz_through_cz(out.ops, max_rounds=1)
            commute_rz_through_cz_rev(out.ops, max_rounds=1)
            _absorb_h(out.ops)
            _cancel_cz_global(out.ops)
            _cancel_adjacent_h(out.ops)
            _strip_identity_1q(out.ops)
    return out
