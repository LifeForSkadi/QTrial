"""自研路由后处理栈（qiskit-free）：置换吸收 + 对易推挤 + 门消解。

复刻 qiskit 优化阶段的公开机制子集（全部为公开论文/教科书算法）：
1. swap 对易推挤：swap(a,b) 与不触碰 a/b 的门交换顺序，推到线路末端
2. swap;swap 消解：同边相邻 swap 对相互抵消
3. 尾部置换吸收：末端 swap 直接删除，最终布局重标记
4. cx 对消解：同比特相邻 cx;cx 抵消；u1 合并（角度相加）
5. CZ 基分解：cx → H-CZ-H（H = u3(pi/2,0,pi)）；swap → 3×cx 展开

正确性：全部变换为保语义等价变换，由态矢量等价测试保证。
"""
from __future__ import annotations

import math

from qtrail.pure.circuit import Circuit, Inst

H_GATE = (math.pi / 2, 0.0, math.pi)  # H 的 u3 参数


def _u3_to_rzsx(th: float, ph: float, la: float) -> list[tuple]:
    """U3(θ,φ,λ) → [(name, params)] 序列（平台 RZ 约定：
    RZ(a)=diag(e^{−ia/2}, e^{ia/2})，与 qiskit 一致）。

    通用模板（qiskit ZSXXZ 实测同款，数值验证 2.5e-16）：
      U3(θ,φ,λ) = RZ(λ)·SX·RZ(θ+π)·SX·RZ(φ+3π)（模全局相位）
    特例：H → rz(π/2),sx,rz(π/2)（3 门）；X → x；θ=0 → 纯 RZ(φ+λ)。
    """
    if abs(th - H_GATE[0]) < 1e-9 and abs(ph) < 1e-9 \
            and abs(la - math.pi) < 1e-9:
        return [("rz", (H_GATE[0],)), ("sx", ()), ("rz", (H_GATE[0],))]
    if abs(th - math.pi) < 1e-9 and abs(ph) < 1e-9 and abs(la - math.pi) < 1e-9:
        return [("x", ())]
    if abs(math.sin(th / 2)) < 1e-12:  # θ=0：u3(0,φ,λ)=RZ(φ+λ)
        return [("rz", (float((ph + la) % (2 * math.pi)),))]
    return [("rz", (float(la),)), ("sx", ()),
            ("rz", (float(th + math.pi),)), ("sx", ()),
            ("rz", (float(ph + 3 * math.pi),))]


def _swap_past(swap_qs: tuple, gate: Inst) -> Inst | None:
    """swap 能否穿过 gate 移到其后：返回重写后的 gate（None = 不能穿）。

    规则（全部为 SWAP 共轭等价变换，语义保持）：
      SWAP(a,b); CX(a,c) → CX(b,c); SWAP(a,b)   （单边触碰：重标记）
      SWAP(a,b); CX(a,b) → CX(b,a); SWAP(a,b)   （双边触碰：控制/靶翻转）
      SWAP(a,b); R(q)     → R(q');   SWAP(a,b)  （单比特门：重标记）
      不相交门直接对易。
    """
    a, b = swap_qs
    qs = gate.qubits
    if gate.name in ("barrier", "measure"):
        return None
    if gate.name == "cx":
        if a not in qs and b not in qs:
            return gate
        if a in qs and b in qs:
            return Inst("cx", (qs[1], qs[0]), gate.params)
        return Inst("cx", tuple(b if q == a else (a if q == b else q)
                                 for q in qs), gate.params)
    if gate.nq == 2:  # cz 等对称门：单边触碰重标记；双边触碰不变
        if a not in qs and b not in qs:
            return gate
        if a in qs and b in qs:
            return gate
        return Inst(gate.name, tuple(b if q == a else (a if q == b else q)
                                     for q in qs), gate.params)
    if gate.nq == 1:
        q = qs[0]
        if q == a:
            return Inst(gate.name, (b,), gate.params)
        if q == b:
            return Inst(gate.name, (a,), gate.params)
        return gate
    return None


def push_swaps(circ: Circuit, max_rounds: int = 200) -> Circuit:
    """swap 向末端推挤（含穿过触碰门的重写）+ swap;swap 消解。

    单向推挤（只向后）避免振荡；到不动点收敛。
    Numba 可用时走编译内核（post_numba，逐位等价），否则纯 Python。
    """
    from qtrail.pure.post_numba import has_numba, numba_push_swaps
    if has_numba() and circ.ops:
        circ.ops = numba_push_swaps(circ, max_rounds)
        circ._deps = None
        return circ
    changed = True
    rounds = 0
    while changed and rounds < max_rounds:
        rounds += 1
        changed = False
        ops = circ.ops
        i = 0
        while i < len(ops) - 1:
            a, b = ops[i], ops[i + 1]
            if a.name == "swap":
                if b.name == "swap" and set(a.qubits) == set(b.qubits):
                    del ops[i:i + 2]           # swap;swap 抵消
                    changed = True
                    continue
                new_b = _swap_past(a.qubits, b)
                if new_b is not None:
                    ops[i] = new_b            # 门重写后放回
                    ops[i + 1] = a            # swap 后移
                    changed = True
                    i += 1
                    continue
            i += 1
    circ._deps = None
    return circ


def absorb_trailing(circ: Circuit, final_layout: dict) -> int:
    """删除末端 swap（其比特在后续无任何门），并重标记最终布局。

    返回被吸收的 swap 数。
    """
    ops = circ.ops
    absorbed = 0
    while True:
        # 最后一个 swap 之后是否还有触碰其比特的门
        last_swap = None
        for i in range(len(ops) - 1, -1, -1):
            if ops[i].name == "swap":
                last_swap = i
                break
        if last_swap is None:
            break
        a, b = ops[last_swap].qubits
        touched = set()
        for inst in ops[last_swap + 1:]:
            touched |= set(inst.qubits)
        if (a in touched) or (b in touched):
            break
        del ops[last_swap]
        absorbed += 1
        for logical, phys in final_layout.items():
            if phys == a:
                final_layout[logical] = b
            elif phys == b:
                final_layout[logical] = a
    circ._deps = None
    return absorbed


def cancel_cx(circ: Circuit) -> int:
    """相邻 cx 对消解 + u1 角度合并。返回消解/合并次数。

    注意：cx 抵消要求**同向同对**（cx(a,b)·cx(a,b)=I）；反向对
    cx(a,b)·cx(b,a) 之积不是恒等，不可抵消（2026-08-22 修复：
    原无序集合比较误消反向对）。u1 为对角门，同对合并恒合法。
    swap 抵消在 push_swaps 中按无序对（SWAP 为对称门）。"""
    ops = circ.ops
    removed = 0
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(ops) - 1:
            a, b = ops[i], ops[i + 1]
            if a.name == "cx" and b.name == "cx" \
                    and a.qubits == b.qubits:
                del ops[i:i + 2]
                removed += 1
                changed = True
                continue
            if a.name == "u1" and b.name == "u1" and a.qubits == b.qubits:
                ops[i] = Inst("u1", a.qubits, (a.params[0] + b.params[0],))
                del ops[i + 1]
                removed += 1
                changed = True
                continue
            i += 1
    circ._deps = None
    return removed


def post_route(circ: Circuit, final_layout: dict,
               do_cancel: bool = True) -> tuple[Circuit, int]:
    """完整后处理：推挤 → 消解 → 吸收。返回 (circuit, 被消除的 swap 数)。"""
    swaps_before = circ.count("swap")
    push_swaps(circ)
    if do_cancel:
        cancel_cx(circ)
        push_swaps(circ)          # 消解后再次推挤
    absorbed = absorb_trailing(circ, final_layout)
    return circ, swaps_before - circ.count("swap")


def merge_adjacent_rz(circ: Circuit) -> int:
    """相邻 rz 角度合并（qiskit 1Q 优化的对应物）。返回合并次数。"""
    ops = circ.ops
    merged = 0
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(ops) - 1:
            a, b = ops[i], ops[i + 1]
            if a.name == "rz" and b.name == "rz" and a.qubits == b.qubits:
                ops[i] = Inst("rz", a.qubits, (a.params[0] + b.params[0],))
                del ops[i + 1]
                merged += 1
                changed = True
                continue
            i += 1
    circ._deps = None
    return merged


def decompose_to_platform(circ: Circuit, expand_swap: bool = True) -> Circuit:
    """分解到平台基 [rz, sx, x, cz]（与 qiskit 支撑版完全同口径）：
    H = RZ(π/2)·SX·RZ(π/2)（qiskit 2.5 实测同款）、u1→RZ、cx→H-CZ-H、
    swap→3 cx 各自展开。返回新电路（原电路不变）。"""
    out = Circuit(circ.n, name=circ.name)
    for inst in circ.ops:
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
            # swap(a,b) = cx(a,b); cx(b,a); cx(a,b)
            for ctl, tgt in ((a, b), (b, a), (a, b)):
                out.append(Inst("rz", (tgt,), (H_GATE[0],)))
                out.append(Inst("sx", (tgt,)))
                out.append(Inst("rz", (tgt,), (H_GATE[0],)))
                out.append(Inst("cz", (ctl, tgt)))
                out.append(Inst("rz", (tgt,), (H_GATE[0],)))
                out.append(Inst("sx", (tgt,)))
                out.append(Inst("rz", (tgt,), (H_GATE[0],)))
        elif inst.name == "u3":
            q = inst.qubits[0]
            for name, params in _u3_to_rzsx(*inst.params):
                out.append(Inst(name, (q,), params))
        elif inst.name == "u1":
            out.append(Inst("rz", inst.qubits, (inst.params[0],)))
        else:
            out.append(inst)
    for inst in circ.measures:
        out.append(inst)
    merge_adjacent_rz(out)
    return out


# 兼容别名（早期调用方）
decompose_to_cz = decompose_to_platform
