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
    import numpy as _np
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
    """相邻 cx 对消解 + u1 角度合并。返回消解/合并次数。"""
    ops = circ.ops
    removed = 0
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(ops) - 1:
            a, b = ops[i], ops[i + 1]
            if a.name == "cx" and b.name == "cx" \
                    and set(a.qubits) == set(b.qubits):
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


def post_route(circ: Circuit, final_layout: dict, spec=None,
               do_cancel: bool = True) -> tuple[Circuit, int]:
    """完整后处理（**硬件可执行口径**）：连通性感知推挤 → cx/u1 消解 →
    尾置换块整体吸收。返回 (circuit, 被消除的 swap 数)。

    正确性口径（2026-08-22 真机校验教训）：语义等价不等于硬件可执行——
    把 2Q 门共轭重标记到新比特对（fold_swaps / 无约束推挤）会破坏
    邻近性约束（天衍平台 qcis_check_regular 实测拒绝）。本后处理只在
    **保连通性**的等价变换下操作：
      1. 推挤：swap 穿越 1Q 门（重标记，无约束）与不相交 2Q 门
         （纯对易，不改对）总是合法；触碰 2Q 门仅当重标记后比特对
         仍在耦合图上才允许（spec.adj 校验），否则阻塞；
      2. 同边相邻 swap 对抵消；cx;cx 抵消与 u1 合并；
      3. 尾置换块整体吸收：末段纯 swap 块（其后无任何门）的净置换
         P 整体并入 final_layout 与测量位置——合法（只改输出标记）。
    残余 swap 即真实所需的路由交换（qiskit 的 0 残余来自其优化级
    对长程 2Q 门的近邻再合成，属门级重合成机制，非标记操作）。
    """
    swaps_before = circ.count("swap")
    if spec is not None:
        push_swaps_safe(circ, spec)
        if do_cancel:
            cancel_cx(circ)
            push_swaps_safe(circ, spec)   # 消解后再次推挤
    elif do_cancel:
        cancel_cx(circ)
    absorbed = absorb_trailing_block(circ, final_layout)
    return circ, swaps_before - circ.count("swap")


def _amat_flat(spec):
    import numpy as np
    amat = np.zeros((spec.n, spec.n), dtype=bool)
    for a in range(spec.n):
        for b in range(a + 1, spec.n):
            if spec.adj[a, b]:
                amat[a, b] = amat[b, a] = True
    return amat


def _swap_past_safe(swap_qs: tuple, gate: Inst, amat) -> Inst | None:
    """连通性感知的 swap 穿越重写（None = 阻塞，swap 留原地）。

    与 _swap_past 的区别：触碰 2Q 门的重标记仅在目标比特对仍在耦合
    图上时返回（否则阻塞）；不相交 2Q 门原样返回（纯对易）；
    双边触碰的 cx 控/靶翻转比特对不变、恒合法。
    """
    a, b = swap_qs
    qs = gate.qubits
    if gate.name == "cx":
        if a not in qs and b not in qs:
            return gate
        if a in qs and b in qs:
            return Inst("cx", (qs[1], qs[0]), gate.params)  # 同对翻转：恒合法
        new_qs = tuple(b if q == a else (a if q == b else q) for q in qs)
        return Inst("cx", new_qs, gate.params) if amat[new_qs] else None
    if gate.nq == 2:  # cz / 异边 swap
        if a not in qs and b not in qs:
            return gate
        if a in qs and b in qs:
            return gate
        new_qs = tuple(b if q == a else (a if q == b else q) for q in qs)
        if gate.name == "swap":
            return Inst("swap", new_qs)           # swap 无邻近约束
        return Inst(gate.name, new_qs, gate.params) if amat[new_qs] else None
    if gate.nq == 1:
        q = qs[0]
        if q == a:
            return Inst(gate.name, (b,), gate.params)
        if q == b:
            return Inst(gate.name, (a,), gate.params)
        return gate
    return None


def push_swaps_safe(circ: Circuit, spec, max_rounds: int = 200) -> Circuit:
    """连通性感知单向推挤（Python 参考实现；Numba 版逐位等价）。

    只向后推、阻塞即停：被阻塞的 swap 与其右侧门永久保持相对位置
    （该门对不变），同边相邻 swap 对抵消。实测大线路数轮内收敛
    （阻塞使各 swap 一轮内各自就位，不再有旧推挤的 O(交换数) 轮链）。
    """
    amat = _amat_flat(spec)
    try:
        from qtrail.pure.post_numba import has_numba, numba_push_swaps_safe
        if has_numba() and circ.ops:
            circ.ops = numba_push_swaps_safe(circ, amat, max_rounds)
            circ._deps = None
            return circ
    except Exception:  # noqa: BLE001 内核不可用时走纯 Python（结果一致）
        pass
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
                    del ops[i:i + 2]
                    changed = True
                    continue
                new_b = _swap_past_safe(a.qubits, b, amat)
                if new_b is not None:
                    ops[i] = new_b
                    ops[i + 1] = a
                    changed = True
                    i += 1
                    continue
            i += 1
    circ._deps = None
    return circ


def absorb_trailing_block(circ: Circuit, final_layout: dict) -> int:
    """尾置换块整体吸收：末段纯 swap 块的净置换并入最终布局与测量。

    与旧 absorb_trailing 的区别：不要求每个 swap 的比特在后续无触碰
    ——整个尾块（其后无任何非 swap 门）作为置换 P 一次性吸收
    （P 作用于输出基标记，不改变任何门的比特对，硬件可执行性不变）。
    """
    import numpy as np
    ops = circ.ops
    k = len(ops)
    while k > 0 and ops[k - 1].name == "swap":
        k -= 1
    if k == len(ops):
        return 0
    arr = np.arange(circ.n, dtype=np.int64)
    for inst in ops[k:]:
        a, b = int(inst.qubits[0]), int(inst.qubits[1])
        arr[a], arr[b] = arr[b], arr[a]
    n_tail = len(ops) - k
    del ops[k:]
    for logical, phys in final_layout.items():
        final_layout[logical] = int(arr[phys])
    circ.measures = [Inst("measure", (int(arr[m.qubits[0]]),),
                          cbits=m.cbits) for m in circ.measures]
    circ._deps = None
    return n_tail


def fold_swaps(circ: Circuit, final_layout: dict) -> Circuit:
    """单遍置换折叠：所有 SWAP 并入标签置换，线路重写为无 SWAP 形式。

    返回 circ（就地修改 ops/measures/final_layout）。语义恒等式
    （置换群共轭作用）：U = P · Π_j G_j^{P_j}，其中 G_j^{P_j} 为门
    经其左侧全部交换的乘积共轭后的重标记门；输出态 |ψ_out⟩ 在旧位置
    p 的内容 = 新线路位置 arr[p] 的内容（arr[q]=P(q)，swap (a,b) 折叠
    为 arr[a],arr[b] 互换）。
    """
    import numpy as np
    arr = np.arange(circ.n, dtype=np.int64)
    out_ops = []
    for inst in circ.ops:
        if inst.name == "swap":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            arr[a], arr[b] = arr[b], arr[a]
        else:
            out_ops.append(Inst(inst.name,
                                tuple(int(arr[q]) for q in inst.qubits),
                                inst.params))
    for logical, phys in final_layout.items():
        final_layout[logical] = int(arr[phys])
    circ.ops = out_ops
    circ.measures = [Inst("measure", (int(arr[m.qubits[0]]),),
                          cbits=m.cbits) for m in circ.measures]
    circ._deps = None
    return circ


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


def _is_h_block(ops, i):
    return (ops[i].name == "rz" and ops[i + 1].name == "sx"
            and ops[i + 2].name == "rz"
            and abs(ops[i].params[0] - H_GATE[0]) < 1e-9
            and abs(ops[i + 2].params[0] - H_GATE[0]) < 1e-9)


def _oneq_matrix(name: str, params: tuple):
    """rz/sx/x → 2×2 矩阵（平台约定：RZ(a)=diag(e^{−ia/2}, e^{ia/2})）。"""
    import numpy as np
    if name == "rz":
        h = params[0] / 2.0
        return np.diag([np.exp(-1j * h), np.exp(1j * h)])
    if name == "sx":
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
    if name == "x":
        return np.array([[0.0, 1.0], [1.0, 0.0]])
    raise ValueError(name)


def _mat_to_u3_params(U) -> tuple:
    """SU(2) 矩阵 → u3(θ,φ,λ) 参数（含全局相位剥离的精确提取）。

    一般 SU(2)：U = e^{iα}·u3(θ,φ,λ)，α 为 a 的辐角（u3 规范形
    a=cos(θ/2) 为实）；e^{iα} 是 1Q 门的全局相位、物理无关，剥离后
    θ = 2·atan2(|b|,|a|)，φ = arg(c)−α，λ = arg(−b)−α。
    """
    import math
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
    else:  # θ>π/2：经 d 求 α（a 幅角数值不稳）
        alpha = (math.atan2(c.imag, c.real) + math.atan2((-b).imag, (-b).real)
                 - math.atan2(d.imag, d.real))
    ph = (math.atan2(c.imag, c.real) - alpha) % (2 * math.pi)
    la = (math.atan2((-b).imag, (-b).real) - alpha) % (2 * math.pi)
    return (th, ph, la)


def merge_1q_runs(out_ops: list) -> int:
    """相邻同比特 1Q 门（rz/sx/x）游程合并为单一 u3 再合成（矩阵精确）。

    机制对标 qiskit Optimize1qGates：H·u1·H 等游程坍缩为一个 U3，
    ZXZ 再合成后 1Q 门数从 7+ 降到 ≤5。返回合并的游程数。
    """
    import numpy as np
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
                out.append(Inst("u3", (q,), _mat_to_u3_params(U)))
                merged += 1
            else:
                out.append(inst)
            i = j
        else:
            out.append(inst)
            i += 1
    if merged:
        out_ops[:] = out
    return merged


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
    # 1Q 优化不动点循环（对标 qiskit Optimize1qGates + 交换消除）：
    # 游程合并为 u3 → ZXZ 再合成 → rz 合并 → H 块奇偶消解，重复至稳定
    for _ in range(3):
        changed = False
        if merge_1q_runs(out.ops) > 0:
            # u3 再合成回平台基（H 特例直接展开）
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
        _cancel_adjacent_h(out.ops)
        changed = changed or len(out.ops) != before
        if not changed:
            break
    return out


# 兼容别名（早期调用方）
decompose_to_cz = decompose_to_platform
