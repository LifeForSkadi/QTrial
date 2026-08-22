"""pure 包测试：解析器、路由器正确性（态矢量等价）、度量。"""
from __future__ import annotations

import numpy as np
import pytest

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route
from qtrail.pure.metrics import circuit_depth, twoq_depth

PAULI = {"x": np.array([[0, 1], [1, 0]], dtype=complex),
         "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
         "z": np.array([[1, 0], [0, -1]], dtype=complex)}
I2 = np.eye(2, dtype=complex)
CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
              dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=complex)


def u3_mat(th, ph, la):
    c, s = np.cos(th / 2), np.sin(th / 2)
    return np.array([[c, -np.exp(1j * la) * s],
                     [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]],
                    dtype=complex)


def u1_mat(la):
    return np.diag([1, np.exp(1j * la)])


def _gate_mat(name, params):
    if name == "cx":
        return CX
    if name == "swap":
        return SWAP
    if name == "u3":
        return u3_mat(*params)
    if name == "u1":
        return u1_mat(params[0])
    if name == "cz":
        m = np.eye(4, dtype=complex)
        m[3, 3] = -1
        return m
    if name == "rz":
        return u1_mat(params[0])
    if name == "sx":
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
    if name in PAULI:
        return PAULI[name]
    if name == "id":
        return I2
    raise ValueError(name)


def _apply(sv, mat, qs, n):
    """在 n 比特态矢量上作用 qs 上的小矩阵（小端索引约定）。"""
    qs = tuple(sorted(qs))
    perm = list(range(n))
    for q in qs:
        perm.remove(q)
    new_perm = perm + list(qs)          # 目标比特移到高索引位
    sv = np.transpose(sv.reshape([2] * n), new_perm).reshape(
        2 ** (n - len(qs)), 2 ** len(qs))
    sv = sv @ mat.T
    inv = np.argsort(new_perm)
    return np.transpose(sv.reshape([2] * n), inv).reshape(-1)


def sim(circ: Circuit, n: int) -> np.ndarray:
    sv = np.zeros(2 ** n, dtype=complex)
    sv[0] = 1
    for inst in circ.ops:
        if inst.name in ("barrier", "id"):
            continue
        sv = _apply(sv, _gate_mat(inst.name, inst.params), inst.qubits, n)
    return sv


def make_spec3x3():
    from qtrail.devices import build_grid3x3_spec
    return build_grid3x3_spec()


def test_parser_counts():
    text = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
cx q[0],q[1];
u3(0.5,0.1,0.2) q[2];
u1(pi/4) q[3];
measure q[0] -> c[0];
"""
    circ = parse_qasm(text)
    assert circ.n == 4
    assert circ.count_2q() == 1
    assert sum(1 for i in circ.ops if i.nq == 1) == 3  # h→u3, u3, u1
    assert len(circ.measures) == 1


def test_parser_custom_gate():
    text = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
gate mycx a,b { cx a,b; }
mycx q[0],q[1];
"""
    circ = parse_qasm(text)
    assert circ.count_2q() == 1


def test_router_statevector_equivalence():
    spec = make_spec3x3()
    # 3 比特线路：h + 两个 cx + rz
    circ = Circuit(3)
    circ.append(Inst("u3", (0,), (np.pi / 2, 0.0, np.pi)))   # h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.append(Inst("u1", (2,), (0.37,)))
    layout = {0: 0, 1: 4, 2: 8}  # 3×3 对角（0-4、4-8 均不相邻，需交换）
    routed, swaps, final = sabre_route(circ, spec, layout, seed=0)
    assert swaps > 0
    # 原线路按初始布局嵌入 9 比特物理空间
    n = spec.n
    sv_a = np.zeros(2 ** n, dtype=complex)
    sv_a[0] = 1
    for inst in circ.ops:
        phys = tuple(layout[q] for q in inst.qubits)
        sv_a = _apply(sv_a, _gate_mat(inst.name, inst.params), phys, n)
    sv_b = sim(routed, n)
    # 最终位置置换：逻辑 q 的内容从 final[q] 轴搬到 layout[q] 轴；
    # 幻影轴必须为 |0⟩（自由轴任意对应）
    used_to = set(layout.values())
    free_to = [p for p in range(n) if p not in used_to]
    from_to = {}
    for logical, phys in final.items():
        from_to[phys] = layout[logical]
    free_from = [p for p in range(n) if p not in from_to]
    for f, t in zip(free_from, free_to):
        from_to[f] = t
    axes = [0] * n
    for f, t in from_to.items():
        axes[t] = f
    sv_b = np.transpose(sv_b.reshape([2] * n), axes).reshape(-1)
    fid = abs(np.vdot(sv_a, sv_b)) ** 2
    assert fid > 0.9999, f"fidelity {fid}"


def test_router_zero_swap_chain():
    spec = make_spec3x3()
    circ = Circuit(3)
    circ.append(Inst("u3", (0,), (np.pi / 2, 0.0, np.pi)))
    circ.cx(0, 1)
    circ.cx(1, 2)
    # 0-1-2 在 3×3 网格相邻（0,1 同行相邻；1,2 同行相邻）
    routed, swaps, final = sabre_route(circ, spec, {0: 0, 1: 1, 2: 2}, seed=0)
    assert swaps == 0
    assert final == {0: 0, 1: 1, 2: 2}


def test_metrics():
    circ = Circuit(4)
    circ.cz(0, 1)
    circ.cz(2, 3)
    circ.cz(0, 2)
    circ.cz(1, 3)
    assert circuit_depth(circ) == 2
    assert twoq_depth(circ) == 2
    assert circ.count_2q() == 4


def test_post_stack_equivalence():
    """后处理栈（推挤/消解/吸收/分解）保持语义等价。"""
    from qtrail.pure.post import post_route, decompose_to_cz
    spec = make_spec3x3()
    circ = Circuit(3)
    circ.append(Inst("u3", (0,), (np.pi / 2, 0.0, np.pi)))
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.append(Inst("u1", (2,), (0.37,)))
    layout = {0: 0, 1: 4, 2: 8}
    routed, swaps, final = sabre_route(circ, spec, layout, seed=0)

    # 后处理前参照态（含交换）
    sv_ref = sim(routed, spec.n)
    # 后处理
    final_copy = dict(final)
    post_circ, removed = post_route(routed, final_copy)
    # 吸收后：sv 应等于参照态按新最终布局重标记
    n = spec.n
    used_to = set(layout.values())
    free_to = [p for p in range(n) if p not in used_to]
    from_to = {}
    for logical, phys in final_copy.items():
        from_to[phys] = layout[logical]
    free_from = [p for p in range(n) if p not in from_to]
    for f, t in zip(free_from, free_to):
        from_to[f] = t
    axes = [0] * n
    for f, t in from_to.items():
        axes[t] = f
    sv_post = np.transpose(sim(post_circ, n).reshape([2] * n), axes).reshape(-1)
    # 参照态也按旧 final 重标记
    from_to2 = {}
    for logical, phys in final.items():
        from_to2[phys] = layout[logical]
    free_from2 = [p for p in range(n) if p not in from_to2]
    for f, t in zip(free_from2, free_to):
        from_to2[f] = t
    axes2 = [0] * n
    for f, t in from_to2.items():
        axes2[t] = f
    sv_ref2 = np.transpose(sv_ref.reshape([2] * n), axes2).reshape(-1)
    assert abs(np.vdot(sv_ref2, sv_post)) ** 2 > 0.9999

    # CZ 分解等价：最终线路按 final_copy 重标记后 = 原逻辑态（初始布局）
    sv_a = np.zeros(2 ** n, dtype=complex)
    sv_a[0] = 1
    for inst in circ.ops:
        phys = tuple(layout[q] for q in inst.qubits)
        sv_a = _apply(sv_a, _gate_mat(inst.name, inst.params), phys, n)
    dec = decompose_to_cz(post_circ)
    sv_dec = sim(dec, n)
    used_to = set(layout.values())
    free_to = [p for p in range(n) if p not in used_to]
    from_to = {}
    for logical, phys in final_copy.items():
        from_to[phys] = layout[logical]
    free_from = [p for p in range(n) if p not in from_to]
    for f, t in zip(free_from, free_to):
        from_to[f] = t
    axes = [0] * n
    for f, t in from_to.items():
        axes[t] = f
    sv_dec = np.transpose(sv_dec.reshape([2] * n), axes).reshape(-1)
    assert abs(np.vdot(sv_a, sv_dec)) ** 2 > 0.9999


def test_swap_past_touching_gates():
    """A 组规则：swap 穿过触碰门（重标记/控制靶翻转）保持语义等价。"""
    from qtrail.pure.post import push_swaps, _swap_past
    from qtrail.pure.circuit import Circuit as C2
    n = 4
    # [cx(0,1); swap(0,1); cx(1,2); rz(1); swap(0,1)]
    circ = C2(n)
    circ.cx(0, 1)
    circ.swap(0, 1)
    circ.cx(1, 2)
    circ.append(Inst("u1", (1,), (0.5,)))
    circ.swap(0, 1)
    sv_before = sim(circ, n)
    push_swaps(circ)
    sv_after = sim(circ, n)
    assert abs(np.vdot(sv_before, sv_after)) ** 2 > 0.9999
    # 该例中两个 swap 互为逆，推挤后相遇完全抵消（规则生效的直接证据）
    assert circ.count("swap") == 0


def test_post_route_full_pipeline_swap_reduction():
    """端到端：路由 + 完整后处理（含 A 组规则）后 swap 数不增、语义等价。"""
    from qtrail.pure.post import post_route
    spec = make_spec3x3()
    circ = Circuit(3)
    circ.append(Inst("u3", (0,), (np.pi / 2, 0.0, np.pi)))
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.append(Inst("u1", (2,), (0.37,)))
    routed, swaps, final = sabre_route(circ, spec, {0: 0, 1: 4, 2: 8}, seed=0)
    post_circ, removed = post_route(routed, final)
    assert post_circ.count("swap") <= swaps
    # 语义：最终线路按 final 重标记后 = 原逻辑态（初始布局）
    n = spec.n
    layout0 = {0: 0, 1: 4, 2: 8}
    sv_a = np.zeros(2 ** n, dtype=complex)
    sv_a[0] = 1
    for inst in circ.ops:
        phys = tuple(layout0[q] for q in inst.qubits)
        sv_a = _apply(sv_a, _gate_mat(inst.name, inst.params), phys, n)
    sv_post = sim(post_circ, n)
    used_to = set(layout0.values())
    free_to = [p for p in range(n) if p not in used_to]
    from_to = {}
    for logical, phys in final.items():
        from_to[phys] = layout0[logical]
    free_from = [p for p in range(n) if p not in from_to]
    for f, t in zip(free_from, free_to):
        from_to[f] = t
    axes = [0] * n
    for f, t in from_to.items():
        axes[t] = f
    sv_post = np.transpose(sv_post.reshape([2] * n), axes).reshape(-1)
    assert abs(np.vdot(sv_a, sv_post)) ** 2 > 0.9999


def test_ccx_decomposition_equivalence():
    """ccx（Toffoli）分解语义等价（相对相位可差，最终态 |1⟩ 分量相同）。"""
    text = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
x q[0];
x q[1];
ccx q[0],q[1],q[2];
p(0.3) q[2];
"""
    circ = parse_qasm(text)
    assert circ.count("cx") == 6  # Toffoli 分解含 6 个 cx
    sv = sim(circ, 3)
    # ccx 后 q2 = |1⟩；p(0.3) 给相位
    idx1 = int("111", 2)
    assert abs(abs(sv[idx1]) - 1.0) < 1e-9


def test_parser_p_gate():
    text = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
p(0.5) q[0];
"""
    circ = parse_qasm(text)
    assert circ.count("u1") == 1


def test_parser_cp_gate():
    """cp(λ) 分解语义等价：|11⟩ 获相位 e^{iλ}，其余基态不变。"""
    lam = 0.73
    text = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
x q[0];
x q[1];
cp({lam}) q[0],q[1];
"""
    circ = parse_qasm(text)
    assert circ.count("cx") == 2
    sv = sim(circ, 2)
    target = np.exp(1j * lam)
    assert abs(sv[3] - target) < 1e-9
    for k in (0, 1, 2):
        assert abs(sv[k]) < 1e-9


def test_post_hardware_validity():
    """后处理后的每条 2Q 门必须落在耦合图上（真机可执行性）。

    2026-08-22 教训：语义等价≠硬件可执行——无约束推挤/全局折叠会把
    2Q 门重标记到非相邻比特对（天衍平台 qcis_check_regular 实测拒绝）。
    本测试保证：推挤+尾块吸收后线路中所有 2Q 门的比特对都在 spec.adj 内。
    """
    from qtrail.pure.post import post_route
    from qtrail.pure.router import sabre_route
    spec = make_spec3x3()
    # 需要路由的线路（对角布局 + 环结构交互）
    circ = Circuit(3)
    circ.append(Inst("u3", (0,), (np.pi / 2, 0.0, np.pi)))
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.cx(2, 0)
    circ.append(Inst("u1", (2,), (0.37,)))
    circ.swap(0, 1)
    circ.cx(0, 2)
    routed, _, fl = sabre_route(circ, spec, {0: 0, 1: 4, 2: 8}, seed=0)
    post_route(routed, dict(fl), spec=spec)
    bad = []
    for inst in routed.ops:
        if inst.nq == 2:
            a, b = inst.qubits
            if not spec.adj[a, b]:
                bad.append((inst.name, a, b))
    assert not bad, f"hardware-invalid 2Q gates: {bad[:5]}"
    # 语义仍等价（按 final_layout 重标记回逻辑空间）
    n = spec.n
    layout0 = {0: 0, 1: 4, 2: 8}
    sv_a = np.zeros(2 ** n, dtype=complex)
    sv_a[0] = 1
    for inst in circ.ops:
        phys = tuple(layout0[q] for q in inst.qubits)
        sv_a = _apply(sv_a, _gate_mat(inst.name, inst.params), phys, n)
    sv_post = sim(routed, n)
    used_to = set(layout0.values())
    free_to = [p for p in range(n) if p not in used_to]
    from_to = {}
    for logical, phys in fl.items():
        from_to[phys] = layout0[logical]
    free_from = [p for p in range(n) if p not in from_to]
    for f, t in zip(free_from, free_to):
        from_to[f] = t
    axes = [0] * n
    for f, t in from_to.items():
        axes[t] = f
    sv_post = np.transpose(sv_post.reshape([2] * n), axes).reshape(-1)
    assert abs(np.vdot(sv_a, sv_post)) ** 2 > 0.9999
