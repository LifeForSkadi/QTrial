# -*- coding: utf-8 -*-
"""v5 后处理栈自审计测试（post_v2.py，2026-11-09）。

覆盖 B 组件（1Q 游程优化）的正确性与收益：
  1. 矩阵往返：随机 SU(2) → u3 参数 → ZXZ 合成 → 矩阵等价（模全局相位）
  2. 游程合并语义：随机 rz/sx/x 游程合并后矩阵等价、门数减少
  3. H 块奇偶消解：H²=I 精确性
  4. v4 vs v5 全管线 A/B：语义等价（态矢量）+ 1Q 门数减少 + 硬件合法
  5. 分解不动点：v5 输出再优化无变化
"""
import math
from pathlib import Path

import numpy as np
import pytest

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.post import _u3_to_rzsx, decompose_to_platform
from qtrail.pure.post_v2 import (_cancel_adjacent_h, _decompose_ablation,
                                 _mat_to_u3_params, _oneq_matrix,
                                 merge_1q_runs)
from qtrail.pure.qasm import load_qasm_file

ROOT = Path(__file__).resolve().parent.parent
H_GATE = (math.pi / 2, 0.0, math.pi)


def _up_to_phase(a, b, tol=1e-9):
    ov = np.vdot(a, b)
    return abs(ov) / (np.linalg.norm(a) * np.linalg.norm(b)) > 1 - tol


def _matrix_of_ops(ops):
    U = np.eye(2, dtype=complex)
    for inst in ops:
        U = _oneq_matrix(inst.name, inst.params) @ U
    return U


def _random_su2(rng):
    """Haar-ish 随机 SU(2)（经随机 u3 参数构造，覆盖全参数空间）。"""
    th = float(rng.uniform(0, math.pi))
    ph = float(rng.uniform(0, 2 * math.pi))
    la = float(rng.uniform(0, 2 * math.pi))
    c, s = math.cos(th / 2), math.sin(th / 2)
    return np.array([[c, -np.exp(1j * la) * s],
                     [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]],
                    dtype=complex)


# ------------------------------------------------------------ 矩阵往返
def test_mat_to_u3_roundtrip_random():
    """随机 SU(2) → 参数 → ZXZ 合成 → 矩阵等价（覆盖全部参数分支）。"""
    rng = np.random.default_rng(7)
    for _ in range(300):
        U = _random_su2(rng)
        th, ph, la = _mat_to_u3_params(U)
        ops = [Inst(g, (0,), p) for g, p in _u3_to_rzsx(th, ph, la)]
        assert _up_to_phase(_matrix_of_ops(ops), U), \
            f"roundtrip failed for u3({th},{ph},{la})"


def test_mat_to_u3_roundtrip_special_cases():
    """θ=π / θ=0 / θ=π/2 分支（数值陷阱所在）。"""
    specials = [
        np.array([[0, 1], [1, 0]], dtype=complex),                    # X
        np.array([[0, -1j], [1j, 0]], dtype=complex),                 # Y
        np.array([[1, 0], [0, 1]], dtype=complex),                    # I
        np.diag([1, 1j]),                                             # S
        np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),    # H
        np.array([[1, -1j], [-1j, 1]], dtype=complex) / math.sqrt(2) * 1j,
    ]
    for U in specials:
        U = U / np.sqrt(np.linalg.det(U))  # 规范化 SU(2)
        th, ph, la = _mat_to_u3_params(U)
        ops = [Inst(g, (0,), p) for g, p in _u3_to_rzsx(th, ph, la)]
        assert _up_to_phase(_matrix_of_ops(ops), U), f"special case {U}"


# ------------------------------------------------------------ 游程合并
def test_merge_1q_runs_semantics_and_reduction():
    """随机游程合并：语义等价 + 门数减少。"""
    rng = np.random.default_rng(11)
    for trial in range(50):
        ops = []
        for _ in range(int(rng.integers(2, 9))):
            name = rng.choice(["rz", "sx", "x"])
            p = (float(rng.uniform(0, 2 * math.pi)),) if name == "rz" else ()
            ops.append(Inst(name, (0,), p))
        U_before = _matrix_of_ops(ops)
        merged = merge_1q_runs(ops)
        if merged:
            if len(ops) == 0:
                # 恒等游程整体删除（x·x/sx⁴ 类）：矩阵应为恒等
                assert _up_to_phase(np.eye(2), U_before)
            else:
                assert len(ops) == 1 and ops[0].name == "u3"
                th, ph, la = ops[0].params
                ops_synth = [Inst(g, (0,), p)
                             for g, p in _u3_to_rzsx(th, ph, la)]
                assert _up_to_phase(_matrix_of_ops(ops_synth), U_before)


def test_merge_1q_runs_single_gate_untouched():
    """单门游程不合并（无收益时不重写）。"""
    ops = [Inst("rz", (0,), (0.5,))]
    assert merge_1q_runs(ops) == 0
    assert len(ops) == 1 and ops[0].name == "rz"


# ------------------------------------------------------------ A3：CX 通勤对消
def test_commute_cancel_cx_removes_pairs():
    """窗口内 cx(a,b)···cx(a,b) 对消（中间门全对易时），反之不动。"""
    from qtrail.pure.post_v2 import commute_cancel_cx
    # 中间 cx(0,2) 同控对易 → 成对消除
    ops = [Inst("cx", (0, 1)), Inst("cx", (0, 2)), Inst("cx", (0, 1))]
    assert commute_cancel_cx(ops) == 2
    assert len(ops) == 1 and ops[0].qubits == (0, 2)
    # 交叉控制 cx(1,0) 不对易 → 不动
    ops = [Inst("cx", (0, 1)), Inst("cx", (1, 0)), Inst("cx", (0, 1))]
    assert commute_cancel_cx(ops) == 0 and len(ops) == 3
    # 共享比特的 1Q 门不对易 → 不动
    ops = [Inst("cx", (0, 1)), Inst("rz", (0,), (0.3,)), Inst("cx", (0, 1))]
    assert commute_cancel_cx(ops) == 0 and len(ops) == 3
    # 不相交比特的 1Q 门对易 → 消除
    ops = [Inst("cx", (0, 1)), Inst("rz", (2,), (0.3,)), Inst("cx", (0, 1))]
    assert commute_cancel_cx(ops) == 2 and len(ops) == 1
    # 窗口外不消除（window 上限；间隔门不对易则不消除）
    ops = [Inst("cx", (0, 1))] + [Inst("rz", (0,), (0.3,))] * 13 \
        + [Inst("cx", (0, 1))]
    assert commute_cancel_cx(ops) == 0 and len(ops) == 15
    # 窗口外但间隔门对易 → 仍不消除（距离超 window）
    ops = [Inst("cx", (0, 1))] + [Inst("rz", (3,), (0.3,))] * 13 \
        + [Inst("cx", (0, 1))]
    assert commute_cancel_cx(ops) == 0 and len(ops) == 15


def test_commute_cancel_cx_semantics():
    """随机窗口电路：消解前后态矢量等价（多随机输入态）。"""
    from qtrail.pure.post_v2 import commute_cancel_cx
    rng = np.random.default_rng(23)
    for trial in range(30):
        n = int(rng.integers(2, 5))
        ops = []
        for _ in range(int(rng.integers(2, 12))):
            g = rng.choice(["cx", "cx", "rz", "sx"])
            if g == "cx":
                a = int(rng.integers(0, n))
                b = int(rng.integers(0, n))
                if a == b:
                    b = (b + 1) % n
                ops.append(Inst("cx", (a, b)))
            else:
                q = int(rng.integers(0, n))
                p = (float(rng.uniform(0, 2 * math.pi)),) if g == "rz" else ()
                ops.append(Inst(g, (q,), p))
        c0 = Circuit(n)
        c0.ops = list(ops)
        c1 = Circuit(n)
        c1.ops = list(ops)
        commute_cancel_cx(c1.ops)
        sv0 = _random_product_state(rng, n)
        assert np.allclose(_sim(c0, n, sv0), _sim(c1, n, sv0), atol=1e-10), \
            f"trial{trial} 消解前后态矢量不等价"


# ------------------------------------------------------------ H 块消解
def test_cancel_adjacent_h_even_odd():
    """偶数个相邻 H 块 = 恒等（删除）；奇数个 = 一个 H 块。"""
    def h_block(q):
        return [Inst("rz", (q,), (H_GATE[0],)), Inst("sx", (q,)),
                Inst("rz", (q,), (H_GATE[0],))]

    # 偶数（4 个）：全部删除
    ops = h_block(0) + h_block(0) + h_block(0) + h_block(0)
    _cancel_adjacent_h(ops)
    assert len(ops) == 0
    # 奇数（3 个）：保留一个
    ops = h_block(0) + h_block(0) + h_block(0)
    _cancel_adjacent_h(ops)
    assert len(ops) == 3
    # 不同比特不相邻：不消解
    ops = h_block(0) + [Inst("cz", (0, 1))] + h_block(1)
    _cancel_adjacent_h(ops)
    assert len(ops) == 7


# ------------------------------------------------------------ v4 vs v5 A/B
def _run_pipeline(case, pipeline):
    from qtrail.config import load_device_config
    from qtrail.devices import build_tianyan287_spec
    from qtrail.pure.mapper import PureMapper
    from qtrail.pure.post import post_route
    from qtrail.pure.router import sabre_route
    spec = build_tianyan287_spec(load_device_config())
    circ = load_qasm_file(ROOT / case)
    routed, swaps, fin = sabre_route(circ, spec, {i: i for i in range(circ.n)},
                                     seed=42)
    post_route(routed, fin, spec.adj)
    if pipeline == "v5":
        final_qc = _decompose_ablation(routed)
    else:
        final_qc = decompose_to_platform(routed)
    return final_qc, spec, fin


def _unmap(circ, fin, n_logical):
    """物理空间线路逆映射回逻辑空间（丢弃未使用物理比特上的门）。"""
    inv = {int(v): int(k) for k, v in fin.items() if int(k) < n_logical}
    out = Circuit(n_logical)
    for inst in circ.ops:
        qs = [inv[int(q)] for q in inst.qubits if int(q) in inv]
        if len(qs) == inst.nq:
            out.append(Inst(inst.name, tuple(qs), inst.params))
    return out


def _apply(sv, mat, qs, n):
    """在 n 比特态矢量上作用 qs 上的小矩阵（与 test_pure.py 同款约定）。"""
    qs = tuple(sorted(qs))
    perm = list(range(n))
    for q in qs:
        perm.remove(q)
    new_perm = perm + list(qs)
    sv = np.transpose(sv.reshape([2] * n), new_perm).reshape(
        2 ** (n - len(qs)), 2 ** len(qs))
    sv = sv @ mat.T
    inv = np.argsort(new_perm)
    return np.transpose(sv.reshape([2] * n), inv).reshape(-1)


def _gate_mat(name, params):
    if name == "rz":
        return np.diag([np.exp(-1j * params[0] / 2),
                        np.exp(1j * params[0] / 2)])
    if name == "sx":
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
    if name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "cz":
        m = np.eye(4, dtype=complex)
        m[3, 3] = -1
        return m
    if name == "cx":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
    if name == "swap":
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
    if name == "u1":
        return np.diag([1.0, np.exp(1j * params[0])])
    if name == "u3":
        th, ph, la = params
        c, s = np.cos(th / 2), np.sin(th / 2)
        return np.array([[c, -np.exp(1j * la) * s],
                         [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]],
                        dtype=complex)
    raise ValueError(name)


def _sim(circ, n, sv=None):
    """纯 numpy 态矢量演化（qiskit 2.5 Statevector 有维度上限，不用它）。"""
    if sv is None:
        sv = np.zeros(2 ** n, dtype=complex)
        sv[0] = 1
    for inst in circ.ops:
        sv = _apply(sv, _gate_mat(inst.name, inst.params), inst.qubits, n)
    return sv


def _random_product_state(rng, n):
    sv = np.ones(1, dtype=complex)
    for _ in range(n):
        th = rng.uniform(0, np.pi)
        ph = rng.uniform(0, 2 * np.pi)
        sv = np.kron(sv, np.array([np.cos(th / 2),
                                   np.sin(th / 2) * np.exp(1j * ph)]))
    return sv


@pytest.mark.parametrize("case", ["examples/qft5.qasm", "examples/ghz10.qasm",
                                  "examples/qaoa12.qasm"])
def test_v5_semantics_equivalent_to_v4(case):
    """v5 分解与 v4 分解语义等价（逆映射回逻辑空间后 |0⟩ 与随机直积
    输入态模拟，全局相位不敏感）。"""
    qc_v4, spec, fin = _run_pipeline(case, "v4")
    qc_v5, _, _ = _run_pipeline(case, "v5")
    n = load_qasm_file(ROOT / case).n
    u4 = _unmap(qc_v4, fin, n)
    u5 = _unmap(qc_v5, fin, n)
    rng = np.random.default_rng(3)
    for trial in range(3):
        sv0 = _random_product_state(rng, n) if trial else None
        sv4 = _sim(u4, n, sv0)
        sv5 = _sim(u5, n, sv0)
        fid = abs(np.vdot(sv4, sv5)) ** 2
        assert fid > 1 - 1e-8, \
            f"{case} trial{trial} v4/v5 语义不等价 fid={fid}"


@pytest.mark.parametrize("case", ["examples/qft5.qasm", "examples/ghz10.qasm",
                                  "examples/qaoa12.qasm"])
def test_v5_reduces_1q_gate_count(case):
    """v5 的 1Q 门数 ≤ v4（B 组件的收益断言）。"""
    qc_v4, _, _ = _run_pipeline(case, "v4")
    qc_v5, _, _ = _run_pipeline(case, "v5")
    n1q = lambda qc: sum(1 for i in qc.ops if i.name in ("rz", "sx", "x"))
    assert n1q(qc_v5) < n1q(qc_v4), \
        f"{case}: v5 1Q {n1q(qc_v5)} 未少于 v4 {n1q(qc_v4)}"
    # B6（cz 对消）后 2Q 门数可减少（语义等价测试保证正确性）
    assert sum(1 for i in qc_v5.ops if i.name == "cz") <= \
        sum(1 for i in qc_v4.ops if i.name == "cz"), "2Q 门数不应增加"


def test_v5_hardware_valid():
    """v5 产物全部 2Q 门相邻（硬件合法性）。"""
    from qtrail.config import load_device_config
    from qtrail.devices import build_tianyan287_spec
    spec = build_tianyan287_spec(load_device_config())
    qc_v5, _, _ = _run_pipeline("examples/qft5.qasm", "v5")
    for inst in qc_v5.ops:
        if inst.name == "cz":
            assert spec.adj[int(inst.qubits[0]), int(inst.qubits[1])]


def test_v5_decompose_idempotent():
    """v5 输出再跑 v5 优化不动点（不再变化）。"""
    qc_v5, _, _ = _run_pipeline("examples/qft5.qasm", "v5")
    ops_before = list(qc_v5.ops)
    for _ in range(3):
        changed = False
        if merge_1q_runs(qc_v5.ops) > 0:
            new_ops = []
            for inst in qc_v5.ops:
                if inst.name == "u3":
                    for g, p in _u3_to_rzsx(*inst.params):
                        new_ops.append(Inst(g, inst.qubits, p))
                else:
                    new_ops.append(inst)
            qc_v5.ops = new_ops
            changed = True
        from qtrail.pure.post import merge_adjacent_rz
        merge_adjacent_rz(qc_v5)
        before = len(qc_v5.ops)
        _cancel_adjacent_h(qc_v5.ops)
        changed = changed or len(qc_v5.ops) != before
        if not changed:
            break
    assert len(qc_v5.ops) == len(ops_before), "v5 输出非不动点"


# ------------------------------------------------------------ C1：重标记
def test_noise_aware_relabel_semantics_and_validity():
    """C1 重标记：语义等价（全局标签置换）+ 硬件合法 + 保真度不劣化。"""
    from qtrail.pure.post_v2 import noise_aware_relabel
    from qtrail.pure.metrics import estimate_fidelity
    from qtrail.devices import build_tianyan287_real_spec
    from qtrail.pure.post import post_route
    from qtrail.pure.router import sabre_route as _sabre
    spec3 = build_tianyan287_real_spec()  # 真机校准（方差大，收益可见）
    for case in ["examples/qft5.qasm", "examples/ghz10.qasm",
                 "examples/qaoa12.qasm"]:
        circ = load_qasm_file(ROOT / case)
        routed, swaps, fin = _sabre(circ, spec3,
                                    {i: i for i in range(circ.n)},
                                    seed=42)
        post_route(routed, fin, spec3.adj)
        fid_before = estimate_fidelity(routed, spec3.calib)
        # 逻辑空间快照（unmap）
        u_before = _unmap(routed, fin, circ.n)
        sv_before = _sim(u_before, circ.n, None)
        n_moved = noise_aware_relabel(routed, spec3, fin)
        # 语义等价：unmap 后态矢量一致
        u_after = _unmap(routed, fin, circ.n)
        sv_after = _sim(u_after, circ.n, None)
        fid_sem = abs(np.vdot(sv_before, sv_after)) ** 2
        assert fid_sem > 1 - 1e-8, \
            f"{case} 重标记破坏语义 fid={fid_sem}（移动 {n_moved} 次）"
        # 硬件合法
        for inst in routed.ops:
            if inst.nq == 2:
                assert spec3.adj[int(inst.qubits[0]), int(inst.qubits[1])], \
                    f"{case} 重标记产生非法 2Q 门 {inst.qubits}"
        # 保真度不劣化（严格 ≥）
        fid_after = estimate_fidelity(routed, spec3.calib)
        assert fid_after >= fid_before - 1e-12, \
            f"{case} 重标记劣化 {fid_before} -> {fid_after}"


# ------------------------------------------------------------ 全局重标记语义
def test_global_relabel_semantics():
    """嵌入路径（0-SWAP 消除 / 部分嵌入）：态矢量等价（含输出置换
    追踪）+ 链化后全部 2Q 门相邻。

    使用 3×3 小设备（全部位置承载逻辑比特）：链路径不离开逻辑空间，
    逆映射完备，可做逐门态矢量验证（105 比特设备上链穿过非逻辑
    中间位，无法用同一逆映射验证——该情形由链矩阵单元测试 +
    代数恒等式组合保证）。
    """
    from qtrail.devices import build_grid3x3_spec
    from qtrail.pure.relabel_vf2 import global_relabel
    from qtrail.pure.post import post_route
    from qtrail.pure.post_v2 import chain_long_gates, fold_only
    from qtrail.pure.router import sabre_route
    spec = build_grid3x3_spec()
    rng = np.random.default_rng(9)
    n = 9  # 3×3 全设备

    def _gate_mat(name, qs):
        """n 比特空间上的门矩阵（cx：控制在前目标在后，按 qs 原始顺序）。"""
        U = np.eye(2 ** n, dtype=complex)
        c, t = int(qs[0]), int(qs[1])
        for x in range(2 ** n):
            if (x >> c) & 1:
                y = x ^ (1 << t)
                U[x, x] = 0.0
                U[y, x] = 1.0
        return U

    def _circuit_mat(ops):
        U = np.eye(2 ** n, dtype=complex)
        for name, qs in ops:
            U = _gate_mat(name, qs) @ U
        return U

    def _perm_mat(M):
        P = np.zeros((2 ** n, 2 ** n), dtype=complex)
        for x in range(2 ** n):
            y = 0
            for k in range(n):
                if (x >> k) & 1:
                    y |= 1 << M[k]
            P[y, x] = 1.0
        return P

    tested = 0
    for trial in range(8):
        circ = Circuit(n)
        for _ in range(int(rng.integers(5, 11))):
            a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
            if a != b:
                circ.cx(a, b)
        routed, _, fin = sabre_route(circ, spec, {i: i for i in range(n)},
                                     seed=42)
        post_route(routed, fin, spec.adj)
        folded_only, arr = fold_only(routed)
        # 折叠 = 原线路（坐标跟踪恒等式；门序可因独立门换序而不同，
        # 矩阵不变）
        assert np.allclose(
            _circuit_mat([(i.name, tuple(int(q) for q in i.qubits))
                          for i in circ.ops]),
            _circuit_mat([(i.name, tuple(int(q) for q in i.qubits))
                          for i in folded_only.ops]), atol=1e-10), \
            f"trial{trial} fold_only 语义错误"
        res = global_relabel(folded_only, spec)
        if res is None:
            continue
        tested += 1
        relabeled, mapping, _exact = res
        newc, _ = chain_long_gates(relabeled, spec)
        # 1. 链化后全部 2Q 门相邻 + 不落禁用比特
        disabled = set(spec.disabled_qubits)
        for inst in newc.ops:
            if inst.nq == 2 and inst.name != "swap":
                assert spec.adj[int(inst.qubits[0]), int(inst.qubits[1])], \
                    f"trial{trial} 非相邻门 {(inst.name, inst.qubits)}"
                assert int(inst.qubits[0]) not in disabled
                assert int(inst.qubits[1]) not in disabled
        # 2. 语义：共轭恒等式 U_new = Π_M U_orig Π_M^{-1}（显式矩阵，
        #    链 = cx 精确等价故不影响矩阵）
        P = _perm_mat([mapping.get(k, k) for k in range(n)])
        U_orig = _circuit_mat([(i.name, tuple(int(q) for q in i.qubits))
                               for i in circ.ops])
        U_new = _circuit_mat([(i.name, tuple(int(q) for q in i.qubits))
                              for i in newc.ops])
        assert np.allclose(P @ U_orig @ P.T, U_new, atol=1e-10), \
            f"trial{trial} 重标记后语义不等价"
    assert tested >= 3, "应有多条线路触发嵌入路径"
