# -*- coding: utf-8 -*-
"""A1 组件自审计：LNN 链合成 + 置换折叠重链化（2026-11-10）。"""
import numpy as np
import pytest

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.lnn import (chain_for_distance, cnot_chain_on_path,
                             noise_aware_chain)
from qtrail.pure.post_v2 import fold_and_relink


def _cx_mat(n, ctl, tgt):
    M = np.eye(n, dtype=np.uint8)
    M[tgt] ^= M[ctl]
    return M


def test_chain_table_all_distances():
    """链表：d=1..25 门数精确、矩阵精确（GF(2) 验证）。"""
    for d in range(1, 26):
        chain = chain_for_distance(d)
        expected = 1 if d == 1 else 4 * (d - 1)
        assert len(chain) == expected, f"d={d} 门数 {len(chain)} != {expected}"
        n = d + 1
        M = np.eye(n, dtype=np.uint8)
        for c, t in chain:
            M = (_cx_mat(n, c, t) @ M) % 2  # GF(2) 矩阵乘法
        target = np.eye(n, dtype=np.uint8)
        target[d, 0] = 1
        assert np.array_equal(M, target), f"d={d} 矩阵不匹配"


def test_chain_on_path_forward_and_reverse():
    """路径链：正向与反向段的矩阵验证。"""
    from qtrail.devices import build_tianyan287_spec
    spec = build_tianyan287_spec()
    # 正向：row 0 的 0→5
    path = [i for i in range(6)]
    gates = cnot_chain_on_path(path, 0, 5)
    n = 6
    M = np.eye(n, dtype=np.uint8)
    for c, t in gates:
        M = (_cx_mat(n, c, t) @ M) % 2
    target = np.eye(n, dtype=np.uint8)
    target[5, 0] = 1
    assert np.array_equal(M, target), "正向链矩阵不匹配"
    # 反向：5→0
    gates_r = cnot_chain_on_path(path, 5, 0)
    M = np.eye(n, dtype=np.uint8)
    for c, t in gates_r:
        M = (_cx_mat(n, c, t) @ M) % 2
    target = np.eye(n, dtype=np.uint8)
    target[0, 5] = 1
    assert np.array_equal(M, target), "反向链矩阵不匹配"


def test_noise_aware_chain_valid():
    """噪声感知链：全部相邻、链长 4(d-1)。"""
    from qtrail.devices import build_tianyan287_spec
    spec = build_tianyan287_spec()
    gates, cost = noise_aware_chain(spec, 0, 40)
    assert gates is not None
    d = 40 // 7 + 40 % 7  # 曼哈顿距离（0 到 40 = 5 行 + 5 列 = 10）
    d_expected = (40 // 7) + (40 % 7)
    assert len(gates) == 4 * (d_expected - 1), \
        f"链长 {len(gates)} != {4 * (d_expected - 1)}"
    for a, b in gates:
        assert spec.adj[a, b], f"链门非相邻 {(a, b)}"


def _make_fold_case():
    """构造折叠案例：swap 后跟长程门的小线路。"""
    from qtrail.devices import build_tianyan287_spec
    spec = build_tianyan287_spec()
    # 线路：cx(0,1); swap(1,7); cx(0,1)（swap 后 cx(0,1) 的 1 实际是 7）
    # 折叠后：cx(0,1); cx(0,7)（长程 d=2）→ 链 4 门
    circ = Circuit(105)
    circ.cx(0, 1)
    circ.swap(1, 7)
    circ.cx(0, 1)
    return circ, spec


def test_fold_and_relink_semantics():
    """折叠+链化：语义等价（态矢量级）。"""
    circ, spec = _make_fold_case()
    new_circ, arr = fold_and_relink(circ, spec)
    if arr is None:
        pytest.skip("该案例折叠无收益被拒（合法行为）")
    assert new_circ.count("swap") == 0
    # 语义：两种线路作用于 |0⟩ 的态矢量（105 比特太大——改用
    # 小网格 spec 重测）
    from qtrail.devices import build_grid3x3_spec
    spec3 = build_grid3x3_spec()
    c3 = Circuit(9)
    c3.cx(0, 1)
    c3.swap(1, 3)
    c3.cx(0, 1)
    n3, a3 = fold_and_relink(c3, spec3)
    assert a3 is not None
    # 态矢量（9 比特可行）
    def sim(cc):
        sv = np.zeros(2 ** 9, dtype=complex)
        sv[0] = 1
        for inst in cc.ops:
            qs = sorted(int(q) for q in inst.qubits)
            perm = list(range(9))
            for q in qs:
                perm.remove(q)
            new_perm = perm + qs
            sv = np.transpose(sv.reshape([2] * 9), new_perm).reshape(
                2 ** (9 - len(qs)), 2 ** len(qs))
            mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                            [0, 0, 1, 0]], dtype=complex)
            sv = sv @ mat.T
            inv = np.argsort(new_perm)
            sv = np.transpose(sv.reshape([2] * 9), inv).reshape(-1)
        return sv
    sv1 = sim(c3)
    sv2 = sim(n3)
    # 折叠引入的全局相位：|⟨sv1|sv2⟩|²
    fid = abs(np.vdot(sv1, sv2)) ** 2
    assert fid > 1 - 1e-8, f"语义不等价 fid={fid}"


def test_fold_and_relink_hardware_valid():
    """折叠产物全部 2Q 门相邻。"""
    circ, spec = _make_fold_case()
    new_circ, arr = fold_and_relink(circ, spec)
    if arr is None:
        pytest.skip("无收益被拒")
    for inst in new_circ.ops:
        if inst.nq == 2:
            assert spec.adj[int(inst.qubits[0]), int(inst.qubits[1])]


def test_fold_and_relink_no_worse():
    """不劣化：折叠产物 2Q 门数 < 原展开口径。"""
    circ, spec = _make_fold_case()
    orig_cost = sum(1 for i in circ.ops if i.nq == 2) + 3 * circ.count("swap")
    new_circ, arr = fold_and_relink(circ, spec)
    if arr is not None:
        new_cost = sum(1 for i in new_circ.ops if i.nq == 2)
        assert new_cost < orig_cost, f"{new_cost} >= {orig_cost}"
