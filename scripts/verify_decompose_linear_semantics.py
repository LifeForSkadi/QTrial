# -*- coding: utf-8 -*-
"""decompose_linear（线性分解，补充四十预注册）语义等价验证。

三层门禁：
1. 矩阵级：_merge_runs_u3cz 的 run 合并产物 = 原 run 矩阵积
   （模全局相位，|tr(U_out† U_in)|/2 ≈ 1）——含 H 块吸收、
   恒等 run 删除、收益检查不膨胀
2. 线路级：decompose_linear vs v4 decompose_to_platform（可信参照）
   态矢量重叠 = 1——输入含 h/x/cx/u3/u1/swap（补充三十教训：
   语义验证必须含非平凡门）+ |0⟩ 与随机直积态与纠缠态输入
3. 平台合法性：输出门集 ⊆ {rz,sx,x,cz}、无 swap/cx/u3/u1/h 残留
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.post import decompose_to_platform
from qtrail.pure.post_v2 import (_cancel_cz_global, _expand_to_u3cz,
                                 _mar_from_ops, _merge_runs_cross,
                                 _merge_runs_u3cz, _oneq_matrix,
                                 _strip_identity_1q, _translate_v4,
                                 _u3_level_to_platform, decompose_linear)

_PLATFORM_1Q = {"rz", "sx", "x"}
_PLATFORM = _PLATFORM_1Q | {"cz"}


def statevec(ops, n, init=None):
    """ops → 态矢量（init=None 用 |0⟩，否则用给定矢量演化）。"""
    qc = QuantumCircuit(n)
    for inst in ops:
        name, q = inst.name, inst.qubits
        if name == "u3":
            qc.u(*inst.params, int(q[0]))
        elif name == "u1":
            qc.p(inst.params[0], int(q[0]))
        elif name == "h":
            qc.h(int(q[0]))
        elif name == "x":
            qc.x(int(q[0]))
        elif name == "cx":
            qc.cx(int(q[0]), int(q[1]))
        elif name == "swap":
            qc.swap(int(q[0]), int(q[1]))
        elif name == "rz":
            qc.rz(inst.params[0], int(q[0]))
        elif name == "sx":
            qc.sx(int(q[0]))
        elif name == "cz":
            qc.cz(int(q[0]), int(q[1]))
        else:
            raise ValueError(f"未支持门 {name}")
    sv = Statevector(qc) if init is None else Statevector(init).evolve(qc)
    return sv.data


def overlap(sv1, sv2):
    return abs(np.vdot(sv1, sv2))


def gate_set_ok(ops):
    bad = [f"{i.name}{i.qubits}" for i in ops if i.name not in _PLATFORM]
    return not bad, bad


def test_matrix_merge():
    """矩阵级：run 合并产物 = 原乘积（模全局相位）。"""
    from qtrail.pure.post_v2 import _U3_H
    rng = np.random.default_rng(7)
    n_fail = 0
    _1Q = ("u3", "u1", "rz", "sx", "x")

    def rand_1q(q):
        g = rng.integers(0, 6)
        if g == 0:
            return Inst("u3", (q,), tuple(rng.uniform(0, 2 * np.pi, 3)))
        if g == 1:
            return Inst("u1", (q,), (float(rng.uniform(0, 2 * np.pi)),))
        if g == 2:
            return Inst("rz", (q,), (float(rng.uniform(0, 2 * np.pi)),))
        if g == 3:
            return Inst("sx", (q,))
        if g == 4:
            return Inst("x", (q,))
        return Inst("u3", (q,), _U3_H)  # H

    def run_product(ops):
        U = np.eye(2, dtype=complex)
        for i in ops:
            U = _oneq_matrix(i.name, i.params) @ U
        return U

    def phase_overlap(U, V):
        """2×2 矩阵模全局相位的重叠：|tr(U†V)|/2。"""
        return abs(np.trace(U.conj().T @ V)) / 2.0

    for t in range(30):
        ops = [rand_1q(0) for _ in range(rng.integers(1, 6))]
        U_in = run_product(ops)
        ops2 = list(ops)
        _merge_runs_u3cz(ops2)
        if not ops2:  # 恒等 run 删除：U_in 应为恒等（模相位）
            ok = abs(abs(np.trace(U_in)) / 2 - 1) < 1e-8
        else:
            U_out = run_product(ops2)
            ok = phase_overlap(U_out, U_in) > 1 - 1e-8
        if not ok:
            print(f"[FAIL] 矩阵级 {t}: 输入 {len(ops)} 门 → 输出 "
                  f"{len(ops2)} 门")
            n_fail += 1

    # 固定特例：H·H → 恒等删除；H 单独保留（收益检查）；u3·H 合并
    ops = [Inst("u3", (0,), _U3_H), Inst("u3", (0,), _U3_H)]
    ops2 = list(ops)
    _merge_runs_u3cz(ops2)
    if ops2:
        print("[FAIL] 特例 H·H 未删除")
        n_fail += 1
    ops = [Inst("u3", (0,), _U3_H)]
    ops2 = list(ops)
    _merge_runs_u3cz(ops2)
    if len(ops2) != 1 or ops2[0].name != "u3":
        print("[FAIL] 特例孤立 H 形态被破坏")
        n_fail += 1
    ops = [Inst("u3", (0,), (0.4, 0.6, 0.8)), Inst("u3", (0,), _U3_H)]
    ops2 = list(ops)
    _merge_runs_u3cz(ops2)
    if len(ops2) != 1:
        print("[FAIL] 特例 u3·H 未合并")
        n_fail += 1
    # x·rz·x = rz(−θ)：3 门 → 1 门（矩阵级自动捕获共轭恒等式）
    ops = [Inst("x", (0,)), Inst("rz", (0,), (0.7,)), Inst("x", (0,))]
    ops2 = list(ops)
    _merge_runs_u3cz(ops2)
    if len(ops2) != 1:
        print("[FAIL] 特例 x·rz·x 未坍缩")
        n_fail += 1
    print(f"矩阵级 {30} 随机 + 4 特例：失败 {n_fail}")
    return n_fail


def test_circuit_equivalence():
    """线路级：decompose_linear vs v4 参照，态矢量重叠 = 1。"""
    rng = np.random.default_rng(42)
    n_fail = 0
    n = 5
    _1Q = ("u3", "u1", "h", "x")

    def rand_circ(n_gates):
        ops = []
        for _ in range(n_gates):
            g = rng.integers(0, 10)
            q = int(rng.integers(0, n))
            if g < 4:   # 1Q
                t = rng.integers(0, 4)
                if t == 0:
                    ops.append(Inst("u3", (q,),
                                    tuple(rng.uniform(0, 2 * np.pi, 3))))
                elif t == 1:
                    ops.append(Inst("u1", (q,),
                                    (float(rng.uniform(0, 2 * np.pi)),)))
                elif t == 2:
                    ops.append(Inst("h", (q,)))
                else:
                    ops.append(Inst("x", (q,)))
            elif g < 8:  # cx
                b = int(rng.integers(0, n))
                while b == q:
                    b = int(rng.integers(0, n))
                ops.append(Inst("cx", (q, b)))
            else:        # swap
                b = int(rng.integers(0, n))
                while b == q:
                    b = int(rng.integers(0, n))
                ops.append(Inst("swap", (q, b)))
        return ops

    def make_circ(ops):
        c = Circuit(n)
        c.ops = list(ops)
        return c

    def run_case(ops, label, init):
        c = make_circ(ops)
        ref = decompose_to_platform(c)
        lin = decompose_linear(c, spec=None, a2_rounds=0)
        sv_in = None if init is None else init
        sv1 = statevec(ref.ops, n, sv_in)
        sv2 = statevec(lin.ops, n, sv_in)
        ov = overlap(sv1, sv2)
        ok_g, bad = gate_set_ok(lin.ops)
        ok = ov > 1 - 1e-9 and ok_g
        print(f"[{'OK ' if ok else 'FAIL'}] {label}: 门 {len(ops)}→"
              f"v4 {len(ref.ops)} / linear {len(lin.ops)} "
              f"重叠 {ov:.10f}" + ("" if ok_g else f" 非法门 {bad[:3]}"))
        return 0 if ok else 1

    # 固定用例：h/x 密集 + 连续 cx（H·H 对消路径）+ swap
    fixed = [
        Inst("h", (0,)), Inst("x", (1,)), Inst("cx", (0, 1)),
        Inst("cx", (1, 2)), Inst("h", (2,)), Inst("u3", (2,),
                                                   (0.3, 0.5, 0.7)),
        Inst("swap", (2, 3)), Inst("cx", (3, 0)), Inst("x", (3,)),
        Inst("u1", (4,), (0.9,)), Inst("cx", (4, 3)),
        Inst("u3", (4,), (1.2, 0.4, 2.1)),
    ]
    # rz(π) 模式用例（2026-11-12 覆盖缺口教训：Z 剥离负结果——
    # 语义验证必须含机制触碰的模式）：[sx,rz(π),sx]（=i·rz(π) 三合一）、
    # rz(π) 链、rz(π) 夹 cz 对、x·rz(π)·x = rz(−π)
    fixed_z = [
        Inst("sx", (0,)), Inst("rz", (0,), (np.pi,)), Inst("sx", (0,)),
        Inst("cz", (0, 1)),
        Inst("rz", (1,), (np.pi,)), Inst("cz", (1, 2)),
        Inst("rz", (1,), (np.pi,)),
        Inst("x", (2,)), Inst("rz", (2,), (np.pi,)), Inst("x", (2,)),
        Inst("cx", (2, 3)), Inst("rz", (3,), (np.pi,)),
        Inst("sx", (3,)), Inst("rz", (3,), (np.pi,)),
    ]
    rng_state = np.random.default_rng(1)
    sv_rnd = rng_state.normal(size=2 ** n) + 1j * rng_state.normal(
        size=2 ** n)
    sv_rnd /= np.linalg.norm(sv_rnd)
    # 随机直积态（每比特独立——局部相位差异也会暴露）
    prod = np.array([1.0 + 0j])
    for _ in range(n):
        v = rng_state.normal(size=2) + 1j * rng_state.normal(size=2)
        v /= np.linalg.norm(v)
        prod = np.kron(prod, v)
    # 纠缠态（GHZ 形）
    ent = np.zeros(2 ** n, dtype=complex)
    ent[0] = 1 / np.sqrt(2)
    ent[2 ** n - 1] = 1 / np.sqrt(2)

    n_fail += run_case(fixed, "固定用例（h/x/cx/swap）|0⟩", None)
    n_fail += run_case(fixed, "固定用例 随机矢量", sv_rnd)
    n_fail += run_case(fixed, "固定用例 随机直积", prod)
    n_fail += run_case(fixed, "固定用例 GHZ 纠缠", ent)

    # rz(π) 模式用例 × 三种输入态（Z 类模式的语义门禁）
    n_fail += run_case(fixed_z, "rz(π) 模式 |0⟩", None)
    n_fail += run_case(fixed_z, "rz(π) 模式 随机矢量", sv_rnd)
    n_fail += run_case(fixed_z, "rz(π) 模式 随机直积", prod)

    for t in range(6):
        ops = rand_circ(int(rng.integers(30, 70)))
        init = sv_rnd if t % 2 else None
        n_fail += run_case(ops, f"随机用例 {t}"
                           + (" 随机矢量" if t % 2 else " |0⟩"), init)
    # 离散角随机用例：角度从 {0, π/2, π, 均匀随机} 混合抽取——
    # rz(π)/rz(π/2) 模式必然出现（连续均匀随机命中概率≈0 的覆盖缺口）
    for t in range(4):
        ops = []
        for _ in range(int(rng.integers(30, 60))):
            g = rng.integers(0, 10)
            q = int(rng.integers(0, n))
            if g < 4:
                ang = rng.choice([0.0, np.pi / 2, np.pi,
                                  float(rng.uniform(0, 2 * np.pi))])
                ops.append(Inst("rz", (q,), (ang,)))
            elif g < 6:
                ops.append(Inst("sx", (q,)))
            elif g < 8:
                ops.append(Inst("x", (q,)))
            else:
                b = int(rng.integers(0, n))
                while b == q:
                    b = int(rng.integers(0, n))
                ops.append(Inst("cx", (q, b)))
        init = sv_rnd if t % 2 else None
        n_fail += run_case(ops, f"离散角随机 {t}"
                           + (" 随机矢量" if t % 2 else " |0⟩"), init)
    return n_fail


def test_expand_merge_stages():
    """分阶段一致性：expand → merge → translate 与整体调用同产物。"""
    n_fail = 0
    c = Circuit(4)
    c.ops = [Inst("u3", (0,), (0.3, 0.5, 0.7)), Inst("cx", (0, 1)),
             Inst("u3", (1,), (1.1, 0.2, 0.9)), Inst("u1", (2,), (0.4,)),
             Inst("swap", (1, 2)), Inst("h", (3,)), Inst("cx", (3, 0))]
    ops = list(c.ops)
    ops = _translate_v4(ops)
    _merge_runs_cross(ops)
    from qtrail.pure.post_v2 import slide_x_through_cz
    slide_x_through_cz(ops, max_rounds=12)
    _cancel_cz_global(ops)
    _mar_from_ops(ops)
    _strip_identity_1q(ops)
    ref = decompose_linear(c, spec=None, a2_rounds=0, cancel_cx=False)
    same = [ (a.name, a.qubits, a.params) for a in ops ] == \
           [ (a.name, a.qubits, a.params) for a in ref.ops ]
    if not same:
        print("[FAIL] 分阶段产物与整体调用不一致")
        n_fail += 1
    print(f"分阶段一致性：{'OK' if not n_fail else 'FAIL'}")
    return n_fail


def test_min_candidates():
    """_u3_to_rzsx_min 候选族矩阵往返：每个候选的乘积 = u3（模相位）。"""
    from qtrail.pure.post import _u3_to_rzsx_min

    def gmat(name, params):
        if name == "rz":
            h = params[0] / 2.0
            return np.diag([np.exp(-1j * h), np.exp(1j * h)])
        if name == "sx":
            return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
        return np.array([[0.0, 1.0], [1.0, 0.0]])

    def u3mat(t, p, l):
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array([[c, -np.exp(1j * l) * s],
                         [np.exp(1j * p) * s, np.exp(1j * (p + l)) * c]])

    rng = np.random.default_rng(3)
    n_fail = 0
    # 随机 u3（任意角）+ 族点 u3（θ=π/2、θ=π、θ=0）
    cases = [(float(rng.uniform(0, np.pi)), float(rng.uniform(0, 2 * np.pi)),
              float(rng.uniform(0, 2 * np.pi))) for _ in range(30)]
    cases += [(np.pi / 2, float(rng.uniform(0, 2 * np.pi)),
               float(rng.uniform(0, 2 * np.pi))) for _ in range(30)]
    cases += [(np.pi, float(rng.uniform(0, 2 * np.pi)),
               float(rng.uniform(0, 2 * np.pi))) for _ in range(30)]
    cases += [(0.0, float(rng.uniform(0, 2 * np.pi)),
               float(rng.uniform(0, 2 * np.pi))) for _ in range(10)]
    for t, p, l in cases:
        U = u3mat(t, p, l)
        V = np.eye(2, dtype=complex)
        for name, params in _u3_to_rzsx_min(t, p, l):
            V = gmat(name, params) @ V
        ov = abs(np.trace(V.conj().T @ U)) / 2.0
        if ov < 1 - 1e-9:
            print(f"[FAIL] 候选族往返 t={t:.4f} 重叠 {ov:.6f} "
                  f"序列 {_u3_to_rzsx_min(t, p, l)}")
            n_fail += 1
    # 特例：sx 形单门 + H 三门 + x 单门
    for params, expect in (((np.pi / 2, -np.pi / 2, np.pi / 2), 1),
                           ((np.pi / 2, 0.0, np.pi), 3),
                           ((np.pi, 0.0, np.pi), 1)):
        got = _u3_to_rzsx_min(*params)
        if len(got) != expect:
            print(f"[FAIL] 特例 {params} 期望 {expect} 门 得 {got}")
            n_fail += 1
    print(f"候选族矩阵往返 {len(cases)} + 3 特例：失败 {n_fail}")
    return n_fail


def main():
    n_fail = 0
    n_fail += test_min_candidates()
    n_fail += test_matrix_merge()
    n_fail += test_circuit_equivalence()
    n_fail += test_expand_merge_stages()
    print(f"\n===== 语义验证：失败 {n_fail} =====")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
