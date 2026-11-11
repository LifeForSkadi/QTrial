# -*- coding: utf-8 -*-
"""fold_keep_coords 非空验（2026-11-13，审查代理 P0 修复验证）：
含 x/h 门的线路（|0⟩ 会变——坐标错误可检测）+ numba -1 腐蚀回归。
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.post_v2 import fold_keep_coords


def fid_3q(ops, fl):
    """3 比特线路 fid：fold_keep 产物 vs 交付判据目标。

    交付判据：U(C')|0⟩ = Π_fin·U(C)|0⟩——Π_fin 把逻辑 i 的读回
    搬到 fin(i)（目标第 p 位 = 原态第 fin⁻¹[p] 位）。
    原线路（逻辑坐标）= x(0), cx(0,2)——swap 是路由产物，不在
    逻辑线路里。ops = 路由器输出（发射时刻坐标）。
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    # 逻辑线路：x(0), cx(0,2)（审查代理用例"X(a); CX(a,c)"）
    qc_orig = QuantumCircuit(3)
    qc_orig.x(0)
    qc_orig.cx(0, 2)
    sv0 = Statevector.from_instruction(qc_orig).data
    inv = {int(fl[k]): k for k in fl}
    perm = [0] * 8
    for idx in range(8):
        new_idx = 0
        for p in range(3):
            if (idx >> inv[p]) & 1:
                new_idx |= 1 << p
        perm[idx] = new_idx
    target = np.array([sv0[perm[i]] for i in range(8)])
    c = Circuit(3, name="t")
    c.ops = list(ops)
    fk, fin = fold_keep_coords(c, fl)
    qc2 = QuantumCircuit(3)
    for inst in fk.ops:
        if inst.name == "x":
            qc2.x(int(inst.qubits[0]))
        elif inst.name == "cx":
            qc2.cx(int(inst.qubits[0]), int(inst.qubits[1]))
    sv2 = Statevector.from_instruction(qc2).data
    return float(abs(np.vdot(target, sv2)) ** 2)


# 玩具线路（审查代理用例）：路由器输出 x(0), swap(0,1), cx(1,2)（发射坐标）
ops = [Inst("x", (0,)), Inst("swap", (0, 1)), Inst("cx", (1, 2))]
fl = {0: 1, 1: 0, 2: 2}
fid = fid_3q(ops, fl)
print(f"玩具线路 fold_keep fid = {fid:.8f}（修复前 fid=0，修复后应 1.0）")
assert fid > 0.999999, "fold_keep 语义仍错"

# numba -1 腐蚀回归
from qtrail.pure.post_numba import has_numba, numba_push_swaps
if has_numba():
    c = Circuit(3, name="n")
    c.ops = [Inst("rz", (0,), (0.5,)), Inst("sx", (0,), ()),
             Inst("x", (1,), ()), Inst("cx", (1, 2))]
    out = numba_push_swaps(c, None)
    bad = [i for i in out if i.nq == 2 and i.name in ("rz", "sx", "x")]
    print(f"numba 腐蚀检查：{len(bad)} 个坏门（应 0）")
    assert not bad, f"numba 腐蚀残留: {bad}"
    print("numba -1 腐蚀回归通过")
else:
    print("（本机无 numba，跳过）")
print("全部非空验通过")
