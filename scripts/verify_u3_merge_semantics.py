# -*- coding: utf-8 -*-
"""merge_u3_runs 语义等价验证（2026-11-12，补充三十六预注册门禁）。

玩具线路：同比特 u3 对（列表中被 cx 交错）+ 不同比特 u3 相邻——
验证滑动+合并后态矢量逐位一致（全局相位已剥离口径，重叠=1）。
多组随机角度 + 固定用例，确保矩阵乘序与相位剥离精确。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qtrail.pure.circuit import Inst
from qtrail.pure.post_v2 import merge_u3_runs


def statevec(ops, n):
    qc = QuantumCircuit(n)
    for inst in ops:
        if inst.name == "u3":
            qc.u(*inst.params, int(inst.qubits[0]))
        elif inst.name == "cx":
            qc.cx(int(inst.qubits[0]), int(inst.qubits[1]))
        elif inst.name == "rz":
            qc.rz(inst.params[0], int(inst.qubits[0]))
        elif inst.name == "sx":
            qc.sx(int(inst.qubits[0]))
        elif inst.name == "x":
            qc.x(int(inst.qubits[0]))
        else:
            raise ValueError(inst.name)
    return Statevector.from_instruction(qc).data


def check(ops, n, label):
    ops2 = list(ops)
    merged = merge_u3_runs(ops2)
    sv1 = statevec(ops, n)
    sv2 = statevec(ops2, n)
    ov = abs(np.vdot(sv1, sv2))
    ok = ov > 1 - 1e-9 and len(ops2) <= len(ops)
    print(f"[{'OK ' if ok else 'FAIL'}] {label}: 门 {len(ops)}->{len(ops2)} "
          f"合并 {merged} 对 重叠 {ov:.10f}")
    return ok


def main():
    rng = np.random.default_rng(42)
    n_fail = 0

    # 固定用例：u3 对被 cx 交错（滑动必要）+ 相邻 u3 对
    n_fail += not check([
        Inst("u3", (0,), (0.3, 0.5, 0.7)),
        Inst("cx", (1, 2)),
        Inst("u3", (0,), (1.1, 2.2, 0.4)),
        Inst("cx", (0, 1)),
        Inst("u3", (1,), (0.9, 0.1, 0.2)),
        Inst("cx", (1, 2)),
        Inst("u3", (1,), (1.7, 0.6, 2.9)),
        Inst("u3", (2,), (0.4, 0.8, 1.2)),
        Inst("u3", (2,), (2.1, 0.9, 0.3)),
    ], 3, "固定用例（交错+相邻）")

    # 随机用例 × 5：随机 u3 对与随机 cx 交错
    for t in range(5):
        ops = []
        for i in range(10):
            ops.append(Inst("u3", (0,), tuple(rng.uniform(0, 2 * np.pi, 3))))
            if i % 2 == 0:
                ops.append(Inst("cx", (1, 2)))
            if i % 3 == 1:
                ops.append(Inst("cx", (0, 2)))
        n_fail += not check(ops, 3, f"随机用例 {t}")

    # 边界：rz 与 u3 相邻（不合并 rz——只合并 u3 对）+ swap 阻塞
    n_fail += not check([
        Inst("u3", (0,), (0.3, 0.5, 0.7)),
        Inst("rz", (0,), (0.2,)),
        Inst("cx", (1, 2)),
        Inst("u3", (0,), (1.1, 2.2, 0.4)),
        Inst("u3", (1,), (0.5, 0.6, 0.7)),
        Inst("cx", (0, 1)),
        Inst("u3", (1,), (2.0, 0.3, 1.1)),
    ], 3, "rz 不参与合并")

    print(f"\n失败 {n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
