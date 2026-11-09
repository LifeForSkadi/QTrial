"""qiskit-free 输出：QASM 2.0 写出 + QCIS 发射器（天衍平台格式）。

QCIS 指令集（依据 cqlib 官方 qasm2qcis 表 utils/const.py，2026-11-09 修正）：
平台原生单比特门 = RZ / X2P / X2M / Y2P / Y2M（X2P 即 SX=exp(-i(π/2)X/2)），
**没有 "X" 指令**（Pauli X 用两个 X2P 表示）。映射：
  rz(λ) → RZ Q λ；sx → X2P Q；x → X2P Q + X2P Q；
  cz → CZ Q Q；u3=H → RZ(π/2) X2P RZ(π/2)（= -i·H，模全局相位，
  与 post.py 的 H 分解口径一致）；
  测量指令用平台要求的 M（2026-08-22 真机校验修正）。
旧版 sx→"RZ(π/2) X RZ(-π/2)" 在数学上 = Y（非 SX），且 "X" 非平台
指令——已按 cqlib 官方转换器输出（X2P）修正；支持平台比特标签
（Q1..Q105，qubit_labels 参数）。
"""
from __future__ import annotations

import math

from qtrail.pure.circuit import Circuit

H_TH = math.pi / 2


def to_qasm(circ: Circuit) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{circ.n}];"]
    for inst in circ.ops:
        if inst.name == "u3":
            th, ph, la = inst.params
            if abs(th - H_TH) < 1e-9 and abs(ph) < 1e-9 \
                    and abs(la - math.pi) < 1e-9:
                lines.append(f"h q[{inst.qubits[0]}];")
            else:
                lines.append(f"u3({th},{ph},{la}) q[{inst.qubits[0]}];")
        elif inst.name == "u1":
            lines.append(f"u1({inst.params[0]}) q[{inst.qubits[0]}];")
        elif inst.name == "rz":
            lines.append(f"rz({inst.params[0]}) q[{inst.qubits[0]}];")
        elif inst.name == "sx":
            lines.append(f"sx q[{inst.qubits[0]}];")
        elif inst.name == "x":
            lines.append(f"x q[{inst.qubits[0]}];")
        elif inst.name == "cz":
            lines.append(f"cz q[{inst.qubits[0]}],q[{inst.qubits[1]}];")
        elif inst.name == "id":
            lines.append(f"id q[{inst.qubits[0]}];")
    if circ.measures:
        lines.append(f"creg c[{len(circ.measures)}];")
        for k, inst in enumerate(circ.measures):
            lines.append(f"measure q[{inst.qubits[0]}] -> c[{k}];")
    return "\n".join(lines) + "\n"


def to_qcis(circ: Circuit, labels=None) -> str:
    """QCIS 文本（平台基；labels 为平台比特标签数组，None = 用内部索引）。

    u3 仅接受 H 形式（其余报错提示——纯管线输出经平台基分解后无 u3）。
    """
    def lab(q):
        return int(labels[q]) if labels is not None else int(q)

    out = []
    for inst in circ.ops:
        if inst.name == "u3":
            th, ph, la = inst.params
            if abs(th - H_TH) < 1e-9 and abs(ph) < 1e-9 \
                    and abs(la - math.pi) < 1e-9:
                # H = RZ(π/2)·X2P·RZ(π/2)（模全局相位 -i）
                out.append(f"RZ Q{lab(inst.qubits[0])} {math.pi / 2}")
                out.append(f"X2P Q{lab(inst.qubits[0])}")
                out.append(f"RZ Q{lab(inst.qubits[0])} {math.pi / 2}")
            else:
                raise ValueError(
                    f"u3({th},{ph},{la}) has no minimal-QCIS form")
        elif inst.name == "u1":
            out.append(f"RZ Q{lab(inst.qubits[0])} {float(inst.params[0])}")
        elif inst.name == "rz":
            out.append(f"RZ Q{lab(inst.qubits[0])} {float(inst.params[0])}")
        elif inst.name == "sx":
            # 平台原生 SX = X2P（cqlib 官方表：\"sx\": \"X2P\"）
            out.append(f"X2P Q{lab(inst.qubits[0])}")
        elif inst.name == "x":
            # 平台无 X 指令：Pauli X = X2P·X2P（cqlib 官方表同款）
            out.append(f"X2P Q{lab(inst.qubits[0])}")
            out.append(f"X2P Q{lab(inst.qubits[0])}")
        elif inst.name == "cz":
            out.append(f"CZ Q{lab(inst.qubits[0])} Q{lab(inst.qubits[1])}")
        elif inst.name == "id":
            continue
    for inst in circ.measures:
        out.append(f"M Q{lab(inst.qubits[0])}")
    return "\n".join(out)
