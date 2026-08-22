"""qiskit-free 输出：QASM 2.0 写出 + QCIS 发射器（天衍平台格式）。

QCIS 格式与 qtrail/utils/qcis.py 的 fallback 发射器一致：
X / RZ / CZ / M；u3=H → RZ(pi/2) X RZ(pi/2)，u1(λ) → RZ(λ)。
（测量指令用平台要求的 M 而非 MEASURE——2026-08-22 真机校验修正。）
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


def to_qcis(circ: Circuit) -> str:
    """QCIS 文本（平台基：u3 仅接受 H 形式；其余报错提示）。"""
    out = []
    for inst in circ.ops:
        if inst.name == "u3":
            th, ph, la = inst.params
            if abs(th - H_TH) < 1e-9 and abs(ph) < 1e-9 \
                    and abs(la - math.pi) < 1e-9:
                out.append(f"RZ Q{inst.qubits[0]} {math.pi / 2}")
                out.append(f"X Q{inst.qubits[0]}")
                out.append(f"RZ Q{inst.qubits[0]} {math.pi / 2}")
            else:
                raise ValueError(
                    f"u3({th},{ph},{la}) has no minimal-QCIS form")
        elif inst.name == "u1":
            out.append(f"RZ Q{inst.qubits[0]} {float(inst.params[0])}")
        elif inst.name == "rz":
            out.append(f"RZ Q{inst.qubits[0]} {float(inst.params[0])}")
        elif inst.name == "sx":
            out.append(f"RZ Q{inst.qubits[0]} {math.pi / 2}")
            out.append(f"X Q{inst.qubits[0]}")
            out.append(f"RZ Q{inst.qubits[0]} {-math.pi / 2}")
        elif inst.name == "x":
            out.append(f"X Q{inst.qubits[0]}")
        elif inst.name == "cz":
            out.append(f"CZ Q{inst.qubits[0]} Q{inst.qubits[1]}")
        elif inst.name == "id":
            continue
    for inst in circ.measures:
        out.append(f"M Q{inst.qubits[0]}")
    return "\n".join(out)
