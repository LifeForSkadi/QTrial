"""qiskit-free 输出：QASM 2.0 写出 + QCIS 发射器（天衍平台格式）。

QCIS 格式与 qtrail/utils/qcis.py 的 fallback 发射器一致：
X / RZ / CZ / MEASURE；u3=H → RZ(pi/2) X RZ(pi/2)，u1(λ) → RZ(λ)。
"""
from __future__ import annotations

import math

from qtrail.pure.circuit import Circuit

H_TH = math.pi / 2


def _lab(q: int, labels) -> str:
    """内部索引 → 输出标签（labels 缺省恒等映射）。"""
    return str(int(labels[q])) if labels is not None else str(q)


def to_qasm(circ: Circuit, labels=None) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{circ.n}];"]
    for inst in circ.ops:
        qs = [_lab(q, labels) for q in inst.qubits]
        if inst.name == "u3":
            th, ph, la = inst.params
            if abs(th - H_TH) < 1e-9 and abs(ph) < 1e-9 \
                    and abs(la - math.pi) < 1e-9:
                lines.append(f"h q[{qs[0]}];")
            else:
                lines.append(f"u3({th},{ph},{la}) q[{qs[0]}];")
        elif inst.name == "u1":
            lines.append(f"u1({inst.params[0]}) q[{qs[0]}];")
        elif inst.name == "rz":
            lines.append(f"rz({inst.params[0]}) q[{qs[0]}];")
        elif inst.name == "sx":
            lines.append(f"sx q[{qs[0]}];")
        elif inst.name == "x":
            lines.append(f"x q[{qs[0]}];")
        elif inst.name == "cz":
            lines.append(f"cz q[{qs[0]}],q[{qs[1]}];")
        elif inst.name == "id":
            lines.append(f"id q[{qs[0]}];")
    if circ.measures:
        lines.append(f"creg c[{len(circ.measures)}];")
        for k, inst in enumerate(circ.measures):
            lines.append(f"measure q[{_lab(inst.qubits[0], labels)}] -> c[{k}];")
    return "\n".join(lines) + "\n"


def to_qcis(circ: Circuit, labels=None) -> str:
    """QCIS 文本（平台基：u3 仅接受 H 形式；其余报错提示）。

    labels：真实平台比特标签数组（live 校准时经 spec.qubit_labels 传入），
    缺省按内部索引输出。
    """
    out = []
    for inst in circ.ops:
        qs = [_lab(q, labels) for q in inst.qubits]
        if inst.name == "u3":
            th, ph, la = inst.params
            if abs(th - H_TH) < 1e-9 and abs(ph) < 1e-9 \
                    and abs(la - math.pi) < 1e-9:
                out.append(f"RZ Q{qs[0]} {math.pi / 2}")
                out.append(f"X Q{qs[0]}")
                out.append(f"RZ Q{qs[0]} {math.pi / 2}")
            else:
                raise ValueError(
                    f"u3({th},{ph},{la}) has no minimal-QCIS form")
        elif inst.name == "u1":
            out.append(f"RZ Q{qs[0]} {float(inst.params[0])}")
        elif inst.name == "rz":
            out.append(f"RZ Q{qs[0]} {float(inst.params[0])}")
        elif inst.name == "sx":
            out.append(f"RZ Q{qs[0]} {math.pi / 2}")
            out.append(f"X Q{qs[0]}")
            out.append(f"RZ Q{qs[0]} {-math.pi / 2}")
        elif inst.name == "x":
            out.append(f"X Q{qs[0]}")
        elif inst.name == "cz":
            out.append(f"CZ Q{qs[0]} Q{qs[1]}")
        elif inst.name == "id":
            continue
    for inst in circ.measures:
        out.append(f"M Q{_lab(inst.qubits[0], labels)}")
    return "\n".join(out)
