"""QCIS export: primary via cqlib's converter, fallback minimal emitter.

QCIS is the Tianyan/GuoDun platform instruction set (e.g. `X Q1`,
`CZ Q0 Q1`, `RZ Q2 0.5`, `MEASURE Q3`). The fallback covers the platform
basis [rz, sx, x, cz] + measure, which is everything our pipeline emits.
"""
from __future__ import annotations

import math
from pathlib import Path

from qiskit import QuantumCircuit


def qasm_to_qcis_cqlib(qasm_str: str) -> str:
    """Convert OpenQASM 2.0 to QCIS via cqlib (best-effort; raises on failure)."""
    from cqlib.utils.qasm_to_qcis import QasmToQcis  # guarded import
    return QasmToQcis().convert_to_qcis(qasm_str)


def qasm_to_qcis_fallback(qc: QuantumCircuit) -> str:
    """Minimal QCIS emitter for the platform basis gates."""
    lines = []
    for inst in qc.data:
        name = inst.operation.name
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if name == "x":
            lines.append(f"X Q{qs[0]}")
        elif name == "sx":
            # SX = RZ(-pi/2) X RZ(pi/2): emit the universal safe form
            lines.append(f"RZ Q{qs[0]} {math.pi / 2}")
            lines.append(f"X Q{qs[0]}")
            lines.append(f"RZ Q{qs[0]} {-math.pi / 2}")
        elif name == "rz":
            lines.append(f"RZ Q{qs[0]} {float(inst.operation.params[0])}")
        elif name == "cz":
            lines.append(f"CZ Q{qs[0]} Q{qs[1]}")
        elif name == "measure":
            lines.append(f"MEASURE Q{qs[0]}")
        elif name in ("barrier", "reset"):
            continue
        else:
            # H/Y/Z etc. only appear if the circuit was NOT decomposed to the
            # platform basis — tell the caller exactly what to do
            raise ValueError(f"gate '{name}' has no minimal-QCIS form; "
                             "transpile to basis [rz, sx, x, cz] first")
    return "\n".join(lines)


def circuit_to_qcis(qc: QuantumCircuit, qasm_str: str | None = None) -> str:
    """QCIS text for a platform-basis circuit; cqlib primary, fallback emitter."""
    qasm_str = qasm_str or qasm2_dumps(qc)
    try:
        return qasm_to_qcis_cqlib(qasm_str)
    except Exception:
        return qasm_to_qcis_fallback(qc)


def qasm2_dumps(qc: QuantumCircuit) -> str:
    from qiskit import qasm2
    return qasm2.dumps(qc)


def write_qcis(qc: QuantumCircuit, path: str | Path) -> str:
    """Write QCIS file; returns the QCIS text."""
    text = circuit_to_qcis(qc)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
