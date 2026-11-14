"""OpenQASM 2.0 <-> qiskit utilities.

qiskit 2.x note: `QuantumCircuit.qasm()` was removed; use qiskit.qasm2.
"""
from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit, qasm2
from qiskit.compiler import transpile

# Basis used to decompose arbitrary gates down to 2Q + 1Q ops for graph building
DECOMPOSE_BASIS = ["cx", "u", "rx", "ry", "rz", "x", "y", "z", "s", "sdg", "t", "tdg", "h"]

# Default basis for the Tianyan-class device pipeline (QCIS native-ish set)
PLATFORM_BASIS = ["rz", "sx", "x", "cz"]


def decompose_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """Reduce any circuit to the standard 2Q/1Q basis.

    Custom QASM-defined gates (e.g. MQTBench's Oracle / qft_dg) are expanded
    recursively, then transpile (O1, incl. HighLevelSynthesis) reduces
    everything to the standard basis.
    """
    qc = qc.decompose(reps=4)  # expand custom gate definitions
    return transpile(qc, basis_gates=DECOMPOSE_BASIS, optimization_level=1)


def load_qasm2(path: str | Path, decompose: bool = True) -> QuantumCircuit:
    """Load an OpenQASM 2.0 file and optionally decompose to a standard basis.

    先经 sanitize_qasm 统一改写 qiskit.qasm2 不支持的门（cp/rxx/sx 等），
    保证任意标准 QASM 2.0 文件可解析。
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    qc = qasm2.loads(sanitize_qasm(text))
    if decompose:
        qc = decompose_circuit(qc)
    return qc


def strip_measurements(qc: QuantumCircuit) -> tuple[QuantumCircuit, list]:
    """Remove measurement/barrier ops, return (clean circuit, removed ops)."""
    import copy
    clean = qc.copy_empty_like()
    removed = []
    for inst in qc.data:
        if inst.operation.name in ("measure", "barrier"):
            removed.append(inst)
        else:
            clean.append(inst.operation, inst.qubits, inst.clbits)
    return clean, removed


def append_measurements(qc: QuantumCircuit, removed: list,
                        final_layout: dict | None = None,
                        original: QuantumCircuit | None = None) -> QuantumCircuit:
    """Re-append measurements, remapping qubits through the final layout.

    removed 中的指令属于原始线路（original 或 qc 本身）；路由后线路的
    比特对象不同，故用 original 解析逻辑比特索引。
    """
    src = original if original is not None else qc
    out = qc.copy()
    if final_layout is None:
        for inst in removed:
            if inst.operation.name == "measure":
                out.append(inst.operation, inst.qubits, inst.clbits)
        return out
    for inst in removed:
        if inst.operation.name == "measure":
            q = inst.qubits[0]
            qi = src.find_bit(q).index
            phys = final_layout.get(qi, qi)
            out.measure(phys, inst.clbits[0])
    return out


def extract_ops(qc: QuantumCircuit) -> list:
    """Extract (name, (q0, ...)) ops for program graph building."""
    ops = []
    for inst in qc.data:
        name = inst.operation.name
        qs = tuple(qc.find_bit(q).index for q in inst.qubits)
        if name in ("measure", "barrier", "reset"):
            continue
        if len(qs) > 2:
            raise ValueError(
                f"Gate '{name}' acts on {len(qs)} qubits; decompose the circuit "
                f"with basis {DECOMPOSE_BASIS} before building a program graph.")
        ops.append((name, qs))
    return ops


def sanitize_qasm(text: str) -> str:
    """Rewrite gates that qiskit.qasm2's parser does not recognize into
    qelib1-standard equivalents:
      u -> u3, p -> u1, sx -> rx(pi/2),
      rxx(θ) a,b -> cx a,b; rx(θ) b; cx a,b （模全局相位等价）
      rzz(θ) a,b -> cx a,b; rz(θ) b; cx a,b
    """
    import re

    QB = r"([A-Za-z_]\w*(?:\[[^\]]+\])?)"  # 任意比特标识符（q[0] / a[0] / q0）

    def _twoq_expand(m, rgate):
        """受控旋转门的等价展开（模全局相位）：cx 目标/控制 + 旋转 + cx。"""
        i, j, t = m.group(2), m.group(3), m.group(1)
        return (f"cx {i}, {j}; {rgate}({t}) {j}; cx {i}, {j};")

    def rxx(m):
        return _twoq_expand(m, "rx")

    def rzz(m):
        return _twoq_expand(m, "rz")

    def cp(m):
        lam = m.group(1)
        i, j = m.group(2), m.group(3)
        return (f"rz(({lam})/2) {i}; rz(({lam})/2) {j}; "
                f"cx {i}, {j}; rz(-({lam})/2) {j}; cx {i}, {j};")

    def swap(m):
        i, j = m.group(1), m.group(2)
        return f"cx {i}, {j}; cx {j}, {i}; cx {i}, {j};"

    def _crot(m, rgate):
        # cr?(θ) i,j ≡ r?(θ/2) j; cx i,j; r?(−θ/2) j; cx i,j（模全局相位）
        i, j, t = m.group(2), m.group(3), m.group(1)
        return (f"{rgate}(({t})/2) {j}; cx {i}, {j}; "
                f"{rgate}(-({t})/2) {j}; cx {i}, {j};")

    def crx(m):
        return _crot(m, "rx")

    def cry(m):
        return _crot(m, "ry")

    def crz(m):
        return _crot(m, "rz")

    text = text.replace("sx ", "rx(pi/2) ").replace("sx;", "rx(pi/2);")
    text = text.replace("sxdg ", "rx(-pi/2) ").replace("sxdg;", "rx(-pi/2);")
    text = re.sub(rf"rxx\(([^)]+)\)\s+{QB}\s*,\s*{QB};", rxx, text)
    text = re.sub(rf"rzz\(([^)]+)\)\s+{QB}\s*,\s*{QB};", rzz, text)
    text = re.sub(rf"\bcp\(([^)]+)\)\s+{QB}\s*,\s*{QB};", cp, text)
    text = re.sub(rf"\bcrx\(([^)]+)\)\s+{QB}\s*,\s*{QB};", crx, text)
    text = re.sub(rf"\bcry\(([^)]+)\)\s+{QB}\s*,\s*{QB};", cry, text)
    text = re.sub(rf"\bcrz\(([^)]+)\)\s+{QB}\s*,\s*{QB};", crz, text)
    text = re.sub(rf"\bswap\s+{QB}\s*,\s*{QB};", swap, text)
    text = re.sub(r"\bu\(", "u3(", text)
    text = re.sub(r"\bp\(", "u1(", text)
    return text


def qasm2_str(qc: QuantumCircuit) -> str:
    """Standard OpenQASM 2.0 (qelib1-only gates) so any conforming parser
    (including qiskit.qasm2) can read the file back."""
    return sanitize_qasm(qasm2.dumps(qc))


def write_qasm2(qc: QuantumCircuit, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(qasm2_str(qc))


def count_ops_after_transpile(qc: QuantumCircuit) -> dict:
    """Simple gate census of a transpiled circuit."""
    counts = {}
    for inst in qc.data:
        counts[inst.operation.name] = counts.get(inst.operation.name, 0) + 1
    return counts
