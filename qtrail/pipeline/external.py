"""外部编译器接入：pytket 编译结果（统一口径：swap 展开为 3 cx）。

用于混合路由竞技：tket 的并行感知路由与优质放置提供深度/保真度下限。
"""
from __future__ import annotations

from qiskit import QuantumCircuit


def tket_compile(qc: QuantumCircuit, cm_edges: list, n_phys: int) -> tuple[QuantumCircuit, int]:
    """pytket 默认映射管线（DecomposeBoxes + DefaultMappingPass）。

    Returns (expanded_circuit, swap_count) where swap gates are materialized
    (implicit permutations replaced) and expanded to 3 cx — the same
    accounting convention as QTrial/Qiskit baselines.
    """
    from pytket.architecture import Architecture
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    from pytket.passes import (DecomposeBoxes, DefaultMappingPass, SequencePass)

    tk = qiskit_to_tk(qc)
    SequencePass([DecomposeBoxes(), DefaultMappingPass(Architecture(cm_edges))]).apply(tk)
    mapped = tk_to_qiskit(tk, replace_implicit_swaps=True)
    sc = mapped.count_ops().get("swap", 0)

    # 展开 swap -> 3 cx（统一 2Q 门与保真度口径）
    expanded = QuantumCircuit()
    for qr in mapped.qregs:
        expanded.add_register(qr)
    for cr in mapped.cregs:
        expanded.add_register(cr)
    for inst in mapped.data:
        if inst.operation.name == "swap":
            a = mapped.find_bit(inst.qubits[0]).index
            b = mapped.find_bit(inst.qubits[1]).index
            expanded.cx(a, b)
            expanded.cx(b, a)
            expanded.cx(a, b)
        else:
            expanded.append(inst.operation, inst.qubits, inst.clbits)
    return expanded, sc
