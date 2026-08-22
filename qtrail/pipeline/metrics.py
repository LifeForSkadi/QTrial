"""Circuit metrics: swap/2Q counts, depth, 2Q depth, estimated fidelity."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


def count_ops(qc: QuantumCircuit) -> dict:
    return dict(qc.count_ops())


def circuit_depth(qc: QuantumCircuit) -> int:
    return qc.depth()


def twoq_depth(qc: QuantumCircuit) -> int:
    """2Q 关键路径深度：只沿 DAG 最长路径数 2Q 门层。

    （2026-08-21 修复：旧实现遍历全部拓扑节点计数，实为 2Q 门总数而非
    深度——由 6 门/3 层手工电路暴露，已补回归测试。）
    """
    dag = circuit_to_dag(qc)
    twoq = {name for name, num in qc.count_ops().items() if num and
            _is_twoq(qc, name)}
    path = dag.longest_path()
    return sum(1 for node in path
               if getattr(node, "op", None) is not None and node.op.name in twoq)


def _is_twoq(qc: QuantumCircuit, name: str) -> bool:
    for inst in qc.data:
        if inst.operation.name == name:
            return len(inst.qubits) == 2
    return False


def twoq_count(qc: QuantumCircuit) -> int:
    total = 0
    for name, num in qc.count_ops().items():
        if _is_twoq(qc, name):
            total += num
    return total


def estimate_fidelity(qc: QuantumCircuit, calib, final_layout: dict | None = None,
                      measure_qubits: list | None = None) -> float:
    """Product-model fidelity estimate using per-qubit/edge calibration.

    F = prod_1Q (1-err1q[q]) * prod_2Q (1-err2q[e]) * prod_meas (1-errro[q])
    """
    f = 1.0
    err2q = calib.err_2q
    for inst in qc.data:
        name = inst.operation.name
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if name == "measure":
            q = qs[0]
            f *= max(1.0 - float(calib.err_ro[q]), 1e-12)
        elif len(qs) == 2:
            e = err2q.get((qs[0], qs[1]), err2q.get((qs[1], qs[0]), None))
            if e is None:
                e = float(np.median(list(err2q.values())))
            f *= max(1.0 - float(e), 1e-12)
        elif len(qs) == 1:
            f *= max(1.0 - float(calib.err_1q[qs[0]]), 1e-12)
        # barriers etc. ignored
    return float(f)


def mean_twoq_error(qc: QuantumCircuit, calib) -> float | None:
    """Gate-count-weighted mean 2Q error over the edges the circuit uses.

    Lower = the mapping placed the circuit on higher-fidelity couplers
    (the QTrail noise-avoidance success metric).
    """
    total_w = 0.0
    total_e = 0.0
    err2q = calib.err_2q
    for inst in qc.data:
        if len(inst.qubits) != 2:
            continue
        qs = tuple(sorted(qc.find_bit(q).index for q in inst.qubits))
        e = err2q.get(qs, err2q.get((qs[1], qs[0]), None))
        if e is None:
            continue
        total_w += 1.0
        total_e += float(e)
    return float(total_e / total_w) if total_w > 0 else None


def compute_metrics(routed: QuantumCircuit, swap_count: int,
                    calib, final_layout: dict | None = None) -> dict:
    """Full metric dict for a routed (platform-basis) circuit."""
    m = {
        "swap_count": swap_count,
        "twoq_count": twoq_count(routed),
        "depth": circuit_depth(routed),
        "twoq_depth": twoq_depth(routed),
        "est_fidelity": estimate_fidelity(routed, calib, final_layout),
        "gate_counts": count_ops(routed),
    }
    m2q = mean_twoq_error(routed, calib)
    if m2q is not None:
        m["mean_2q_err"] = m2q
    return m
