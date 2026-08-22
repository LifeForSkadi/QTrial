"""QCIS export: fallback emitter correctness on platform-basis circuits."""
import math

import pytest
from qiskit import QuantumCircuit

from qtrail.utils.qcis import qasm_to_qcis_fallback, circuit_to_qcis


def _platform_circuit():
    from qiskit import ClassicalRegister
    qc = QuantumCircuit(3, 3)
    qc.rz(math.pi / 2, 0)
    qc.sx(0)
    qc.x(1)
    qc.cz(1, 2)
    qc.measure(2, 0)
    return qc


def test_fallback_emitter_output():
    qc = _platform_circuit()
    text = qasm_to_qcis_fallback(qc)
    lines = text.splitlines()
    assert "RZ Q0 1.5707963267948966" in lines
    assert "CZ Q1 Q2" in lines
    assert "MEASURE Q2" in lines
    # sx expands to RZ X RZ
    assert lines.count("X Q0") == 1
    assert any(l.startswith("RZ Q0") for l in lines)


def test_fallback_rejects_unsupported_gates():
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValueError, match="transpile"):
        qasm_to_qcis_fallback(qc)


def test_circuit_to_qcis_primary_path():
    qc = _platform_circuit()
    text = circuit_to_qcis(qc)
    assert len(text.strip()) > 0
    # cqlib converter (primary) emits QCIS tokens like X2P/CZ/RZ
    assert any(tok in text for tok in ("X2P", "CZ", "RZ", "MEASURE"))
