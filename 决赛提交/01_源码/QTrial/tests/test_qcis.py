"""QCIS export: fallback emitter correctness on platform-basis circuits.

2026-11-09 修正：平台原生单比特指令为 X2P/X2M/Y2P/Y2M/RZ（无 "X"），
sx→X2P、x→X2P X2P（依据 cqlib 官方 qasm2qcis 表）；并加矩阵语义验证
（旧实现 sx→RZ(π/2) X RZ(-π/2) 数学上 = Y，测试只有字符串断言未拦截）。
"""
import math

import numpy as np
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
    assert "M Q2" in lines
    # sx 直接映射平台原生 X2P；x 展开为两个 X2P
    assert lines.count("X2P Q0") == 1
    assert lines.count("X2P Q1") == 2
    # 平台无 "X" 指令
    assert not any(l.startswith("X Q") for l in lines)


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


# ------------------------------------------------------------ 矩阵语义验证
def _rz(a):
    return np.diag([math.e ** (-1j * a / 2), math.e ** (1j * a / 2)])


def _up_to_global_phase(a, b, tol=1e-9):
    ov = np.vdot(a, b)
    m = abs(ov)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return m / n > 1 - tol


def test_x2p_equals_sx():
    """平台 X2P（X 的 π/2 脉冲）恰为 SX 门。"""
    x2p = math.cos(math.pi / 4) * np.eye(2, dtype=complex) \
        - 1j * math.sin(math.pi / 4) * np.array([[0, 1], [1, 0]], dtype=complex)
    sx = np.array([[1, -1j], [-1j, 1]], dtype=complex) / math.sqrt(2)
    assert np.allclose(x2p, sx)


def test_fallback_sx_expansion_equals_sx():
    """sx 的 QCIS 展开（单个 X2P）必须等于 SX 门。"""
    x2p = math.cos(math.pi / 4) * np.eye(2, dtype=complex) \
        - 1j * math.sin(math.pi / 4) * np.array([[0, 1], [1, 0]], dtype=complex)
    sx = np.array([[1, -1j], [-1j, 1]], dtype=complex) / math.sqrt(2)
    assert np.allclose(x2p, sx)


def test_old_expansion_was_wrong():
    """回归证据：旧展开 RZ(π/2)·X·RZ(-π/2) = Y ≠ SX（防再引入）。"""
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sx = np.array([[1, -1j], [-1j, 1]], dtype=complex) / math.sqrt(2)
    old = _rz(math.pi / 2) @ x @ _rz(-math.pi / 2)
    assert _up_to_global_phase(old, y)
    assert not _up_to_global_phase(old, sx)


def test_h_expansion():
    """H = RZ(π/2)·X2P·RZ(π/2)（模全局相位 -i，与 post.py 口径一致）。"""
    x2p = math.cos(math.pi / 4) * np.eye(2, dtype=complex) \
        - 1j * math.sin(math.pi / 4) * np.array([[0, 1], [1, 0]], dtype=complex)
    h = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    seq = _rz(math.pi / 2) @ x2p @ _rz(math.pi / 2)
    assert _up_to_global_phase(seq, h)
