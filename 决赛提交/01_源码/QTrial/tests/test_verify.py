"""功能等价性验证测试：精确/随机化模式 + 平台接口预留。"""
import numpy as np
import pytest
from qiskit import QuantumCircuit

from qtrail.verify.equivalence import (exact_fidelity, randomized_fidelity,
                                       verify_equivalence)
from qtrail.verify.platform import TianyanVerifier, _compare_distributions


def _circuit_4q():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rx(np.pi / 3, 0)
    qc.rz(np.pi / 5, 2)
    qc.cx(3, 1)
    qc.h(3)
    return qc


def test_exact_fidelity_one_for_identical():
    qc = _circuit_4q()
    assert exact_fidelity(qc, qc.copy(), None) == pytest.approx(1.0, abs=1e-9)


def test_exact_fidelity_detects_broken_circuit():
    qc = _circuit_4q()
    broken = qc.copy()
    # 额外插入一个 cx —— 功能被破坏
    broken.cx(0, 2)
    fid = exact_fidelity(qc, broken, None)
    assert fid < 0.999


def test_verify_equivalence_exact_pass():
    qc = _circuit_4q()
    r = verify_equivalence(qc, qc.copy(), layout=None)
    assert r["equivalent"] is True
    assert r["method"] == "exact"
    assert r["fidelity"] >= 0.999


def test_verify_equivalence_exact_fail():
    qc = _circuit_4q()
    broken = qc.copy()
    broken.x(0)
    r = verify_equivalence(qc, broken, layout=None)
    assert r["equivalent"] is False


def test_verify_equivalence_with_layout_permutation():
    """布局置换后的等价线路应验证通过（全局相位不敏感）。"""
    qc = _circuit_4q()
    permuted = QuantumCircuit(4)
    layout = {0: 2, 1: 3, 2: 0, 3: 1}
    for inst in qc.data:
        qs = [layout[qc.find_bit(q).index] for q in inst.qubits]
        permuted.append(inst.operation, [permuted.qubits[i] for i in qs], [])
    r = verify_equivalence(qc, permuted, layout=layout)
    assert r["equivalent"] is True


def test_randomized_mode_on_22q():
    """22 比特触发随机化模式，等价线路应通过。"""
    rng = np.random.default_rng(0)
    qc = QuantumCircuit(22)
    qc.h(range(22))
    for i in range(21):
        qc.cx(i, i + 1)
    mapped = qc.copy()  # 与自身比较（等价）
    r = verify_equivalence(qc, mapped, layout=None, max_qubits=20,
                           n_samples=4)
    assert r["method"].startswith("randomized")
    assert r["equivalent"] is True


def test_platform_verifier_needs_key():
    v = TianyanVerifier(login_key=None)
    assert v.available() is False
    with pytest.raises(RuntimeError, match="TIANYAN_LOGIN_KEY"):
        v._get_platform()


def test_platform_verifier_key_provided():
    import os
    os.environ["TIANYAN_LOGIN_KEY"] = "test-key-placeholder"
    try:
        v = TianyanVerifier()
        assert v.available() is True
    finally:
        os.environ.pop("TIANYAN_LOGIN_KEY", None)


def test_compare_distributions_identical():
    r = _compare_distributions({"00": 50, "11": 50}, {"00": 50, "11": 50})
    assert r["equivalent"] is True
    assert r["classical_fidelity"] == pytest.approx(1.0, abs=1e-6)
    assert r["tvd"] == pytest.approx(0.0, abs=1e-6)


def test_compare_distributions_different():
    r = _compare_distributions({"00": 100}, {"11": 100})
    assert r["equivalent"] is False
    assert r["classical_fidelity"] == pytest.approx(0.0, abs=1e-6)
    assert r["tvd"] == pytest.approx(1.0, abs=1e-6)
