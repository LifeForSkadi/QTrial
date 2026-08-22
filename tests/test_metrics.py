
def test_twoq_depth_is_depth_not_count():
    """回归：2Q 深度 = 2Q 关键路径层数，不是 2Q 门总数。"""
    from qiskit import QuantumCircuit
    from qtrail.pipeline.metrics import twoq_depth, twoq_count
    qc = QuantumCircuit(4)
    qc.cz(0, 1); qc.cz(2, 3)   # layer 1
    qc.cz(0, 2); qc.cz(1, 3)   # layer 2
    qc.cz(0, 3); qc.cz(1, 2)   # layer 3
    assert twoq_count(qc) == 6
    assert twoq_depth(qc) == 3
