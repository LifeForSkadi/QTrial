"""深度感知路由器测试：正确性（态矢量等价）与基本性质。"""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

from qtrail.pipeline.lexiroute import lexiroute


def _grid4x4():
    return CouplingMap.from_grid(4, 4)


def test_lexiroute_statevector_equivalence():
    """路由后线路与原线路态矢量等价（全局相位不敏感）。"""
    cm = _grid4x4()
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rx(np.pi / 3, 0)
    qc.rz(np.pi / 5, 2)
    qc.cx(3, 1)
    qc.h(3)
    layout = {0: 0, 1: 1, 2: 2, 3: 3}
    routed, swaps, final = lexiroute(qc, cm, layout, seed=0)

    # 原线路嵌入最终映射后的物理空间
    full = QuantumCircuit(cm.size())
    for inst in qc.data:
        qs = [final[qc.find_bit(q).index] for q in inst.qubits]
        full.append(inst.operation, [full.qubits[i] for i in qs], [])
    sv1 = Statevector(full)
    sv2 = Statevector(routed)
    fid = abs(np.dot(np.conj(sv1.data), sv2.data)) ** 2
    assert fid >= 0.999, f"fidelity {fid}"


def test_lexiroute_all_twoq_gates_executed():
    cm = _grid4x4()
    qc = QuantumCircuit(4)
    qc.cx(0, 3)   # 距离远，需要路由
    qc.cx(1, 3)
    qc.cx(0, 2)
    routed, swaps, _ = lexiroute(qc, cm, {0: 0, 1: 1, 2: 2, 3: 3}, seed=0)
    n_cx = routed.count_ops().get("cx", 0)
    assert n_cx == 3  # 原始 3 个 cx 全部执行
    assert swaps >= 1  # 需要插入 SWAP


def test_lexiroute_handles_1q_gates_and_deps():
    cm = _grid4x4()
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)          # 依赖前一个 cx 的同一比特
    qc.cx(1, 2)
    routed, _, _ = lexiroute(qc, cm, {0: 0, 1: 1, 2: 2}, seed=0)
    assert routed.count_ops().get("cx", 0) == 2
    assert routed.count_ops().get("h", 0) == 2


def test_lexiroute_terminates_on_dense_circuit():
    cm = _grid4x4()
    qc = QuantumCircuit(4)
    for i in range(4):
        for j in range(i + 1, 4):
            qc.cx(i, j)
    routed, swaps, _ = lexiroute(qc, cm, {0: 0, 1: 1, 2: 2, 3: 3}, seed=0)
    assert routed.count_ops().get("cx", 0) == 6
    assert swaps > 0


def test_lexiroute_depth_parallelism():
    """无依赖的门应并行（同层执行），深度应等于并行层数而非串行门数。"""
    cm = _grid4x4()
    qc = QuantumCircuit(8)
    # 四对互不依赖的相邻 cx（逻辑 0-1/2-3/4-5/6-7）
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(4, 5)
    qc.cx(6, 7)
    # 布局到四对相邻物理位置
    layout = {0: 0, 1: 1, 2: 4, 3: 5, 4: 8, 5: 9, 6: 12, 7: 13}
    routed, swaps, _ = lexiroute(qc, cm, layout, seed=0)
    assert swaps == 0
    # 四对互不依赖：应在一个并行层内完成
    assert routed.depth() == 1
