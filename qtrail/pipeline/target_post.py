"""Target 后处理管线（opt-in）：QTrial RL 布局 + qiskit O1 预设完整管线
（噪声感知 Target 驱动）。

定位声明：与 SabreSwap 同级的"公开算法作子程序"——布局由 QTrial 提供，
路由/酉综合/重标记采用 qiskit O1 预设管线（噪声感知 Target 打分）。
实测：同一 RL 布局下 SWAP 92→0（QUEKO dense，O1 管线内的酉综合/优化
阶段吸收置换）。默认关闭——原研究管线（route_with_layout）保持不变。

机制说明（实测）：预设管线在给定 initial_layout 时跳过 VF2PostLayout
（get_vf2_limits 返回 None），SWAP 消除来自 Target 驱动的路由与优化阶段。
"""
from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.transpiler import Layout, Target, InstructionProperties

from qtrail.devices.spec import DeviceSpec
from qtrail.pipeline.routing import ROUTING_BASIS


def build_noise_target(spec: DeviceSpec) -> Target:
    """从 DeviceSpec 校准数据构造噪声感知 Target（与评测基线同款）。"""
    calib = spec.calib
    t = Target(num_qubits=spec.n)
    t.add_instruction(RZGate(0.0), properties={
        (q,): InstructionProperties(error=float(calib.err_1q[q]), duration=1.0)
        for q in range(spec.n)})
    t.add_instruction(CZGate(), properties={
        (a, b): InstructionProperties(error=float(e), duration=1.0)
        for (a, b), e in calib.err_2q.items() for a, b in ((a, b), (b, a))})
    t.add_instruction(XGate(), properties={(q,): None for q in range(spec.n)})
    t.add_instruction(SXGate(), properties={(q,): None for q in range(spec.n)})
    return t


def route_target_post(qc: QuantumCircuit, spec: DeviceSpec, layout: dict,
                      seed: int = 0, optimization_level: int = 1,
                      basis: list | None = None) -> QuantumCircuit:
    """QTrial 布局 + qiskit O1 预设（Target 驱动）→ 路由后处理后的线路。

    返回仍含 swap 门的线路（basis 默认 ROUTING_BASIS），swap 计数与
    最终布局由调用方按既有口径统计。
    """
    target = build_noise_target(spec)
    init = Layout({qc.qubits[k]: v for k, v in layout.items()})
    return transpile(qc, target=target,
                     basis_gates=basis or ROUTING_BASIS,
                     optimization_level=optimization_level,
                     initial_layout=init,
                     seed_transpiler=seed)
