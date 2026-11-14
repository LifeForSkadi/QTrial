"""Routing on top of an RL layout: SabreSwap with a fixed initial layout.

The swap count is measured by a routing-only pass that keeps `swap` in the
basis (CO-MAP's methodology: isolate layout+routing stage), then the routed
circuit is decomposed to the platform basis [rz, sx, x, cz].
"""
from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.transpiler.passes import (ApplyLayout, EnlargeWithAncilla,
                                      FullAncillaAllocation, SabreSwap, SetLayout)

from qtrail.utils.qasm_io import PLATFORM_BASIS

ROUTING_BASIS = ["cx", "u", "rz", "sx", "x", "h", "swap"]


def coupling_map_from_spec(spec) -> CouplingMap:
    edges = [(i, j) for i in range(spec.n) for j in range(spec.n) if spec.adj[i, j]]
    return CouplingMap(edges)


def route_with_layout(qc: QuantumCircuit, cm: CouplingMap, layout: dict,
                      seed: int = 0, preset: bool = True,
                      method: str = "sabre") -> tuple[QuantumCircuit, int, dict]:
    """Route qc with a fixed initial layout.

    method:
      sabre（默认）: 完整 O1 预设管线（基线同款机械，公平对比用）
      lexi: 自研深度感知路由器（LexiRoute 思想的自主实现，深度优先）

    Returns (routed_circuit_with_swaps, swap_count, final_layout) where
    final_layout maps logical -> physical after all swaps.
    """
    qc_clean = qc.copy()
    # barriers can interfere with routing dependency analysis
    for inst in list(qc_clean.data):
        if inst.operation.name == "barrier":
            qc_clean.data.remove(inst)

    if method == "lexi":
        from qtrail.pipeline.lexiroute import lexiroute
        return lexiroute(qc_clean, cm, layout, seed=seed)

    init_layout = Layout({qc_clean.qubits[k]: v for k, v in layout.items()})
    if preset:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(optimization_level=1,
                                          initial_layout=init_layout,
                                          coupling_map=cm,
                                          basis_gates=ROUTING_BASIS,
                                          seed_transpiler=seed)
        routed = pm.run(qc_clean)
    else:
        # manual pass chain (qiskit >= 2 SabreSwap requires a physical circuit)
        pm = PassManager([
            SetLayout(init_layout),
            FullAncillaAllocation(coupling_map=cm),
            EnlargeWithAncilla(),
            ApplyLayout(),
            SabreSwap(coupling_map=cm, heuristic="decay", seed=seed),
        ])
        routed = pm.run(qc_clean)

    swap_count = routed.count_ops().get("swap", 0)

    # final logical->physical mapping: trace the swap sequence in the routed
    # circuit (statevector-verified; routing_permutation() in qiskit 2.5
    # indexes output positions, not physical indices)
    final = {k: v for k, v in layout.items()}
    for inst in routed.data:
        if inst.operation.name == "swap":
            p0 = routed.find_bit(inst.qubits[0]).index
            p1 = routed.find_bit(inst.qubits[1]).index
            for logical, phys in final.items():
                if phys == p0:
                    final[logical] = p1
                elif phys == p1:
                    final[logical] = p0
    return routed, swap_count, final


def decompose_to_platform(qc: QuantumCircuit, cm: CouplingMap,
                          optimization_level: int = 1,
                          seed: int = 0) -> QuantumCircuit:
    """Decompose a routed (physical-qubit, swap-containing) circuit to the
    platform basis [rz, sx, x, cz]. No further routing happens here."""
    return transpile(qc, coupling_map=cm, basis_gates=PLATFORM_BASIS,
                     optimization_level=optimization_level,
                     layout_method="trivial", routing_method="none",
                     seed_transpiler=seed)
