"""Baseline compilers: Qiskit SABRE (O0-O3), trivial-layout routing."""
from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qtrail.pipeline.routing import ROUTING_BASIS
from qtrail.utils.qasm_io import PLATFORM_BASIS


def sabre_swap_count(qc: QuantumCircuit, cm: CouplingMap,
                     optimization_level: int = 1, seed: int = 0) -> tuple[int, QuantumCircuit]:
    """Run full SABRE (layout+routing) with swap kept in basis; count swaps."""
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level, coupling_map=cm,
        basis_gates=ROUTING_BASIS, seed_transpiler=seed)
    routed = pm.run(qc)
    return routed.count_ops().get("swap", 0), routed


def sabre_transpile(qc: QuantumCircuit, cm: CouplingMap,
                    optimization_level: int = 1, seed: int = 0) -> QuantumCircuit:
    """Standard SABRE transpilation to the platform basis."""
    return transpile(qc, coupling_map=cm, basis_gates=PLATFORM_BASIS,
                     optimization_level=optimization_level,
                     seed_transpiler=seed)


def trivial_route(qc: QuantumCircuit, cm: CouplingMap, seed: int = 0):
    """Identity layout + SabreSwap (sanity floor)."""
    pm = PassManager([SabreSwap(coupling_map=cm, heuristic="decay", seed=seed)])
    routed = pm.run(qc)
    return routed.count_ops().get("swap", 0), routed


def cqlib_baseline(qasm_path: str, machine: str = "tianyan-287"):
    """Cqlib's own transpile (guarded: needs token / platform). Returns None
    on any failure so the caller can skip gracefully."""
    try:
        from cqlib import TianYanPlatform
        from cqlib.mapping import mapping as cmap
        platform = TianYanPlatform(login_key="", auto_login=False,
                                   machine_name=machine)
        # without a token this path cannot download config; fail gracefully
        try:
            platform.download_config(machine=machine)
        except Exception:
            return None
        with open(qasm_path, encoding="utf-8") as f:
            qasm_str = f.read()
        from cqlib.utils.qasm_to_qcis import QasmToQcis
        qcis = QasmToQcis().qasm_to_qcis(qasm_str)
        transpiled = cmap.transpile_qcis(qcis, platform)
        return transpiled
    except Exception:
        return None
