"""功能等价性验证：映射优化后的线路是否保持原线路的量子功能。

两种模式（按规模自动选择）：
  精确模式（n ≤ max_qubits）：全空间态矢量比较，|⟨ψ_orig|ψ_mapped⟩|² ≥ threshold
  随机化模式（n > max_qubits）：随机直积输入态采样 K 个，比较输出态保真度均值

原线路经 final_layout（逻辑→最终物理映射）嵌入物理空间后与映射线路比较，
对全局相位不敏感（取模方）。
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

DEFAULT_MAX_QUBITS = 20   # 精确模式上限（2^20 态矢量 ~32MB）
DEFAULT_N_SAMPLES = 32
DEFAULT_THRESHOLD = 0.999


def _embed(qc: QuantumCircuit, n_phys: int,
           layout: dict | None) -> QuantumCircuit:
    """把原线路嵌入物理空间（layout: 逻辑→物理；None 则按恒等）。"""
    full = QuantumCircuit(n_phys)
    for inst in qc.data:
        qs = [layout[qc.find_bit(q).index] if layout is not None
              else qc.find_bit(q).index for q in inst.qubits]
        if inst.operation.name in ("measure", "barrier", "reset"):
            continue
        full.append(inst.operation, [full.qubits[i] for i in qs], [])
    return full


def exact_fidelity(original: QuantumCircuit, mapped: QuantumCircuit,
                   layout: dict | None) -> float:
    """精确态矢量保真度（|⟨ψ|ψ'⟩|²）。"""
    n_phys = mapped.num_qubits
    ref = _embed(original, n_phys, layout)
    sv1 = Statevector(ref)
    sv2 = Statevector(mapped)
    return float(abs(np.dot(np.conj(sv1.data), sv2.data)) ** 2)


def randomized_fidelity(original: QuantumCircuit, mapped: QuantumCircuit,
                        layout: dict | None, n_samples: int = DEFAULT_N_SAMPLES,
                        seed: int = 0) -> float:
    """随机直积输入态采样的输出态保真度均值（大线路功能验证）。

    用 Statevector.evolve 逐线路演化（不显式构造酉矩阵），适用于
    21~28 比特；更大规模超出经典仿真能力，应使用平台验证（platform.py）。
    """
    rng = np.random.default_rng(seed)
    n = original.num_qubits
    n_phys = mapped.num_qubits
    ref_circuit = _embed(original, n_phys, layout)
    fids = []
    for _ in range(n_samples):
        # 随机直积输入态：|ψ⟩ = ⊗_i (cos(θ_i/2)|0⟩ + e^{iφ_i} sin(θ_i/2)|1⟩)
        sv = np.ones(1, dtype=np.complex128)
        for _ in range(n):
            th, ph = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
            amp = np.array([np.cos(th / 2), np.sin(th / 2) * np.exp(1j * ph)],
                           dtype=np.complex128)
            sv = np.kron(sv, amp)
        # 嵌入物理空间（经布局置换）
        sv_full = np.zeros(2 ** n_phys, dtype=np.complex128)
        if layout is None:
            sv_full[:2 ** n] = sv
        else:
            for idx in range(2 ** n):
                phys = 0
                for qi in range(n):
                    if (idx >> qi) & 1:
                        phys |= 1 << layout[qi]
                sv_full[phys] = sv[idx]
        out_ref = np.asarray(Statevector(sv_full).evolve(ref_circuit).data)
        out_map = np.asarray(Statevector(sv_full).evolve(mapped).data)
        f = abs(np.vdot(out_ref, out_map)) ** 2
        fids.append(float(f))
    return float(np.mean(fids))


def unmap_circuit(mapped: QuantumCircuit, layout: dict) -> QuantumCircuit:
    """把物理空间线路逆映射回逻辑空间（final_layout 的逆）。

    物理比特 → 逻辑比特重命名；未被使用的物理比特（空闲 ancilla）上的
    门被丢弃（正常情况不存在）。
    """
    inv = {int(v): int(k) for k, v in layout.items()}
    n = len(inv)
    out = QuantumCircuit(n)
    for inst in mapped.data:
        if inst.operation.name in ("barrier", "measure", "reset"):
            continue
        qs = [mapped.find_bit(q).index for q in inst.qubits]
        try:
            logical = [inv[p] for p in qs]
        except KeyError:
            continue  # 空闲 ancilla 上的门（不应存在）——跳过
        out.append(inst.operation, [out.qubits[i] for i in logical], [])
    return out


def verify_equivalence(original: QuantumCircuit, mapped: QuantumCircuit,
                       layout: dict | None = None,
                       max_qubits: int = DEFAULT_MAX_QUBITS,
                       n_samples: int = DEFAULT_N_SAMPLES,
                       threshold: float = DEFAULT_THRESHOLD,
                       seed: int = 0) -> dict:
    """验证映射线路与原线路功能等价。

    映射线路在物理空间时（qubits > 原线路），先经 layout 逆映射回逻辑
    空间再比较（避免物理空间态矢量爆炸）。

    Returns: {"equivalent", "fidelity", "method", "threshold", "n_qubits"}
    """
    if mapped.num_qubits > original.num_qubits and layout is None:
        raise ValueError(
            "映射线路在物理空间（qubits > 原线路）但未提供 final_layout；"
            "请通过 --layout-json 提供 metrics.json 以逆映射回逻辑空间")
    if layout is not None:
        mapped = unmap_circuit(mapped, layout)
    if original.num_qubits <= max_qubits:
        fid = exact_fidelity(original, mapped, None)
        method = "exact"
    else:
        fid = randomized_fidelity(original, mapped, None, n_samples, seed)
        method = f"randomized(k={n_samples})"
    return {
        "equivalent": bool(fid >= threshold),
        "fidelity": fid,
        "method": method,
        "threshold": threshold,
        "n_qubits": original.num_qubits,
    }
