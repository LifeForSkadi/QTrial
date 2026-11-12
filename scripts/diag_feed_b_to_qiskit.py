# -*- coding: utf-8 -*-
"""决定性实验（2026-11-12）：B（decompose_linear）的输出再喂 qiskit O3
后处理——若 qiskit 能继续压缩 → 我们的 pass 缺机制；压不动 → 差距在
翻译/分解阶段（u3 分解候选选择）。
"""
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.post_v2 import decompose_linear


def main():
    with open("scripts/tmp_decompose_ab/dump_qv32.pkl", "rb") as f:
        dump = pickle.load(f)
    spec = build_tianyan287_spec(load_device_config())
    out = decompose_linear(dump, spec=spec)
    n1q = sum(1 for i in out.ops if i.nq == 1)
    print(f"B 输出: 1Q={n1q} cz={out.count('cz')}", flush=True)

    from qiskit import QuantumCircuit, transpile
    from qtrail.pipeline.baselines import build_target
    touched = sorted({int(q) for i in out.ops for q in i.qubits})
    remap = {q: i for i, q in enumerate(touched)}
    qc = QuantumCircuit(len(touched))
    for inst in out.ops:
        if inst.name == "cz":
            qc.cz(remap[int(inst.qubits[0])], remap[int(inst.qubits[1])])
        elif inst.name == "rz":
            qc.rz(inst.params[0], remap[int(inst.qubits[0])])
        elif inst.name == "sx":
            qc.sx(remap[int(inst.qubits[0])])
        elif inst.name == "x":
            qc.x(remap[int(inst.qubits[0])])
    target = build_target(spec)
    init = {qc.qubits[i]: int(touched[i]) for i in range(len(touched))}
    out2 = transpile(qc, target=target, basis_gates=["rz", "sx", "x", "cz"],
                     optimization_level=3, initial_layout=init,
                     seed_transpiler=42)
    c = Counter(i.operation.name for i in out2.data)
    n1q2 = sum(v for k, v in c.items() if k in ("rz", "sx", "x"))
    print(f"qiskit O3 再处理: 1Q={n1q2} 2Q={c.get('cz', 0)} {dict(c)}",
          flush=True)
    # 对照：原始 dump 直接 qiskit
    touched = sorted({int(q) for i in dump.ops for q in i.qubits})
    remap = {q: i for i, q in enumerate(touched)}
    qc = QuantumCircuit(len(touched))
    for inst in dump.ops:
        if inst.name == "cx":
            qc.cx(remap[int(inst.qubits[0])], remap[int(inst.qubits[1])])
        elif inst.name == "u1":
            qc.rz(inst.params[0], remap[int(inst.qubits[0])])
        elif inst.name == "u3":
            qc.u(*inst.params, remap[int(inst.qubits[0])])
        elif inst.name == "swap":
            qc.swap(remap[int(inst.qubits[0])],
                    remap[int(inst.qubits[1])])
    init = {qc.qubits[i]: int(touched[i]) for i in range(len(touched))}
    out3 = transpile(qc, target=target, basis_gates=["rz", "sx", "x", "cz"],
                     optimization_level=3, initial_layout=init,
                     seed_transpiler=42)
    c3 = Counter(i.operation.name for i in out3.data)
    n1q3 = sum(v for k, v in c3.items() if k in ("rz", "sx", "x"))
    print(f"原始 dump 直接 qiskit: 1Q={n1q3} 2Q={c3.get('cz', 0)} "
          f"{dict(c3)}", flush=True)


if __name__ == "__main__":
    main()
