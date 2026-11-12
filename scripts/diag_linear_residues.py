# -*- coding: utf-8 -*-
"""qv32 linear 输出残块分析：len-2 游程的构成与 H 半残块数，
定位剥壳不完全的阻塞模式（与 qiskit Q 输出对照）。
"""
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.post_v2 import decompose_linear

HALF = np.pi / 2


def analyze(ops, label):
    comp = Counter(i.name for i in ops)
    n1q = sum(v for k, v in comp.items() if k in ("rz", "sx", "x"))
    print(f"[{label}] 总={len(ops)} 1Q={n1q} 构成={dict(comp)}")
    # 段化逐比特
    runs = []
    i, n = 0, len(ops)
    while i < n:
        if ops[i].nq == 1:
            j = i
            by_q = {}
            while j < n and ops[j].nq == 1:
                by_q.setdefault(ops[j].qubits[0], []).append(ops[j])
                j += 1
            for q, gs in by_q.items():
                runs.append(gs)
            i = j
        else:
            i += 1
    len_dist = Counter(len(r) for r in runs)
    half_l = 0    # [sx, rz(π/2)]
    half_r = 0    # [rz(π/2), sx]
    full_h = 0
    for r in runs:
        if len(r) == 3 and r[0].name == "rz" and r[1].name == "sx" \
                and r[2].name == "rz" and abs(r[0].params[0] - HALF) < 1e-6 \
                and abs(r[2].params[0] - HALF) < 1e-6:
            full_h += 1
        elif len(r) == 2:
            if r[0].name == "sx" and r[1].name == "rz" \
                    and abs(r[1].params[0] - HALF) < 1e-6:
                half_l += 1
            elif r[0].name == "rz" and r[1].name == "sx" \
                    and abs(r[0].params[0] - HALF) < 1e-6:
                half_r += 1
    comp2 = Counter(tuple(g.name for g in r) for r in runs if len(r) <= 3)
    print(f"  游程分布={dict(sorted(len_dist.items()))} 全H残={full_h} "
          f"半残[sx,rz]={half_l} 半残[rz,sx]={half_r}")
    print(f"  短游程构成={dict(sorted(comp2.items()))}")


def main():
    with open("scripts/tmp_decompose_ab/dump_qv32.pkl", "rb") as f:
        dump = pickle.load(f)
    spec = build_tianyan287_spec(load_device_config())
    out = decompose_linear(dump, spec=spec)
    analyze(out.ops, "B linear")
    # 对照 qiskit Q（复用 diag 的 transpile 路径）
    from qiskit import QuantumCircuit, transpile
    from qtrail.pipeline.baselines import build_target
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
    target = build_target(spec)
    init = {qc.qubits[i]: int(touched[i]) for i in range(len(touched))}
    out = transpile(qc, target=target, basis_gates=["rz", "sx", "x", "cz"],
                    optimization_level=3, initial_layout=init,
                    seed_transpiler=42)
    ops = []
    from qtrail.pure.circuit import Inst
    for i in out.data:
        if i.operation.name == "measure":
            continue
        qs = tuple(out.find_bit(q).index for q in i.qubits)
        ops.append(Inst(i.operation.name, qs,
                        tuple(float(p) for p in i.operation.params)))
    analyze(ops, "Q qiskit")


if __name__ == "__main__":
    main()
