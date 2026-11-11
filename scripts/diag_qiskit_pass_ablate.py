# -*- coding: utf-8 -*-
"""qiskit O3 pass 解剖（2026-11-13）：fold 后线路上逐个移除/子集化
O3 的 pass，定位 1Q 差距（我们 736 vs qiskit 482）的贡献主体。

候选解剖：
  A. 完整 O3（482 基准）
  B. 仅 1Q 核心（Optimize1qGatesDecomposition + CommutationAnalysis
     + CommutativeCancellation + Optimize1qGates）
  C. B 去掉 CommutativeCancellation
  D. B 去掉 CommutationAnalysis
  E. 仅 Optimize1qGates（合成+分解，无通勤）
"""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route
from qtrail.pure.mapper import post_route
from qtrail.pure.post_v2 import fold_and_relink

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
BP = Path("data/benchpress/qasm")


def to_qiskit(routed):
    from qiskit import QuantumCircuit
    touched = set()
    for inst in routed.ops:
        for q in inst.qubits:
            touched.add(int(q))
    remap = {q: i for i, q in enumerate(sorted(touched))}
    qc = QuantumCircuit(len(touched))
    for inst in routed.ops:
        if inst.name == "cx":
            qc.cx(remap[int(inst.qubits[0])], remap[int(inst.qubits[1])])
        elif inst.name == "u1":
            qc.rz(inst.params[0], remap[int(inst.qubits[0])])
        elif inst.name == "u3":
            qc.u(inst.params[0], inst.params[1], inst.params[2],
                 remap[int(inst.qubits[0])])
        elif inst.name == "swap":
            qc.swap(remap[int(inst.qubits[0])], remap[int(inst.qubits[1])])
    return qc, touched


def stat(out, label):
    c = Counter(i.name for i in out.data)
    n1q = sum(v for k, v in c.items() if k in ("rz", "sx", "x"))
    n2q = sum(v for k, v in c.items() if k in ("cz", "cx", "swap"))
    print(f"  [{label}] 1Q={n1q} 2Q={n2q} {dict(c)}", flush=True)
    return n1q


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    p = None
    for cand in BP.rglob("*.qasm"):
        if "square_heisenberg_N16" in cand.name:
            p = cand
            break
    circ = parse_qasm(p.read_text(encoding="utf-8"))

    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    res = mapper.map_circuit(circ, circuit_id="heis16")
    torch.manual_seed(42)
    layout = {i: int(res["layout"][i]) for i in range(circ.n)}
    routed, swaps, fl = sabre_route(circ, spec, layout, seed=42,
                                    lam_ms=mapper.lam_ms, trials=16,
                                    mix_rollout=(circ.n > 10))
    routed, removed = post_route(routed, fl, spec.adj)
    routed, fin = fold_and_relink(routed, spec)
    qc, touched = to_qiskit(routed)

    from qiskit import transpile
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes.optimization import (
        Optimize1qGatesDecomposition, Optimize1qGates,
        CommutationAnalysis, CommutativeCancellation)
    from qtrail.pipeline.baselines import build_target, PLATFORM_BASIS
    target = build_target(spec)
    fin = fin or {}
    phys = {i: int(fin.get(q, q)) for i, q in enumerate(sorted(touched))}
    init = {qc.qubits[i]: phys[i] for i in range(len(touched))}
    basis = PLATFORM_BASIS

    # A. 完整 O3
    t0 = time.time()
    out = transpile(qc, target=target, basis_gates=basis,
                    optimization_level=3, initial_layout=init,
                    seed_transpiler=42)
    stat(out, f"O3 完整（{time.time()-t0:.1f}s）")

    # E. 仅 Optimize1qGates（u 族合成）+ 平台分解
    pm = PassManager([
        Optimize1qGates(target=target),
        Optimize1qGatesDecomposition(basis=basis, target=target),
    ])
    t0 = time.time()
    out = pm.run(qc)
    stat(out, f"E 仅 O1qGates+Decomp（{time.time()-t0:.1f}s）")

    # B. 1Q 核心（+通勤分析+对消）
    pm = PassManager([
        Optimize1qGatesDecomposition(basis=basis, target=target),
        CommutationAnalysis(),
        CommutativeCancellation(),
        Optimize1qGates(target=target),
        Optimize1qGatesDecomposition(basis=basis, target=target),
    ])
    t0 = time.time()
    out = pm.run(qc)
    stat(out, f"B 1Q 核心+Comm（{time.time()-t0:.1f}s）")

    # D. B 去掉 CommutationAnalysis
    pm = PassManager([
        Optimize1qGatesDecomposition(basis=basis, target=target),
        CommutativeCancellation(),
        Optimize1qGates(target=target),
        Optimize1qGatesDecomposition(basis=basis, target=target),
    ])
    t0 = time.time()
    out = pm.run(qc)
    stat(out, f"D 无 CommAnalysis（{time.time()-t0:.1f}s）")

    # C. B 去掉 CommutativeCancellation
    pm = PassManager([
        Optimize1qGatesDecomposition(basis=basis, target=target),
        CommutationAnalysis(),
        Optimize1qGates(target=target),
        Optimize1qGatesDecomposition(basis=basis, target=target),
    ])
    t0 = time.time()
    out = pm.run(qc)
    stat(out, f"C 无 CommCancel（{time.time()-t0:.1f}s）")


if __name__ == "__main__":
    main()
