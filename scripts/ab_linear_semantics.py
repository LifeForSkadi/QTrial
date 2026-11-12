# -*- coding: utf-8 -*-
"""heis16 生产语义锚点 A/B（2026-11-12）：fixpoint 与 linear 双路径
端到端语义（紧凑子空间法——产物触碰比特压缩映射 + final_layout
嵌入原线路，k≤20 精确）+ 真机合法性。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.circuit import Inst
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
CASE = ("h16", "data/benchpress/qasm/square-heisenberg/"
        "square_heisenberg_N16.qasm")


def to_qc(ops, n):
    qc = QuantumCircuit(n)
    for inst in ops:
        qs = [int(q) for q in inst.qubits]
        if inst.name == 'rz':
            qc.rz(inst.params[0], qs[0])
        elif inst.name == 'sx':
            qc.sx(qs[0])
        elif inst.name == 'x':
            qc.x(qs[0])
        elif inst.name == 'cz':
            qc.cz(qs[0], qs[1])
        elif inst.name == 'cx':
            qc.cx(qs[0], qs[1])
        elif inst.name == 'u3':
            qc.u(*inst.params, qs[0])
        elif inst.name == 'u1':
            qc.p(inst.params[0], qs[0])
    return qc


def run(linear):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    tag, path = CASE
    circ = parse_qasm(Path(path).read_text(encoding="utf-8"))
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16,
                        linear_decompose=linear)
    t0 = time.time()
    res = mapper.map_circuit(circ, circuit_id=tag)
    final_qc = res["routed_circuit"]
    tag2 = "linear" if linear else "fixpt "
    print(f"[{tag2}] fid={res['metrics']['est_fidelity']:.4e} "
          f"wall={time.time()-t0:.0f}s", flush=True)

    # 真机合法性（门集/相邻/禁用/swap 残留）
    probs = []
    disabled = set(spec.disabled_qubits)
    for inst in final_qc.ops:
        qs = [int(q) for q in inst.qubits]
        if inst.name == "swap":
            probs.append(f"swap{qs}")
        if inst.name not in {"rz", "sx", "x", "cz"}:
            probs.append(f"门集{inst.name}")
        if any(q >= spec.n or q in disabled for q in qs):
            probs.append(f"比特{qs}")
        if inst.name == "cz" and not spec.adj[qs[0], qs[1]]:
            probs.append(f"相邻{qs}")
    print(f"[{tag2}] 合法性: {'通过' if not probs else '失败 ' + str(probs[:3])}",
          flush=True)

    # 紧凑子空间语义
    fin = res["final_layout"]
    qubits = sorted({int(q) for i in final_qc.ops for q in i.qubits})
    all_q = sorted(set(qubits) | {int(fin.get(i, i))
                                  for i in range(circ.n)})
    cmap = {q: kk for kk, q in enumerate(all_q)}
    emb = [Inst(i.name, tuple(cmap[int(fin[int(q)])] for q in i.qubits),
                i.params) for i in circ.ops]
    fin2 = [Inst(i.name, tuple(cmap[int(q)] for q in i.qubits),
                 i.params) for i in final_qc.ops]
    k = len(all_q)
    sv1 = np.array(Statevector(to_qc(emb, k)).data)
    sv2 = np.array(Statevector(to_qc(fin2, k)).data)
    fid = abs(np.vdot(sv1, sv2)) ** 2
    print(f"[{tag2}] 端到端语义（紧凑子空间 k={k}）fid={fid:.6f}（应=1）",
          flush=True)
    return fid, probs


def main():
    fa, pa = run(linear=False)
    fb, pb = run(linear=True)
    ok = (fa > 1 - 1e-9 and fb > 1 - 1e-9 and not pa and not pb)
    print(f"\n语义锚点裁决：fixpt={fa:.6f} linear={fb:.6f} → "
          f"{'达标' if ok else '不达标'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
