# -*- coding: utf-8 -*-
"""删 swap 语义验证（2026-11-13）：路由输出的门坐标是否已含
swap 效果（删 swap + 门不动 = 语义保持、产物全相邻）？
小线路 heisenberg N16 上：路由后删 swap → 紧凑子空间 fid 对比
（原线路 fin 嵌入 vs 删 swap 产物）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route
from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.post_v2 import decompose_to_platform_v2

LAYOUT = {0: 74, 1: 80, 2: 75, 3: 82, 4: 68, 5: 59, 6: 73, 7: 60,
          8: 76, 9: 66, 10: 61, 11: 67, 12: 69, 13: 62, 14: 83,
          15: 81, 16: 55, 17: 54, 18: 53, 19: 52}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)

    p = None
    for cand in Path("data/benchpress/qasm").rglob("*.qasm"):
        if "square_heisenberg_N16" in cand.name:
            p = cand
            break
    circ = parse_qasm(p.read_text(encoding="utf-8"))
    # 用 heisenberg 自己的布局（跑管线拿）
    from qtrail.models import QAPolicy
    from qtrail.pure.mapper import PureMapper
    CKPT = ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_"
            "c0.05_best.pt")
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    res = mapper.map_circuit(circ, circuit_id="heis16")
    layout = {i: int(res["layout"][i]) for i in range(circ.n)}

    torch.manual_seed(42)
    routed, swaps, fl = sabre_route(circ, spec, layout, seed=42,
                                    lam_ms=0.0, trials=16, mix_rollout=True)
    print(f"routed swaps={swaps} n={routed.n}", flush=True)

    # 删 swap、门不动 → 非相邻统计
    ops_del = [i for i in routed.ops if i.name != "swap"]
    nonadj = sum(1 for i in ops_del
                 if i.nq == 2 and not spec.adj[int(i.qubits[0]),
                                               int(i.qubits[1])])
    print(f"删 swap 后：门数={len(ops_del)} 非相邻={nonadj}", flush=True)

    # 语义验证：紧凑子空间 fid（原线路 fin 嵌入 vs 删 swap 产物）
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    import numpy as np
    # 触碰比特压缩映射
    touched = set()
    for inst in circ.ops:
        for q in inst.qubits:
            touched.add(int(layout[int(q)]))
    touched |= set(int(q) for i in ops_del for q in i.qubits)
    touched = sorted(touched)
    k = len(touched)
    remap = {q: i for i, q in enumerate(touched)}
    qc1c = QuantumCircuit(k)
    for inst in circ.ops:
        if inst.nq == 2:
            qc1c.cx(remap[int(layout[int(inst.qubits[0])])],
                    remap[int(layout[int(inst.qubits[1])])])
        elif inst.name == "u1":
            qc1c.rz(inst.params[0], remap[int(layout[int(inst.qubits[0])])])
        elif inst.name == "u3":
            qc1c.u(*inst.params, remap[int(layout[int(inst.qubits[0])])])
    qc2c = QuantumCircuit(k)
    for inst in ops_del:
        if inst.name == "cx":
            qc2c.cx(remap[int(inst.qubits[0])], remap[int(inst.qubits[1])])
        elif inst.name == "u1":
            qc2c.rz(inst.params[0], remap[int(inst.qubits[0])])
        elif inst.name == "u3":
            qc2c.u(*inst.params, remap[int(inst.qubits[0])])
    sv1c = Statevector.from_instruction(qc1c)
    sv2c = Statevector.from_instruction(qc2c)
    # fin 重排 sv1c
    fin_t = {}
    for i in range(spec.n):
        fin_t[i] = int(fl.get(i, i))
    # 构造 sv1c 的 fin 置换（k 位子空间的坐标重排）
    perm = [0] * (2 ** k)
    for idx in range(2 ** k):
        # idx 的比特 j（压缩坐标）对应物理 touched[j] → fin 后 →
        # 新压缩坐标
        new_idx = 0
        for j in range(k):
            if (idx >> j) & 1:
                phys = touched[j]
                new_phys = fin_t.get(phys, phys)
                nj = remap[new_phys]
                new_idx |= 1 << nj
        perm[idx] = new_idx
    sv1p = np.array([sv1c.data[perm[idx]] for idx in range(2 ** k)])
    fid = float(abs(np.vdot(sv1p, sv2c.data)) ** 2)
    print(f"删 swap 语义 fid = {fid:.8f}（1.0 = 语义保持）", flush=True)


if __name__ == "__main__":
    main()
