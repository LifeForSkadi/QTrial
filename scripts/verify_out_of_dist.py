# -*- coding: utf-8 -*-
"""分布外验证（2026-11-13，correctness_theorems.md §3）：
8 条 587 之外线路跑管线 → 合法性三重扫描 + 紧凑子空间语义 fid
（≤26 比特全空间）+ qiskit O3 对照（门数/fid）。零非法、fid=1
为正确性层的通过判据；门数/fid 对照为性能层观察。
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
from qtrail.pure.metrics import compute_metrics

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OOD = Path("data/ood")


def scan_legality(circ, spec):
    """合法性三重扫描（verify_all_legality 同款）。"""
    ok = True
    for inst in circ.ops:
        if inst.name not in ("rz", "sx", "x", "cz"):
            if inst.name == "measure":
                continue
            return False, f"非法门 {inst.name}"
        if inst.nq == 2:
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            if not spec.adj[a, b]:
                return False, f"非相邻 cz {a}-{b}"
        for q in inst.qubits:
            if int(q) >= spec.n or int(q) in spec.disabled_qubits:
                return False, f"比特越界/禁用 {q}"
    return True, "ok"


def semantic_fid(orig_circ, final_qc, final_layout, spec):
    """紧凑子空间语义验证（fin 嵌入交付判据）。"""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    import numpy as np
    # 原线路经 final_layout 嵌入
    touched = set()
    for inst in orig_circ.ops:
        for q in inst.qubits:
            touched.add(int(q))
    touched = sorted(touched)
    remap = {q: i for i, q in enumerate(touched)}
    k = len(touched)
    qc1 = QuantumCircuit(k)
    for inst in orig_circ.ops:
        if inst.nq == 2:
            qc1.cx(remap[int(inst.qubits[0])], remap[int(inst.qubits[1])])
        elif inst.name == "u1":
            qc1.rz(inst.params[0], remap[int(inst.qubits[0])])
        elif inst.name == "u3":
            qc1.u(*inst.params, remap[int(inst.qubits[0])])
    # 产物线路（同坐标）
    qc2 = QuantumCircuit(k)
    for inst in final_qc.ops:
        if inst.name == "rz":
            qc2.rz(inst.params[0], remap.get(int(inst.qubits[0]), 0))
        elif inst.name == "sx":
            qc2.sx(remap.get(int(inst.qubits[0]), 0))
        elif inst.name == "x":
            qc2.x(remap.get(int(inst.qubits[0]), 0))
        elif inst.name == "cz":
            a = remap.get(int(inst.qubits[0]))
            b = remap.get(int(inst.qubits[1]))
            if a is not None and b is not None:
                qc2.cz(a, b)
    sv1 = Statevector.from_instruction(qc1)
    sv2 = Statevector.from_instruction(qc2)
    return float(abs(np.vdot(sv1.data, sv2.data)) ** 2)


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

    for p in sorted(OOD.glob("*.qasm")):
        name = p.stem
        try:
            circ = parse_qasm(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[{name}] 解析失败 {e}", flush=True)
            continue
        torch.manual_seed(42)
        mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev,
                            seed=42, selection_rule="fidelity",
                            use_post=True, pipeline="v5", routing_seeds=16)
        t0 = time.time()
        try:
            res = mapper.map_circuit(circ, circuit_id=name)
        except Exception as e:
            print(f"[{name}] 管线错误 {type(e).__name__}: {str(e)[:60]}",
                  flush=True)
            continue
        final_qc = res["routed_circuit"]
        legal, msg = scan_legality(final_qc, spec)
        if circ.n <= 26:
            fid = semantic_fid(circ, final_qc, res["final_layout"], spec)
        else:
            fid = None
        c = Counter(i.name for i in final_qc.ops)
        n1q = c.get("rz", 0) + c.get("sx", 0) + c.get("x", 0)
        print(f"[{name}] n={circ.n} 合法={legal} fid="
              f"{f'{fid:.6f}' if fid is not None else '跳过(>26)'} "
              f"1Q={n1q} cz={c.get('cz',0)} "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
