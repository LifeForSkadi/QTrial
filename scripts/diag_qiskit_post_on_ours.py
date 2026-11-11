# -*- coding: utf-8 -*-
"""qiskit 后处理解剖（2026-11-13）：把我们的 fold 后线路（cx 288 +
u3 576 + u1 160 + swap 16）喂 qiskit O3 后处理，看 qiskit 的 pass
能把我们的上游线路压到多少（vs 我们 736 / qiskit 自路由 489）。

若 qiskit 后处理压到 ~489 → 后处理有我们缺的机制（解剖 pass 序列）；
若只到 ~700+ → 差距在 qiskit 路由结构（RL 布局的 2Q 优势 + 1Q 劣势
是路由结构的代价）。
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

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
BP = Path("data/benchpress/qasm")


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
    # fold 折叠（与 _v5_pipeline 候选 2 同路径）
    from qtrail.pure.post_v2 import fold_and_relink
    routed, fin = fold_and_relink(routed, spec)
    print(f"fold 后线路: {Counter(i.name for i in routed.ops)}",
          flush=True)

    # 触碰比特压缩映射（路由器空位穿越坐标 → 0..k-1）
    touched = set()
    for inst in routed.ops:
        for q in inst.qubits:
            touched.add(int(q))
    remap = {q: i for i, q in enumerate(sorted(touched))}
    fin = fin or {}
    phys = {i: int(fin.get(q, q)) for i, q in enumerate(sorted(touched))}
    print(f"触碰 {len(touched)} 位 → 压缩 0..{len(touched)-1}",
          flush=True)

    # 我们的 fold 后线路 → qiskit QASM → transpile O3
    from qiskit import QuantumCircuit, transpile
    from qtrail.pipeline.baselines import build_target, PLATFORM_BASIS
    qc = QuantumCircuit(len(touched))
    for inst in routed.ops:
        if inst.name == "cx":
            a, b = remap[int(inst.qubits[0])], remap[int(inst.qubits[1])]
            qc.cx(a, b)
        elif inst.name == "u1":
            qc.rz(inst.params[0], remap[int(inst.qubits[0])])
        elif inst.name == "u3":
            qc.u(inst.params[0], inst.params[1], inst.params[2],
                 remap[int(inst.qubits[0])])
        elif inst.name == "swap":
            qc.swap(remap[int(inst.qubits[0])],
                    remap[int(inst.qubits[1])])
        elif inst.name == "u2":
            qc.u(inst.params[0], inst.params[1], inst.params[2],
                 remap[int(inst.qubits[0])])
    target = build_target(spec)
    # final_layout 嵌入（折叠坐标 = 最终物理位）
    init = {}
    for i in range(len(touched)):
        init[qc.qubits[i]] = phys[i]
    t0 = time.time()
    out = transpile(qc, target=target, basis_gates=PLATFORM_BASIS,
                    optimization_level=3, initial_layout=init,
                    seed_transpiler=42)
    c = Counter(i.name for i in out.data)
    n1q = sum(v for k, v in c.items() if k in ("rz", "sx", "x"))
    n2q = sum(v for k, v in c.items() if k in ("cz", "cx", "swap"))
    print(f"[{time.time()-t0:.0f}s] qiskit O3 处理我们的 fold 后线路: "
          f"总={len(out.data)} 1Q={n1q} 2Q={n2q} {dict(c)}", flush=True)
    print("对照：我们管线 1Q=736 cz=72；qiskit 自路由 1Q=489 cz=126",
          flush=True)


if __name__ == "__main__":
    main()
