# -*- coding: utf-8 -*-
"""上游 1Q 膨胀链诊断（2026-11-13）：heisenberg N16。

定位 1Q 差距（736 vs qiskit 489）的膨胀阶段：
原始 → RL 布局 → sabre_route → post_route → fold_and_relink →
_decompose_ablation，逐阶段统计 cx/swap/1Q 门数；
qiskit 同口径对照（SabreLayout+SabreSwap+O3）。
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


def stat(ops, label):
    c = Counter(i.name for i in ops)
    n1q = sum(v for k, v in c.items() if k in ("rz", "sx", "x", "u1", "u3"))
    n2q = sum(v for k, v in c.items() if k in ("cx", "cz", "swap"))
    print(f"  [{label}] 总门={len(ops)} 1Q={n1q} 2Q={n2q} {dict(c)}",
          flush=True)
    return n1q, n2q


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
    print(f"原始线路 n={circ.n}", flush=True)
    stat(circ.ops, "原始")

    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    t0 = time.time()
    res = mapper.map_circuit(circ, circuit_id="heis16")
    print(f"[{time.time()-t0:.0f}s] map_circuit（含布局/联合选择）",
          flush=True)
    stat(res["routed_circuit"].ops, "最终产物")

    torch.manual_seed(42)
    layout = {i: int(res["layout"][i]) for i in range(circ.n)}
    routed, swaps, fl = sabre_route(circ, spec, layout, seed=42,
                                    lam_ms=mapper.lam_ms, trials=16,
                                    mix_rollout=(circ.n > 10))
    print(f"路由后 swaps={swaps}", flush=True)
    stat(routed.ops, "sabre_route 后（swap 展开）")

    routed2, removed = post_route(routed, fl, spec.adj)
    print(f"post_route 吸收 {removed}", flush=True)
    stat(routed2.ops, "post_route 后")

    # fold_and_relink（v5 候选 2 的老折叠路径）
    from qtrail.pure.post_v2 import fold_and_relink
    routed3, fin3 = fold_and_relink(routed2, spec)
    stat(routed3.ops, "fold_and_relink 后（decompose 输入）")

    # qiskit 同口径对照
    from qiskit import transpile
    from qtrail.pipeline.baselines import build_target, PLATFORM_BASIS
    from qtrail.pipeline.mapper import Mapper as QiskitMapper
    from qtrail.utils.qasm_io import load_qasm2, strip_measurements
    qc = load_qasm2(p)
    qc_clean, _ = strip_measurements(qc)
    target = build_target(spec)
    qmapper = QiskitMapper(spec, policy=policy, cfg=cfg, dev="cpu",
                           seed=42)
    res_q = qmapper.map_circuit(qc_clean, circuit_id="heis16",
                                optimization_level=1)
    init = {qc_clean.qubits[j]: int(res_q.layout[j])
            for j in range(qc_clean.num_qubits)}
    out = transpile(qc_clean, target=target,
                    basis_gates=PLATFORM_BASIS,
                    optimization_level=3,
                    initial_layout=init,
                    seed_transpiler=42)
    qc1 = Counter(i.name for i in out.data)
    qn1q = sum(v for k, v in qc1.items() if k in ("rz", "sx", "x"))
    qn2q = sum(v for k, v in qc1.items() if k in ("cx", "cz", "swap"))
    print(f"  [qiskit O3] 总门={len(out.data)} 1Q={qn1q} 2Q={qn2q} "
          f"{dict(qc1)}", flush=True)


if __name__ == "__main__":
    main()
