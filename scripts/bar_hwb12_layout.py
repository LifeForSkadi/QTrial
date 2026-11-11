# -*- coding: utf-8 -*-
"""hwb12 布局对照实验（2026-11-13）：qiskit SabreLayout 布局 vs
我们的贪心图匹配（降级路径）注入同一路由器，对比 swap 数。
假设检验：布局目标/算法差异是 10 万 swap 的主因（可实现的修复 =
巨型线路降级路径换成 SabreLayout 同款）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"


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
    for cand in Path("data/benchpress/qasm").rglob("*.qasm"):
        if "hwb12" in cand.name:
            p = cand
            break
    circ = parse_qasm(p.read_text(encoding="utf-8", errors="replace"))
    print(f"hwb12 n={circ.n} ops={len(circ.ops)}", flush=True)

    # qiskit 布局：transpile O3 的 final_layout（b_o3 实测 0 swap）
    from qiskit import transpile
    from qtrail.pipeline.baselines import build_target
    from qtrail.utils.qasm_io import load_qasm2, strip_measurements
    target = build_target(spec)
    qc = load_qasm2(p)
    qc_clean, _ = strip_measurements(qc)
    t0 = time.time()
    out = transpile(qc_clean, target=target,
                    basis_gates=["rz", "sx", "x", "cz"],
                    optimization_level=3, seed_transpiler=42)
    n_sw = out.count_ops().get("swap", 0)
    # qiskit 2.x 标准 API：initial_index_layout（虚拟 → 物理）
    il = out.layout.initial_index_layout(filter_ancillas=True)
    if isinstance(il, dict):
        layout_q = {int(k): int(v) for k, v in il.items()
                    if int(k) < qc_clean.num_qubits}
    else:
        layout_q = {j: int(il[j]) for j in range(qc_clean.num_qubits)}
    print(f"transpile O3 {time.time()-t0:.0f}s → 产物 swap={n_sw}",
          flush=True)
    print(f"qiskit 布局（虚拟→物理）: {layout_q}", flush=True)

    # 我们的路由器用 qiskit 布局（layout_q 键已是 int 虚拟下标）
    layout = {i: int(layout_q.get(i, layout_q.get(str(i), i)))
              for i in range(circ.n)}
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    t0 = time.time()
    routed, swaps, fl = sabre_route(circ, spec, layout, seed=42,
                                    lam_ms=mapper.lam_ms, trials=16,
                                    mix_rollout=(circ.n > 10))
    print(f"我们的路由器 + qiskit 布局：swaps={swaps}"
          f"（{time.time()-t0:.0f}s）", flush=True)
    print("对照：贪心降级布局 = 100001 swap（诊断 v2 实测）", flush=True)


if __name__ == "__main__":
    main()
