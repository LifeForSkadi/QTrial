# -*- coding: utf-8 -*-
"""A2 消融侦察（2026-11-13）：heisenberg N16 上定位 1Q 膨胀源。
A2 on（resynth_every=1）vs A2 off（with_resynth=False）的 1Q 门数
对比 + 中间产物（A2 前）的段结构。决定攻坚目标：A2 块模板 or 上游。
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

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
BP = Path("data/benchpress/qasm")
PI = 3.141592653589793


def count(ops):
    c = Counter()
    for i in ops:
        c[i.name] += 1
    return c


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
    t0 = time.time()
    res = mapper.map_circuit(circ, circuit_id="heis16")
    ops_on = res["routed_circuit"].ops
    print(f"A2 on: {count(ops_on)}  1Q="
          f"{sum(v for k, v in count(ops_on).items() if k in ('rz','sx','x'))}"
          f"  twoq={count(ops_on).get('cz',0)}  (墙钟 {time.time()-t0:.0f}s)")

    # A2 off：直接复现 v5 pipeline 的 with_resynth=False 路径
    from qtrail.pure.post import decompose_to_platform
    from qtrail.pure.router import sabre_route
    from qtrail.pure.mapper import post_route
    from qtrail.pure.post_v2 import _decompose_ablation
    torch.manual_seed(42)
    t0 = time.time()
    layout = {i: int(res["layout"][i]) for i in range(circ.n)}
    routed, swaps, final_layout = sabre_route(
        circ, spec, layout, seed=42, lam_ms=mapper.lam_ms,
        trials=16, mix_rollout=(circ.n > 10))
    if mapper.use_post:
        routed, removed = post_route(routed, final_layout, spec.adj)
    # v5 无 A2：绕 _v5_pipeline 太深，直接用 _decompose_ablation
    # resynth_every=0（等价 with_resynth=False 的 A2 关闭）
    routed2, final_layout2, final_qc = mapper._v5_pipeline(
        routed, final_layout, circ, 0.0, with_resynth=False)
    ops_off = final_qc.ops
    print(f"A2 off: {count(ops_off)}  1Q="
          f"{sum(v for k, v in count(ops_off).items() if k in ('rz','sx','x'))}"
          f"  twoq={count(ops_off).get('cz',0)}  (墙钟 {time.time()-t0:.0f}s)")

    # 中间产物：A2 前的线路（with_resynth=False 的内部 routed 即
    # fold+relink 后、A2 前——打印段结构）
    n1q = sum(v for k, v in count(ops_off).items() if k in ("rz", "sx", "x"))
    n1q_on = sum(v for k, v in count(ops_on).items()
                 if k in ("rz", "sx", "x"))
    print(f"\nA2 贡献：1Q {n1q} -> {n1q_on}（+{n1q_on-n1q}），"
          f"cz {count(ops_off).get('cz',0)} -> {count(ops_on).get('cz',0)}"
          f"（{count(ops_on).get('cz',0)-count(ops_off).get('cz',0)}）")
    print(f"参考：qiskit O3 = cz 126 / 1Q 489（同口径）")


if __name__ == "__main__":
    main()
