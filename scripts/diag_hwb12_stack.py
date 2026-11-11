# -*- coding: utf-8 -*-
"""hwb12 卡点诊断 v2（2026-11-13）：多时点栈转储 + 分阶段计时
定位超线性瓶颈（hwb11 24.5 万门 896s 能过、hwb12 51.4 万门超时）。
"""
import sys
import time
import faulthandler
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
from qtrail.pure.circuit import Circuit

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"

# 多时点栈转储（覆盖 240/600/1500s——不 exit，跑完看全流程）
faulthandler.dump_traceback_later(240, repeat=True, exit=False)


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

    # 分阶段：布局 → 路由 → post_route → v5 管线 → 总
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    t0 = time.time()
    # 布局（重电路降级路径 >2 万门：贪心图匹配，无 RL）
    from qtrail.pure.layout import greedy_graph_match_layout
    import numpy as np
    rng = np.random.default_rng(42)
    pi = greedy_graph_match_layout(circ, spec, rng)
    layout = {i: int(pi[i]) for i in range(circ.n)}
    print(f"布局 {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    routed, swaps, fl = sabre_route(circ, spec, layout, seed=42,
                                    lam_ms=mapper.lam_ms, trials=16,
                                    mix_rollout=(circ.n > 10))
    print(f"sabre_route {time.time()-t0:.0f}s（swaps={swaps}）",
          flush=True)

    t0 = time.time()
    routed, removed = post_route(routed, fl, spec.adj)
    print(f"post_route {time.time()-t0:.0f}s（吸收 {removed}）",
          flush=True)

    t0 = time.time()
    _, _, final_qc = mapper._v5_pipeline(routed, fl, circ, 0.0)
    print(f"_v5_pipeline {time.time()-t0:.0f}s", flush=True)
    print(f"产物 {len(final_qc.ops)} 门", flush=True)


if __name__ == "__main__":
    main()
