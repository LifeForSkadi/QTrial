# -*- coding: utf-8 -*-
"""零角省略 A/B（2026-11-13）：_u3_to_rzsx_min 的零角省略是否阻断
rz 跨段合并。基线（strip 生产）已有：N16=1024、N25=1684、qft=3522
（bar_h_first.out 同口径）。本脚本跑 no_strip 三线路对比。
预注册判据：+≥5% 门数减少。
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
from qtrail.pure.circuit import Circuit

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
BP = Path("data/benchpress/qasm")
TARGETS = ["square_heisenberg_N16", "square_heisenberg_N25", "qft_N020"]
BASELINE = {"square_heisenberg_N16": 1024,
            "square_heisenberg_N25": 1684,
            "qft_N020": 3522}


def patch_no_strip():
    """替换 _u3_to_rzsx_min：删除零角省略候选。"""
    import math
    from qtrail.pure.post import _u3_to_rzsx
    PI = 3.141592653589793

    def _no_strip(th, ph, la):
        def _mod2pi(a):
            return float(a % (2 * PI))
        cands = [_u3_to_rzsx(th, ph, la)]
        if abs(math.sin(th / 2)) < 1e-9:  # θ≈0 对角特例（精确，非省略）
            cands.append([("rz", (_mod2pi(ph + la),))])
        return min(cands, key=lambda g: (len(g), 1 if any(
            x[0] == "x" for x in g) else 0))

    import qtrail.pure.post as post
    post._u3_to_rzsx_min = _no_strip
    import qtrail.pure.post_v2 as pv2
    pv2._u3_to_rzsx_min = _no_strip


def count(ops):
    c = Counter(i.name for i in ops)
    return c, c["rz"] + c["sx"] + c.get("x", 0)


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
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    patch_no_strip()

    from qtrail.pure.post_v2 import decompose_to_platform_v2
    for tgt in TARGETS:
        p = None
        for cand in BP.rglob("*.qasm"):
            if tgt in cand.name:
                p = cand
                break
        circ = parse_qasm(p.read_text(encoding="utf-8"))
        torch.manual_seed(42)
        res = mapper.map_circuit(circ, circuit_id=tgt)
        torch.manual_seed(42)
        layout = {i: int(res["layout"][i]) for i in range(circ.n)}
        routed, swaps, fl = sabre_route(circ, spec, layout, seed=42,
                                        lam_ms=mapper.lam_ms, trials=16,
                                        mix_rollout=(circ.n > 10))
        routed, removed = post_route(routed, fl, spec.adj)
        cr = Circuit(routed.n, name=tgt)
        cr.ops = list(routed.ops)
        t0 = time.time()
        out = decompose_to_platform_v2(cr, spec=spec, resynth_every=1)
        c_n, n_n = count(out.ops)
        base = BASELINE[tgt]
        print(f"[{tgt}] 基线(strip)={base} no_strip={n_n}"
              f"（{time.time()-t0:.0f}s） 比值={n_n/base:.3f}"
              f" → {'达标' if n_n <= base*0.95 else '未达标'}", flush=True)
        print(f"  no_strip {dict(c_n)}", flush=True)


if __name__ == "__main__":
    main()
