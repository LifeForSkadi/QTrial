# -*- coding: utf-8 -*-
"""H 优先组 v2 判据实验（2026-11-13）：仅 H 组件（滑动+H 对消+
H·rz·H→rx，无 commute）在 merge 之前。预注册判据：3 线路 bar
+≥5% 门数减少。基线（h_first=False 生产）：N16=1024、N25=1684、
qft=3522（bar_h_first.out 同口径）。
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
from qtrail.pure.post_v2 import _decompose_ablation

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
BP = Path("data/benchpress/qasm")
TARGETS = ["square_heisenberg_N16", "square_heisenberg_N25", "qft_N020"]
BASELINE = {"square_heisenberg_N16": 1024,
            "square_heisenberg_N25": 1684,
            "qft_N020": 3522}


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
        out = _decompose_ablation(cr, spec=spec, resynth_every=1,
                                       h_first=True)
        c_n, n_n = count(out.ops)
        base = BASELINE[tgt]
        print(f"[{tgt}] 基线={base} h_first_v2={n_n}"
              f"（{time.time()-t0:.0f}s） 比值={n_n/base:.3f}"
              f" → {'达标' if n_n <= base*0.95 else '未达标'}", flush=True)
        print(f"  h_first_v2 {dict(c_n)}", flush=True)


if __name__ == "__main__":
    main()
