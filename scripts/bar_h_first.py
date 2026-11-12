# -*- coding: utf-8 -*-
"""H 优先消解判据实验（2026-11-13，预注册 bar）：同输入 routed、
同路径 _decompose_ablation（resynth_every=1），仅 h_first
循环顺序差异。3 线路：square_heisenberg_N16/N25、qft_N020。
判据：h_first 门数 ≤ 基线的 95%（−5%）。
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
TARGETS = ["square_heisenberg_N16", "square_heisenberg_N25",
           "qft_N020"]


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
        routed, swaps, fl = sabre_route(
            circ, spec, layout, seed=42, lam_ms=mapper.lam_ms,
            trials=16, mix_rollout=(circ.n > 10))
        routed, removed = post_route(routed, fl, spec.adj)
        # 同输入 routed 的 A/B
        circ_r = Circuit(routed.n, name=tgt)
        circ_r.ops = list(routed.ops)
        t0 = time.time()
        base = _decompose_ablation(circ_r, spec=spec, resynth_every=1)
        tb = time.time() - t0
        c_b, n1q_b = count(base.ops)
        t0 = time.time()
        new = _decompose_ablation(circ_r, spec=spec, resynth_every=1,
                                       h_first=True)
        tn = time.time() - t0
        c_n, n1q_n = count(new.ops)
        ratio = n1q_n / n1q_b if n1q_b else 1.0
        verdict = "达标" if ratio <= 0.95 else "未达标"
        print(f"[{tgt}] 基线 1Q={n1q_b}（{tb:.0f}s） "
              f"h_first 1Q={n1q_n}（{tn:.0f}s） 比值={ratio:.3f} "
              f"→ {verdict}", flush=True)
        print(f"  基线 {dict(c_b)}", flush=True)
        print(f"  新   {dict(c_n)}", flush=True)


if __name__ == "__main__":
    main()
