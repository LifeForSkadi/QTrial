# -*- coding: utf-8 -*-
"""fold_keep 候选选择诊断（2026-11-13）：qft_N020 上打印各候选
评分与选中者——修复版 qft 变差（fid 0.0585→0.00115）的根因。
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
from qtrail.pure.mapper import post_route
from qtrail.pure.post_v2 import (fold_keep_coords, fold_and_relink,
                                 fold_only, _decompose_ablation)
from qtrail.pure.post import decompose_to_platform
from qtrail.pure.metrics import compute_metrics

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
        if "qft_N020" in cand.name:
            p = cand
            break
    circ = parse_qasm(p.read_text(encoding="utf-8"))

    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    res = mapper.map_circuit(circ, circuit_id="qft20")
    torch.manual_seed(42)
    layout = {i: int(res["layout"][i]) for i in range(circ.n)}
    routed, swaps, fl = sabre_route(circ, spec, layout, seed=42,
                                    lam_ms=mapper.lam_ms, trials=16,
                                    mix_rollout=(circ.n > 10))
    print(f"qft20 routed swaps={swaps}", flush=True)

    # 各候选独立评分（复刻 _v5_pipeline 的评分逻辑）
    cands = []
    fk, fin_k = fold_keep_coords(routed, fl, layout)
    cands.append(("fold_keep(2b)", fk, fin_k))
    folded, fold_arr = fold_and_relink(routed, spec)
    if fold_arr is not None:
        flb = dict(fl)
        for k, v in flb.items():
            flb[k] = int(fold_arr[int(v)])
        cands.append(("fold_and_relink(2)", folded, flb))
    cands.append(("裸路由(3)", routed.copy(), dict(fl)))
    # 候选 1（fold_only + relabel 路径——此处只列 fold_only 粗评）
    fo, arr_o = fold_only(routed)
    cands.append(("fold_only(1 前置)", fo, dict(fl)))

    for name, c0, c1 in cands:
        q = decompose_to_platform(c0)
        m = compute_metrics(q, c0.count("swap"), spec.calib,
                         predictor=mapper.fidelity_predictor)
        print(f"[{name}] swaps={m['swap_count']} depth={m['depth']} "
              f"fid={m['est_fidelity']:.6g}", flush=True)


if __name__ == "__main__":
    main()
