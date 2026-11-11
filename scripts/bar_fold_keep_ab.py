# -*- coding: utf-8 -*-
"""fold_keep A/B（2026-11-13，干净对照）：同代码 fold_keep_on
开关 × 3 线路完整管线 fid/门数。旧基线数字被 numba 腐蚀污染
（(q,-1) 修复前的表不可信）——本 A/B 是修复后的首次干净对照。
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
TARGETS = ["square_heisenberg_N16", "square_heisenberg_N25", "qft_N020"]


def run_one(tgt, spec, policy, cfg, dev, on):
    p = None
    for cand in BP.rglob("*.qasm"):
        if tgt in cand.name:
            p = cand
            break
    circ = parse_qasm(p.read_text(encoding="utf-8"))
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16, fold_keep_on=on)
    t0 = time.time()
    res = mapper.map_circuit(circ, circuit_id=tgt)
    c = Counter(i.name for i in res["routed_circuit"].ops)
    m = res["metrics"]
    return (c.get("rz", 0) + c.get("sx", 0) + c.get("x", 0),
            c.get("cz", 0), m["est_fidelity"], time.time() - t0)


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

    for tgt in TARGETS:
        n1q_on, cz_on, fid_on, t_on = run_one(tgt, spec, policy, cfg,
                                              dev, True)
        n1q_off, cz_off, fid_off, t_off = run_one(tgt, spec, policy, cfg,
                                                  dev, False)
        print(f"[{tgt}] fold_keep ON:  1Q={n1q_on} cz={cz_on} "
              f"fid={fid_on:.6g}（{t_on:.0f}s）", flush=True)
        print(f"[{tgt}] fold_keep OFF: 1Q={n1q_off} cz={cz_off} "
              f"fid={fid_off:.6g}（{t_off:.0f}s）", flush=True)
        print(f"  fid 比值 {fid_on/fid_off:.3f}、1Q 比值 "
              f"{n1q_on/n1q_off:.3f}", flush=True)


if __name__ == "__main__":
    main()
