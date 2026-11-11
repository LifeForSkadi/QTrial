# -*- coding: utf-8 -*-
"""fold_keep 完整管线 bar（2026-11-13）：3 线路完整 map_circuit
（含候选选择+relabel+A2）产物 vs 基线（736/1684→管线数字）。
基线（v5 生产）：heis N16 1Q=736 cz=72（完整管线口径）。
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
        p = None
        for cand in BP.rglob("*.qasm"):
            if tgt in cand.name:
                p = cand
                break
        circ = parse_qasm(p.read_text(encoding="utf-8"))
        torch.manual_seed(42)
        mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                            selection_rule="fidelity", use_post=True,
                            pipeline="v5", routing_seeds=16)
        t0 = time.time()
        res = mapper.map_circuit(circ, circuit_id=tgt)
        c = Counter(i.name for i in res["routed_circuit"].ops)
        m = res["metrics"]
        print(f"[{tgt}] 1Q={c.get('rz',0)+c.get('sx',0)+c.get('x',0)} "
              f"cz={c.get('cz',0)} fid={m['est_fidelity']:.6g} "
              f"swaps={res['swap_count']}（{time.time()-t0:.0f}s）",
              flush=True)
        print(f"  {dict(c)}", flush=True)


if __name__ == "__main__":
    main()
