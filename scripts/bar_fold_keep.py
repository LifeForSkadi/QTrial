# -*- coding: utf-8 -*-
"""fold_keep_coords 对比（2026-11-13）：heisenberg N16 三种 fold
路径 → _decompose_ablation 后的门数 + 语义 fid。
A) fold_keep_coords（前向，新）  B) fold_and_relink（保守，生产）
C) fold_only + chain（候选 1）
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
from qtrail.pure.post_v2 import (fold_keep_coords, fold_and_relink,
                                 fold_only, chain_long_gates,
                                 _decompose_ablation)

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"


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

    p = None
    for cand in Path("data/benchpress/qasm").rglob("*.qasm"):
        if "square_heisenberg_N16" in cand.name:
            p = cand
            break
    circ = parse_qasm(p.read_text(encoding="utf-8"))

    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    res = mapper.map_circuit(circ, circuit_id="heis16")
    torch.manual_seed(42)
    layout = {i: int(res["layout"][i]) for i in range(circ.n)}
    routed, swaps, fl = sabre_route(circ, spec, layout, seed=42,
                                    lam_ms=mapper.lam_ms, trials=16,
                                    mix_rollout=(circ.n > 10))
    routed, removed = post_route(routed, fl, spec.adj)
    print(f"routed swaps={swaps}", flush=True)

    # A) fold_keep_coords
    t0 = time.time()
    fk, fin_k = fold_keep_coords(routed, fl)
    out_a = _decompose_ablation(fk, spec=spec, resynth_every=1)
    c_a, n_a = count(out_a.ops)
    print(f"A) fold_keep: {dict(c_a)} 1Q={n_a} ({time.time()-t0:.0f}s)",
          flush=True)

    # B) fold_and_relink（生产路径）
    t0 = time.time()
    fb, fin_b = fold_and_relink(routed, spec)
    out_b = _decompose_ablation(fb, spec=spec, resynth_every=1)
    c_b, n_b = count(out_b.ops)
    print(f"B) fold_and_relink: {dict(c_b)} 1Q={n_b} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # C) fold_only + chain
    t0 = time.time()
    fo, arr_o = fold_only(routed)
    fo_ch, _ = chain_long_gates(fo, spec)
    out_c = _decompose_ablation(fo_ch, spec=spec, resynth_every=1)
    c_c, n_c = count(out_c.ops)
    print(f"C) fold_only+chain: {dict(c_c)} 1Q={n_c} "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"基线（v5 完整管线）1Q=736 cz=72", flush=True)


if __name__ == "__main__":
    main()
