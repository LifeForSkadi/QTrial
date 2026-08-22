"""Noise-awareness ablation: compare noise-aware vs topology-only models
on the same circuit subset (swaps / fidelity / mean 2Q error).

Usage: python scripts/ablation_noise.py [--limit 60]
Writes tables/ablation_noise.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import Config, TABLES_DIR, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.mapper import Mapper
from qtrail.utils.qasm_io import load_qasm2, strip_measurements
from qtrail.utils.bench import iter_queko_files

CKPT_NOISE = "checkpoints/tianyan-287_gat_combined_calib_best.pt"
CKPT_TOPO = "checkpoints/tianyan-287_gat_topology_onehot_best.pt"


def layout_noise_cost(pi: dict, qc_clean, spec) -> float | None:
    """Weighted mean 2Q error over the interacting pairs of a layout
    (the QTrail noise-avoidance metric, measured on the LAYOUT itself)."""
    from qtrail.problems import build_program_graph
    from qtrail.utils.qasm_io import extract_ops
    g = build_program_graph(qc_clean.num_qubits, extract_ops(qc_clean),
                            compute_feats=False)
    err2q = spec.calib.err_2q
    median = float(np.median(list(err2q.values())))
    tot_w, tot_e = 0.0, 0.0
    for i in range(g.n):
        for j in range(i + 1, g.n):
            w = float(g.adj[i, j])
            if w <= 0:
                continue
            e = err2q.get((int(pi[i]), int(pi[j])),
                          err2q.get((int(pi[j]), int(pi[i])), median))
            tot_w += w
            tot_e += w * e
    return tot_e / tot_w if tot_w > 0 else None


def run_model(checkpoint: str, circuits, spec, cfg, seed=42) -> dict:
    import torch
    policy, ckpt = QAPolicy.load_checkpoint(checkpoint, device_n=spec.n,
                                            map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda", seed=seed)
    swaps, fid, err2q, static = [], [], [], []
    for name, qc_clean in circuits:
        # route=False: compare the RL LAYOUTS directly (no hybrid adoption)
        res = mapper.map_circuit(qc_clean, circuit_id=name, route=False)
        static.append(res.static_cost)
        le = layout_noise_cost(res.layout, qc_clean, spec)
        if le is not None:
            err2q.append(le)
    return {
        "mean_static_cost": float(np.mean(static)),
        "mean_layout_2q_err": float(np.mean(err2q)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)

    files = list(iter_queko_files("BIGD"))[:args.limit]
    circuits = []
    for p in files:
        qc = load_qasm2(p)
        qc_clean, _ = strip_measurements(qc)
        circuits.append((f"queko/{p.stem}", qc_clean))

    print(f"noise-aware model on {len(circuits)} circuits ...")
    t0 = time.time()
    r_noise = run_model(CKPT_NOISE, circuits, spec, cfg, seed=args.seed)
    print(f"  {r_noise} ({time.time()-t0:.0f}s)")
    print("topology-only model ...")
    t0 = time.time()
    r_topo = run_model(CKPT_TOPO, circuits, spec, cfg, seed=args.seed)
    print(f"  {r_topo} ({time.time()-t0:.0f}s)")

    out = {
        "n_circuits": len(circuits),
        "noise_aware": r_noise,
        "topology_only": r_topo,
        "delta": {
            "static_cost_delta_pct": 100 * (r_noise["mean_static_cost"] / max(r_topo["mean_static_cost"], 1e-12) - 1),
            "layout_2q_err_reduction_pct": 100 * (1 - r_noise["mean_layout_2q_err"] / max(r_topo["mean_layout_2q_err"], 1e-12)),
        },
    }
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / "ablation_noise.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out["delta"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
