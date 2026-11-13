# -*- coding: utf-8 -*-
"""定向重跑 v8 基准行（机制改动后只更新受影响线路）。

用法：
  python scripts/refresh_rows_v8.py --circuits graphstate_50,graphstate_105

按 `final_report_bench_v8.py` 同口径（seed 42、routing_seeds=16、CPU、
v5 管线）重跑指定线路，更新主行文件与全部分片文件中该线路的行。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm

CKPT = ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt")
OUT = Path("tables/final_report_v8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", required=True)
    args = ap.parse_args()
    names = {x for x in args.circuits.split(",") if x}

    buckets, _ = pickle.loads(
        (OUT / "buckets_cache.pkl").read_bytes())
    texts, where = {}, {}
    for b in ("small", "medium", "large"):
        for n, t in buckets[b]:
            if n in names:
                texts[n] = t
                where[n] = b
    missing = names - set(texts)
    if missing:
        raise SystemExit(f"未找到线路: {sorted(missing)}")

    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location="cpu")
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    new_rows = {}
    for name, text in texts.items():
        t0 = time.time()
        torch.manual_seed(42)
        mapper = PureMapper(spec, policy=policy, cfg=cfg, dev="cpu", seed=42,
                            selection_rule="fidelity", use_post=True,
                            pipeline="v5", routing_seeds=16)
        res = mapper.map_circuit(parse_qasm(text), circuit_id=name)
        m = res["metrics"]
        new_rows[name] = {
            "circuit": name, "n": parse_qasm(text).n,
            "qt_swaps": res["swap_count"], "qt_depth": m["depth"],
            "qt_twoq_depth": m["twoq_depth"],
            "qt_twoq_count": m.get("twoq_count"),
            "qt_fidelity": m["est_fidelity"],
            "qt_logf_product": m.get("log_est_fidelity_product"),
            "qt_logf_time": m.get("log_est_fidelity_time"),
            "qt_est_time_us": m.get("est_time_us"),
            "qt_wall": round(time.time() - t0, 2),
        }
        print(f"{name}: 2Q={m['twoq_count']} 2Qd={m['twoq_depth']} "
              f"depth={m['depth']} logf={m['log_est_fidelity']:.3f} "
              f"wall={new_rows[name]['qt_wall']}s", flush=True)

    files = [OUT / f"{b}_fidelity_rows.jsonl" for b in set(where.values())]
    files += [OUT / p.name for p in OUT.glob("*_fidelity_rows_shard*.jsonl")]
    for f in files:
        if not f.exists():
            continue
        lines = [l for l in f.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        rows = [json.loads(l) for l in lines]
        hit = 0
        for i, r in enumerate(rows):
            if r["circuit"] in new_rows:
                rows[i] = new_rows[r["circuit"]]
                hit += 1
        if hit:
            f.write_text("".join(json.dumps(r, ensure_ascii=False,
                                            default=float) + "\n"
                                 for r in rows), encoding="utf-8")
            print(f"updated {hit} row(s) in {f}", flush=True)


if __name__ == "__main__":
    main()
