# -*- coding: utf-8 -*-
"""全样本真机可运行验证（587 条）：每条跑生产 mapper（v5 管线、
routing_seeds=16、timeout 900s），产物做真机合法性三重扫描 +
指标收集。断点续跑、--shard 分片。

输出：tables/final_report_v7/legality_rows[_shardK].jsonl
每行：circuit/n/twoq/depth/fid/wall + legal(bool) + 违规明细。
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
BP = Path("data/benchpress/qasm")
CACHE = Path("data/mqtbench/stratified")
OUT = Path("tables/final_report_v7")


def load_circuits():
    """与 final_report_bench_v7 完全同口径的 587 线路清单
    （benchpress + mqtbench/stratified + queko BIGD 三源）。"""
    buckets = {"small": [], "medium": [], "large": []}
    for p in BP.rglob("*.qasm"):
        text = p.read_text(encoding="utf-8", errors="replace")
        try:
            circ = parse_qasm(text)
            if circ.n == 0:
                continue
        except Exception:
            continue
        if circ.n > 105:
            continue
        b = "small" if circ.n <= 10 else ("medium" if circ.n <= 50 else "large")
        buckets[b].append((f"bp_{p.parent.name}_{p.stem}", text))
    for p in sorted(CACHE.glob("*.qasm")):
        name = p.stem
        size = int(name.rsplit("_", 1)[1])
        if size > 105:
            continue
        b = "small" if size <= 10 else ("medium" if size <= 50 else "large")
        buckets[b].append((name, sanitize_qasm(p.read_text(encoding="utf-8"))))
    for p in iter_queko_files("BIGD"):
        if re.search(r"\.(3|4)D1_\.(4|5)D2_", p.name) \
                or re.search(r"\.0D1_\.1D2_", p.name):
            buckets["medium"].append(
                (p.stem, sanitize_qasm(p.read_text(encoding="utf-8"))))
    return buckets


def run_with_timeout(fn, timeout):
    box = {}

    def target():
        try:
            box["res"] = fn()
        except Exception as e:
            box["err"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, "timeout"
    if "err" in box:
        raise box["err"]
    return box.get("res"), None


def check_legality(final_qc, spec):
    problems = []
    names = set()
    disabled = set(spec.disabled_qubits)
    for idx, inst in enumerate(final_qc.ops):
        names.add(inst.name)
        qs = [int(q) for q in inst.qubits]
        if inst.name == "swap":
            problems.append(f"swap残留@{idx}")
        if any(q >= spec.n for q in qs):
            problems.append(f"越界{qs}@{idx}")
        if any(q in disabled for q in qs):
            problems.append(f"禁用{qs}@{idx}")
        if inst.name == "cz" and not spec.adj[qs[0], qs[1]]:
            problems.append(f"cz非相邻{qs}@{idx}")
    if not names <= {"rz", "sx", "x", "cz", "measure"}:
        problems.append(f"门集{names - {'rz', 'sx', 'x', 'cz', 'measure'}}")
    return problems


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="分片 k/N")
    ap.add_argument("--buckets", default="small,medium,large",
                    help="只跑指定层（并行分片用）")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--seeds", type=int, default=16)
    args = ap.parse_args()
    shard_id = shard_n = None
    if args.shard:
        shard_id, shard_n = (int(x) for x in args.shard.split("/"))
    buckets_sel = [b for b in args.buckets.split(",") if b]

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    buckets = load_circuits()
    total = sum(len(v) for v in buckets.values())
    print(f"circuits: small {len(buckets['small'])} / medium "
          f"{len(buckets['medium'])} / large {len(buckets['large'])} "
          f"(total {total})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = (f"_{buckets_sel[0]}_shard{shard_id}"
              if shard_id is not None else f"_{buckets_sel[0]}")
    out_file = OUT / f"legality_rows{suffix}.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["circuit"])

    for bucket in buckets_sel:
        items = list(enumerate(buckets[bucket]))
        if shard_id is not None:
            lo = shard_id * len(items) // shard_n
            hi = (shard_id + 1) * len(items) // shard_n
            items = items[lo:hi]
        for i, (name, ptext) in items:
            if name in done:
                continue
            t0 = time.time()
            row = {"circuit": name, "n": parse_qasm(ptext).n}
            try:
                torch.manual_seed(42)
                mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev,
                                    seed=42, selection_rule="fidelity",
                                    use_post=True, pipeline="v5",
                                    routing_seeds=args.seeds)
                res, terr = run_with_timeout(
                    lambda: mapper.map_circuit(parse_qasm(ptext),
                                               circuit_id=name),
                    args.timeout)
                if terr is not None:
                    row["error"] = "timeout"
                else:
                    final_qc = res["routed_circuit"]
                    m = res["metrics"]
                    probs = check_legality(final_qc, spec)
                    row["legal"] = not probs
                    if probs:
                        row["violations"] = probs[:8]
                    row["twoq"] = m["twoq_count"]
                    row["depth"] = m["depth"]
                    row["fid"] = m["est_fidelity"]
                    row["wall"] = round(time.time() - t0, 1)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e)[:80]}"
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            flag = "✓" if row.get("legal") else (
                "✗" + str(row.get("violations", row.get("error"))))
            print(f"[{bucket}] {name}: {flag}", flush=True)


if __name__ == "__main__":
    main()
