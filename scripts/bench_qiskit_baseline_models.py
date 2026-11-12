# -*- coding: utf-8 -*-
"""qiskit 基线双模型重评（2026-11-13，用户要求：与 Qiskit 对比 + 新模型）。

对 587 线路集重跑 qiskit 4 个基线配置（盲目 O1/O3、感知 O1/O3；配置与
`final_report_bench.py` 完全同口径），把产物转成纯 Circuit 后用
**product**（历史）与 **time**（时间感知）两种保真度模型评分。

用法：
  python scripts/bench_qiskit_baseline_models.py [--buckets small,medium,large]
      [--shard k/N] [--timeout 900]

输出：tables/qiskit_baseline_models/{bucket}_rows[_shardK].jsonl
字段：{prefix}_{depth,twoq_depth,twoq_count,logf_product,logf_time,
est_product,time_us,wall,error}
"""
from __future__ import annotations

import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util

import torch
from qiskit import transpile

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.metrics import (_product_log_fidelity, _time_log_fidelity,
                                 circuit_depth, twoq_count, twoq_depth)

OUT = Path("tables/qiskit_baseline_models")
RULE = "fidelity"
CONFIGS = ((1, "b_o1", False), (3, "b_o3", False),
           (1, "aw_o1", True), (3, "aw_o3", True))


def _load_v2():
    p = Path("scripts/final_report_bench.py")
    spec = importlib.util.spec_from_file_location("frb", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_with_timeout(fn, timeout):
    """守护线程执行；超时返回 (None, 'timeout')，不阻塞进程退出。"""
    box = {}
    q = queue.Queue()

    def target():
        try:
            box["res"] = fn()
        except Exception as e:  # noqa: BLE001
            box["err"] = e
        q.put(True)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    try:
        q.get(timeout=timeout) if timeout else q.get()
    except queue.Empty:
        return None, "timeout"
    if "err" in box:
        raise box["err"]
    return box.get("res"), None


def qc_to_circuit(qc) -> Circuit:
    circ = Circuit(qc.num_qubits)
    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        qs = tuple(int(qc.find_bit(q).index) for q in inst.qubits)
        params = tuple(float(p) for p in inst.operation.params)
        circ.append(Inst(name, qs, params))
    return circ


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="small,medium,large")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    shard_id = shard_n = None
    if args.shard:
        shard_id, shard_n = (int(x) for x in args.shard.split("/"))
        assert 0 <= shard_id < shard_n

    frb = _load_v2()
    spec = build_tianyan287_spec(load_device_config())
    calib = spec.calib
    target = frb.build_target(spec)
    cm = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
          if spec.adj[i, j]]
    buckets, skipped = frb.load_circuits()
    print(f"circuits: " + " / ".join(
        f"{b} {len(buckets[b])}" for b in buckets), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for bucket in [b for b in args.buckets.split(",") if b]:
        suffix = f"_shard{shard_id}" if shard_id is not None else ""
        out_file = OUT / f"{bucket}_rows{suffix}.jsonl"
        done = set()
        if out_file.exists():
            for line in out_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["circuit"])
        items = list(enumerate(buckets[bucket]))
        if shard_id is not None:
            lo = shard_id * len(items) // shard_n
            hi = (shard_id + 1) * len(items) // shard_n
            items = items[lo:hi]
        for i, (name, ptext, qc) in items:
            if name in done or qc is None:
                continue
            row = {"circuit": name, "bucket": bucket, "n": qc.num_qubits}
            qc_clean = qc.copy()
            qc_clean.data = [i for i in qc_clean.data
                             if i.operation.name not in ("measure", "barrier")]
            for opt, prefix, aware in CONFIGS:
                kw = {"target": target} if aware else {"coupling_map": cm}
                t0 = time.time()
                try:
                    def run():
                        return transpile(
                            qc_clean,
                            basis_gates=["rz", "sx", "x", "cz"],
                            optimization_level=opt, seed_transpiler=42, **kw)
                    out_qc, terr = run_with_timeout(run, args.timeout)
                    if terr == "timeout":
                        row[f"{prefix}_error"] = f"timeout_{args.timeout:.0f}s"
                        continue
                    circ = qc_to_circuit(out_qc)
                    lp = _product_log_fidelity(circ, calib)
                    lt, t_us = _time_log_fidelity(circ, calib)
                    row[f"{prefix}_depth"] = circuit_depth(circ)
                    row[f"{prefix}_twoq_depth"] = twoq_depth(circ)
                    row[f"{prefix}_twoq_count"] = twoq_count(circ)
                    row[f"{prefix}_logf_product"] = float(lp)
                    row[f"{prefix}_logf_time"] = float(lt)
                    row[f"{prefix}_est_product"] = float(math.exp(lp))
                    row[f"{prefix}_time_us"] = float(t_us)
                    row[f"{prefix}_wall"] = round(time.time() - t0, 2)
                except Exception as e:  # noqa: BLE001
                    row[f"{prefix}_error"] = f"{type(e).__name__}: {str(e)[:60]}"
            with out_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 5 == 0:
                print(f"  [{bucket}] {i + 1}/{len(buckets[bucket])} {name}",
                      flush=True)
        print(f"[{bucket}] 完成 -> {out_file}", flush=True)


if __name__ == "__main__":
    main()
