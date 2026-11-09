# -*- coding: utf-8 -*-
"""真机拓扑口径基准（选项 C 实验，2026-11-09）。

QTrial pure（fidelity 规则、修复后代码）在【真机拓扑 + 平台真实校准】
下重测（c0.05 权重零重训迁移）。线路集与 final_report_bench_v3.py
完全同口径（Benchpress ≤105 + MQTBench stratified + QUEKO 子集），
便于与合成口径直接对比。

输出：tables/final_report_v3_real/（新目录）
  {bucket}_fidelity_rows.jsonl + summary.md（含合成口径对照）
断点续跑。--only-queko 只跑 QUEKO 部分（快速参数设置实验）。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_real_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OUT = Path("tables/final_report_v3_real")
SYN = Path("tables/final_report_v3")
CACHE = Path("data/mqtbench/stratified")
BP = Path("data/benchpress/qasm")
RULE = "fidelity"


def load_circuits(only_queko: bool = False):
    from qiskit import qasm2
    buckets = {"small": [], "medium": [], "large": []}
    skipped = 0
    if not only_queko:
        for p in BP.rglob("*.qasm"):
            text = p.read_text(encoding="utf-8", errors="replace")
            try:
                circ = parse_qasm(text)
                if circ.n == 0:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue
            if circ.n > 105:
                skipped += 1
                continue
            b = ("small" if circ.n <= 10
                 else ("medium" if circ.n <= 50 else "large"))
            buckets[b].append((f"bp_{p.parent.name}_{p.stem}", text))
        for p in sorted(CACHE.glob("*.qasm")):
            name = p.stem
            size = int(name.rsplit("_", 1)[1])
            if size > 105:
                continue
            b = ("small" if size <= 10
                 else ("medium" if size <= 50 else "large"))
            buckets[b].append((name, sanitize_qasm(p.read_text(encoding="utf-8"))))
    for p in iter_queko_files("BIGD"):
        if re.search(r"\.(3|4)D1_\.(4|5)D2_", p.name) \
                or re.search(r"\.0D1_\.1D2_", p.name):
            buckets["medium"].append(
                (p.stem, sanitize_qasm(p.read_text(encoding="utf-8"))))
    return buckets, skipped


def trimmed(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * 0.1))
    if 2 * k >= len(vals):
        return statistics.mean(vals)
    return statistics.mean(vals[k:len(vals) - k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="small,medium,large")
    ap.add_argument("--only-queko", action="store_true")
    ap.add_argument("--dev", default=None)
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dev={dev}", flush=True)

    dev_cfg = load_device_config("configs/device_tianyan287_real.yaml")
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_real_spec(dev_cfg)
    print(f"真机 spec: n={spec.n} 可用边={int(spec.adj.sum() / 2)} "
          f"禁用比特={spec.disabled_qubits} 校准={dev_cfg.calibration}",
          flush=True)
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    buckets, skipped = load_circuits(only_queko=args.only_queko)
    OUT.mkdir(parents=True, exist_ok=True)
    for bucket in args.buckets.split(","):
        rows = []
        out_file = OUT / f"{bucket}_{RULE}_rows.jsonl"
        done_names = set()
        if out_file.exists():
            for line in out_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done_names.add(r["circuit"])
                    rows.append(r)
        n_done = len(done_names)
        for i, (name, ptext) in enumerate(buckets[bucket]):
            if name in done_names:
                continue
            t0 = time.time()
            row = {"circuit": name, "n": parse_qasm(ptext).n}
            try:
                mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev,
                                    seed=42, selection_rule=RULE, use_post=True)
                res = mapper.map_circuit(parse_qasm(ptext), circuit_id=name)
                row["qt_swaps"] = res["swap_count"]
                row["qt_depth"] = res["metrics"]["depth"]
                row["qt_twoq_depth"] = res["metrics"]["twoq_depth"]
                row["qt_fidelity"] = res["metrics"]["est_fidelity"]
                row["qt_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["qt_error"] = f"{type(e).__name__}: {str(e)[:60]}"
            rows.append(row)
            with open(out_file, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i + 1 + n_done) % 5 == 0:
                print(f"  [{bucket}] {i + 1 + n_done}/{len(buckets[bucket])} "
                      f"({name})", flush=True)
        print(f"[{bucket}] 完成 {len(buckets[bucket])} 条 -> {out_file}",
              flush=True)

    # ---- 汇总（含合成口径对照）
    titles = {"small": "≤10 比特", "medium": "11-50 比特",
              "large": "51-105 比特"}
    syn = {}
    for bucket in ("small", "medium", "large"):
        p = SYN / f"{bucket}_{RULE}_rows.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    syn[r["circuit"]] = r
    lines = ["# 真机拓扑口径 QTrial pure 汇总（v3-real，2026-11-09）\n",
             "设备：真机拓扑（168 可用边 + 4 禁用比特）+ 平台真实校准快照；"
             "权重 c0.05 零重训迁移；后处理连通性感知（修复后）。\n",
             "合成口径对照 = tables/final_report_v3/（188 矩形网格+合成校准）。\n"]
    for bucket in ("small", "medium", "large"):
        p = OUT / f"{bucket}_{RULE}_rows.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in
                p.read_text(encoding="utf-8").splitlines() if l.strip()]
        ok = [r for r in rows if "qt_fidelity" in r]
        n_err = len(rows) - len(ok)

        def stat(key):
            vals = [r[key] for r in ok if key in r]
            if not vals:
                return None
            return (statistics.mean(vals), trimmed(vals),
                    statistics.median(vals))

        def fmt(t):
            return (f"{t[0]:.4f}/{t[1]:.4f}/{t[2]:.4f}" if t and t[0] is not None
                    else "-")

        syn_ok = [syn[r["circuit"]] for r in ok
                  if r["circuit"] in syn and "qt_fidelity" in syn[r["circuit"]]]

        def synstat(key):
            vals = [r[key] for r in syn_ok]
            if not vals:
                return None
            return (statistics.mean(vals), trimmed(vals),
                    statistics.median(vals))

        lines.append(f"\n## {titles[bucket]}（{len(ok)} 条完成，{n_err} 条错误；"
                     f"对照 {len(syn_ok)} 条）\n")
        lines.append("| 指标（均值/截尾/中位） | 真机拓扑 | 合成口径 |\n")
        lines.append("|---|---|---|\n")
        for label, key in (("SWAP 数", "qt_swaps"), ("深度", "qt_depth"),
                           ("2Q 深度", "qt_twoq_depth"),
                           ("估计保真度", "qt_fidelity")):
            lines.append(f"| {label} | {fmt(stat(key))} | {fmt(synstat(key))} |\n")
    (OUT / "summary.md").write_text("".join(lines), encoding="utf-8")
    print(f"\n汇总 -> {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
