# -*- coding: utf-8 -*-
"""最终分层报告 v3：QTrial pure 修复后口径重测（仅本方案，基线不变）。

修复（2026-11-09）：后处理栈硬件连通性感知（_swap_past 重标记须保持
2Q 端点相邻）+ QCIS 指令集修正。修复使输出线路全部硬件合法，但残留
SWAP 增加——本脚本重测 QTrial pure（fidelity 规则）在 587 条线路上的
新口径数字。

数据集与 v2（final_report_bench.py）完全同口径：
  Benchpress ≤105 比特全部 + MQTBench stratified 缓存 + QUEKO BIGD 子集。
基线（感知 O1/O3、盲目 O1/O3、pytket）走 qiskit/pytket 管线，不受本次
修复影响，不重测；对比时读取旧表（tables/final_report/）。

输出：tables/final_report_v3/（新目录；旧 tables/final_report/ 原样保留）
  {bucket}_fidelity_rows.jsonl  +  summary.md（含旧口径 QTrial 对照）
断点续跑：已存在的 rows 自动跳过。
"""
from __future__ import annotations

import json
import re
import statistics
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
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OUT = Path("tables/final_report_v3")
OLD = Path("tables/final_report")
CACHE = Path("data/mqtbench/stratified")
BP = Path("data/benchpress/qasm")
RULE = "fidelity"


def load_circuits():
    """与 v2 完全同口径的三组线路。"""
    from qiskit import qasm2
    buckets = {"small": [], "medium": [], "large": []}
    skipped = 0
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
        b = "small" if circ.n <= 10 else ("medium" if circ.n <= 50 else "large")
        name = f"bp_{p.parent.name}_{p.stem}"
        buckets[b].append((name, text))
    for p in sorted(CACHE.glob("*.qasm")):
        name = p.stem
        size = int(name.rsplit("_", 1)[1])
        if size > 105:
            continue
        b = "small" if size <= 10 else ("medium" if size <= 50 else "large")
        text = sanitize_qasm(p.read_text(encoding="utf-8"))
        buckets[b].append((name, text))
    for p in iter_queko_files("BIGD"):
        if re.search(r"\.(3|4)D1_\.(4|5)D2_", p.name) \
                or re.search(r"\.0D1_\.1D2_", p.name):
            text = sanitize_qasm(p.read_text(encoding="utf-8"))
            buckets["medium"].append((p.stem, text))
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
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dev={dev}", flush=True)
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    buckets, skipped = load_circuits()
    total = sum(len(v) for v in buckets.values())
    print(f"circuits: small {len(buckets['small'])} / medium "
          f"{len(buckets['medium'])} / large {len(buckets['large'])} "
          f"(total {total}, skipped {skipped})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for bucket in ("small", "medium", "large"):
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

    # ---- 汇总（含旧口径对照）
    titles = {"small": "≤10 比特", "medium": "11-50 比特",
              "large": "51-105 比特"}
    old = {}
    for bucket in ("small", "medium", "large"):
        p = OLD / f"{bucket}_{RULE}_rows.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    old[r["circuit"]] = r
    lines = ["# QTrial pure 修复后口径重测汇总（v3，2026-11-09）\n",
             "修复：后处理栈硬件连通性感知（非相邻 2Q 门归零）+ QCIS 指令集"
             "修正。基线（感知/盲目 O1/O3、pytket）不受影响未重测。\n",
             "旧口径数字 = tables/final_report/（v2，修复前）。\n"]
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
                return None, None, None
            return (statistics.mean(vals), trimmed(vals),
                    statistics.median(vals))

        def fmt(t):
            if t is None or t[0] is None:
                return "-"
            return f"{t[0]:.4f}/{t[1]:.4f}/{t[2]:.4f}"

        sw_new, dep_new, f2_new, f_new = stat("qt_swaps"), stat("qt_depth"), \
            stat("qt_twoq_depth"), stat("qt_fidelity")
        # 旧口径（同线路集合）
        old_ok = [old[r["circuit"]] for r in ok
                  if r["circuit"] in old and "qt_fidelity" in old[r["circuit"]]]

        def ostat(key):
            vals = [r[key] for r in old_ok]
            if not vals:
                return None
            return (statistics.mean(vals), trimmed(vals),
                    statistics.median(vals))

        sw_old, dep_old, f2_old, f_old = ostat("qt_swaps"), ostat("qt_depth"), \
            ostat("qt_twoq_depth"), ostat("qt_fidelity")
        lines.append(f"\n## {titles[bucket]}（{len(ok)} 条完成，{n_err} 条错误）\n")
        lines.append("| 指标（均值/截尾/中位） | v3 修复后 QTrial | v2 旧口径 QTrial |\n")
        lines.append("|---|---|---|\n")
        lines.append(f"| SWAP 数 | {fmt(sw_new)} | {fmt(sw_old)} |\n")
        lines.append(f"| 深度 | {fmt(dep_new)} | {fmt(dep_old)} |\n")
        lines.append(f"| 2Q 深度 | {fmt(f2_new)} | {fmt(f2_old)} |\n")
        lines.append(f"| 估计保真度 | {fmt(f_new)} | {fmt(f_old)} |\n")
    (OUT / "summary.md").write_text("".join(lines), encoding="utf-8")
    print(f"\n汇总 -> {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
