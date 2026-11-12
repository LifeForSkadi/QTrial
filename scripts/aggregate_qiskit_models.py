# -*- coding: utf-8 -*-
"""聚合 Qiskit 基线双模型重评（product + time）→ summary_models.md。

用法：python scripts/aggregate_qiskit_models.py

输入：tables/qiskit_baseline_models/{bucket}_rows[_shardK].jsonl
输出：同目录 {bucket}_rows.jsonl（合并去重）+ summary_models.md
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path("tables/qiskit_baseline_models")
BUCKETS = ("small", "medium", "large")
CONFIGS = (("b_o1", "盲目 O1"), ("b_o3", "盲目 O3"),
           ("aw_o1", "感知 O1"), ("aw_o3", "感知 O3"))
TITLES = {"small": "≤10 比特", "medium": "11-50 比特",
          "large": "51-105 比特"}


def load_rows(bucket: str) -> list[dict]:
    rows = []
    for p in sorted(OUT.glob(f"{bucket}_rows*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    seen = set()
    uniq = []
    for r in rows:
        if r["circuit"] not in seen:
            seen.add(r["circuit"])
            uniq.append(r)
    return uniq


def med(vals):
    return statistics.median(vals) if vals else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Qiskit 基线双模型重评（2026-11-13）\n\n",
             "4 配置（盲目/感知 × O1/O3），transpile 配置与 "
             "`final_report_bench.py` 完全一致；产物转纯 Circuit 后用 "
             "**product** 与 **time** 两种模型评分。\n\n",
             "注：2Q 门数为 `cz`/`cx`/`swap` 计数；SWAP 在 qiskit 产物中"
             "已分解为 cz，故不再单列 swap。\n"]
    for bucket in BUCKETS:
        rows = load_rows(bucket)
        if not rows:
            continue
        main_p = OUT / f"{bucket}_rows.jsonl"
        main_p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                  for r in rows), encoding="utf-8")
        lines.append(f"\n## {TITLES[bucket]}（{len(rows)} 条）\n\n")
        lines.append("| 配置 | 完成 | 深度中位 | 2Q 数中位 | "
                     "product logf 中位 | time logf 中位 | time 中位 µs |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        stats = {}
        for prefix, label in CONFIGS:
            ok = [r for r in rows if f"{prefix}_logf_product" in r]
            depth = [r[f"{prefix}_depth"] for r in ok]
            twoq = [r[f"{prefix}_twoq_count"] for r in ok]
            lp = [r[f"{prefix}_logf_product"] for r in ok]
            lt = [r[f"{prefix}_logf_time"] for r in ok]
            tu = [r[f"{prefix}_time_us"] for r in ok]
            stats[prefix] = (med(lp), med(lt))
            lines.append(
                f"| {label} | {len(ok)}/{len(rows)} | {med(depth):.0f} | "
                f"{med(twoq):.0f} | {med(lp):.4f} | {med(lt):.4f} | "
                f"{med(tu):.0f} |\n")
        # 排序对比：product vs time 下哪个配置最优
        rp = sorted(stats.items(), key=lambda kv: kv[1][0], reverse=True)
        rt = sorted(stats.items(), key=lambda kv: kv[1][1], reverse=True)
        lines.append(f"\nproduct 排名（logf 中位）：{' > '.join(k for k, _ in rp)}\n")
        lines.append(f"time 排名（logf 中位）：{' > '.join(k for k, _ in rt)}\n")
        if rp[0][0] != rt[0][0]:
            lines.append("**排名发生变化**：time 模型最优配置与 product 不同。\n")
    (OUT / "summary_models.md").write_text("".join(lines), encoding="utf-8")
    print("written", OUT / "summary_models.md")


if __name__ == "__main__":
    main()
