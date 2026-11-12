# -*- coding: utf-8 -*-
"""聚合 final_report_v8 分片 → 主行文件 + 分层对比报告（第 3–6 项新管线）。

用法：python scripts/aggregate_v8.py

- 合并 `{bucket}_fidelity_rows_shard*.jsonl` → `{bucket}_fidelity_rows.jsonl`
  （按 circuit 去重）；
- 输出 `tables/final_report_v8/summary_v8.md`：各层 qt 指标（均值/截尾/
  中位）与 v7、v3、感知 O3、pytket 对比；相对感知 O3 的中位比值与胜率
  （仅可比线路）。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path("tables/final_report_v8")
V7 = Path("tables/final_report_v7")
V3 = Path("tables/final_report_v3")
OLD = Path("tables/final_report")
V6 = Path("tables/final_report_v6")
BUCKETS = ("small", "medium", "large")
TITLES = {"small": "≤10 比特", "medium": "11-50 比特",
          "large": "51-105 比特"}


def _load(p: Path) -> dict:
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["circuit"]] = r
    return out


def _trimmed(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * 0.1))
    return (statistics.mean(vals[k:len(vals) - k])
            if 2 * k < len(vals) else statistics.mean(vals))


def _fmt(v):
    return f"{v[0]:.4f}/{v[1]:.4f}/{v[2]:.4f}" if v else "-"


def _stat(rows, prefix, key):
    vals = []
    for r in rows:
        full = f"{prefix}_{key}"
        if full in r and r[full] is not None:
            vals.append(r[full])
    if not vals:
        return None
    return (statistics.mean(vals), _trimmed(vals), statistics.median(vals))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# QTrial v8 分层汇总（2026-11-13 新管线）\n\n",
             "v8 = v7 同口径 + A2 自研 VW 合成（默认）+ 自研 VF2++ "
             "精确嵌入 + 候选类型化/Inst 冻结。基线数字读旧表"
             "（基线未重跑，管线改动不影响基线）。\n"]
    for bucket in BUCKETS:
        rows = []
        main_p = OUT / f"{bucket}_fidelity_rows.jsonl"
        if main_p.exists():
            rows += list(_load(main_p).values())
        for sp in sorted(OUT.glob(f"{bucket}_fidelity_rows_shard*.jsonl")):
            rows += list(_load(sp).values())
        seen, uniq = set(), []
        for r in rows:
            if r["circuit"] not in seen:
                seen.add(r["circuit"])
                uniq.append(r)
        rows = uniq
        if not rows:
            continue
        main_p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                  for r in rows), encoding="utf-8")
        ok = [r for r in rows if "qt_fidelity" in r]
        n_err = len(rows) - len(ok)
        v7 = _load(V7 / f"{bucket}_fidelity_rows.jsonl")
        v3 = _load(V3 / f"{bucket}_fidelity_rows.jsonl")
        old = _load(OLD / f"{bucket}_fidelity_rows.jsonl")
        v6 = _load(V6 / f"{bucket}_fidelity_rows.jsonl")
        ok7 = [v7[r["circuit"]] for r in ok if r["circuit"] in v7]
        ok3 = [v3[r["circuit"]] for r in ok if r["circuit"] in v3]
        ok6 = [v6[r["circuit"]] for r in ok if r["circuit"] in v6]
        oko = [old[r["circuit"]] for r in ok if r["circuit"] in old]
        lines.append(f"\n## {TITLES[bucket]}（{len(ok)} 条完成，{n_err} 条错误）\n\n")
        lines.append("| 指标（均值/截尾/中位） | v8 | v7 | v6 老路由器 | v3 修复后 | 感知 O3 | pytket |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for label, key in (("SWAP 数", "swaps"), ("深度", "depth"),
                           ("2Q 深度", "twoq_depth"),
                           ("估计保真度", "fidelity")):
            lines.append(
                f"| {label} | {_fmt(_stat(ok, 'qt', key))} "
                f"| {_fmt(_stat(ok7, 'qt', key))} "
                f"| {_fmt(_stat(ok6, 'qt', key))} "
                f"| {_fmt(_stat(ok3, 'qt', key))} "
                f"| {_fmt(_stat(oko, 'aw_o3', key))} "
                f"| {_fmt(_stat(oko, 'tket', key))} |\n")
        # 相对感知 O3：中位比值 + 胜率（仅可比线路）
        ratios, wins, n = [], 0, 0
        for r in ok:
            o = old.get(r["circuit"])
            if not o or not o.get("aw_o3_fidelity"):
                continue
            q = r["qt_fidelity"]
            b = o["aw_o3_fidelity"]
            ratios.append(q / b)
            n += 1
            if q > b:
                wins += 1
        if n:
            lines.append(f"\n相对感知 O3：可比 {n} 条，中位比值 "
                         f"{statistics.median(ratios):.3f}，胜率 "
                         f"{100 * wins / n:.0f}%。\n")
        # 相对 v7（同线路）
        r7, w7, n7 = [], 0, 0
        for r in ok:
            o = v7.get(r["circuit"])
            if not o or not o.get("qt_fidelity"):
                continue
            q = r["qt_fidelity"]
            b = o["qt_fidelity"]
            r7.append(q / b)
            n7 += 1
            if q > b:
                w7 += 1
        if n7:
            lines.append(f"相对 v7（同管线前代）：可比 {n7} 条，中位比值 "
                         f"{statistics.median(r7):.3f}，胜率 "
                         f"{100 * w7 / n7:.0f}%。\n")
    (OUT / "summary_v8.md").write_text("".join(lines), encoding="utf-8")
    print("written", OUT / "summary_v8.md")


if __name__ == "__main__":
    main()
