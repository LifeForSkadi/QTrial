# -*- coding: utf-8 -*-
"""v8 修正口径汇总：基线补回读出误差后的分层对比。

背景：`final_report_bench.py` 跑 qiskit 基线前剥离 measure，导致基线
保真度不含读出项，而 QTrial 产物带 measure、含读出项。本脚本读取
`fix_baseline_readout.py` 生成的修正表（`*_baseline_readout.jsonl`），
输出与 QTrial 对称的对比。

只有原线路本身带 measure 的线路受影响（large 35/199、medium 64/258、
small 78/127）；无 measure 的线路两侧都只算门误差。

输出：tables/final_report_v8/summary_v8_corrected.md
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path("tables/final_report_v8")
BUCKETS = ("small", "medium", "large")
TITLES = {"small": "≤10 比特", "medium": "11-50 比特",
          "large": "51-105 比特"}
ARMS = ("b_o1", "b_o3", "aw_o1", "aw_o3")
LABELS = {"qt": "QTrial", "b_o1": "盲目O1", "b_o3": "盲目O3",
          "aw_o1": "感知O1", "aw_o3": "感知O3", "tket": "pytket"}


def load(p: Path) -> dict:
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["circuit"]] = r
    return out


def load_qt(bucket: str) -> dict:
    out = load(OUT / f"{bucket}_fidelity_rows.jsonl")
    for sp in sorted(OUT.glob(f"{bucket}_fidelity_rows_shard*.jsonl")):
        out.update(load(sp))
    return out


def trim(vals):
    vals = sorted(vals)
    k = max(1, int(len(vals) * 0.1))
    return (statistics.mean(vals[k:len(vals) - k])
            if 2 * k < len(vals) else statistics.mean(vals))


def f3(x):
    return f"{x:.4g}"


def main():
    stats = {}
    lines = ["# QTrial v8 修正口径汇总（基线补回读出误差）\n\n",
             "**口径修正（2026-11-13 审计）**：`final_report_bench.py` 在"
             "transpile 前剥离 measure，基线保真度不含读出；QTrial 产物"
             "带 measure、含读出。本表用 `fix_baseline_readout.py` 重跑"
             "四条 qiskit 基线、用 `fix_pytket_readout.py` 重跑 pytket"
             "（门级保真度逐条复现旧表），按原线路实际测量的逻辑比特"
             "补回读出项。无 measure 的线路两侧均只计门误差。\n\n",
             "统计：均值/截尾(去10%)/中位数（乘积保真度，含读出）；"
             "log10 为等权对数口径（大线路乘积下溢时唯一可比的量）。\n"]
    for bucket in BUCKETS:
        qt = load_qt(bucket)
        cor = load(OUT / f"{bucket}_baseline_readout.jsonl")
        ptk = load(OUT / f"{bucket}_pytket_readout.jsonl")
        for sp in sorted(OUT.glob(f"{bucket}_pytket_readout_shard*.jsonl")):
            ptk.update(load(sp))
        names = [c for c in qt if c in cor
                 and all(f"{m}_fidelity_with_ro" in cor[c] for m in ARMS)]
        names_t = [c for c in names
                   if "tket_fidelity_with_ro" in ptk.get(c, {})]
        if not names:
            continue
        n_meas = sum(1 for c in names if cor[c].get("b_o1_n_measured", 0) > 0)
        lines.append(f"\n## {TITLES[bucket]}（{len(names)} 条；"
                     f"带 measure {n_meas} 条；pytket {len(names_t)} 条）\n\n")
        lines.append("| 指标 | QTrial | 盲目O1 | 盲目O3 | 感知O1 | 感知O3 "
                     "| pytket |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for label, fn in (("保真度 均值", statistics.mean),
                          ("保真度 截尾", trim),
                          ("保真度 中位", statistics.median)):
            cells = []
            for m in ("qt",) + ARMS + ("tket",):
                if m == "tket":
                    vals = [ptk[c]["tket_fidelity_with_ro"] for c in names_t]
                else:
                    vals = [qt[c]["qt_fidelity"] if m == "qt"
                            else cor[c][f"{m}_fidelity_with_ro"]
                            for c in names]
                cells.append(f3(fn(vals)) if vals else "-")
            lines.append(f"| {label} | " + " | ".join(cells) + " |\n")
        for label, fn in (("log10 均值", statistics.mean),
                          ("log10 中位", statistics.median)):
            cells = []
            for m in ("qt",) + ARMS + ("tket",):
                if m == "tket":
                    vals = [ptk[c]["tket_logf_with_ro"] / math.log(10)
                            for c in names_t]
                elif m == "qt":
                    vals = [qt[c]["qt_logf_product"] / math.log(10)
                            for c in names]
                else:
                    vals = [cor[c][f"{m}_logf_with_ro"] / math.log(10)
                            for c in names]
                cells.append(f"{fn(vals):.2f}" if vals else "-")
            lines.append(f"| {label} | " + " | ".join(cells) + " |\n")
        lines.append("\n")
        stats[bucket] = {"n": len(names), "n_meas": n_meas,
                         "n_tket": len(names_t)}
        for m in ARMS + ("tket",):
            if m == "tket":
                cs = names_t
                wins = sum(1 for c in cs if qt[c]["qt_fidelity"]
                           > ptk[c]["tket_fidelity_with_ro"])
                ratios = [qt[c]["qt_fidelity"]
                          / ptk[c]["tket_fidelity_with_ro"] for c in cs
                          if ptk[c]["tket_fidelity_with_ro"] > 0]
                vals = [ptk[c]["tket_fidelity_with_ro"] for c in cs]
            else:
                cs = names
                wins = sum(1 for c in cs if qt[c]["qt_fidelity"]
                           > cor[c][f"{m}_fidelity_with_ro"])
                ratios = [qt[c]["qt_fidelity"]
                          / cor[c][f"{m}_fidelity_with_ro"] for c in cs
                          if cor[c][f"{m}_fidelity_with_ro"] > 0]
                vals = [cor[c][f"{m}_fidelity_with_ro"] for c in cs]
            stats[bucket][m] = {
                "wins": wins, "n": len(cs),
                "median_ratio": statistics.median(ratios),
                "mean_fidelity": statistics.mean(vals),
                "trim_fidelity": trim(vals),
                "median_fidelity": statistics.median(vals),
            }
            lines.append(f"- vs {LABELS[m]}：胜 {wins}/{len(cs)}，"
                         f"中位比值 {statistics.median(ratios):.3g}\n")
        vals = [qt[c]["qt_fidelity"] for c in names]
        stats[bucket]["qt"] = {
            "mean_fidelity": statistics.mean(vals),
            "trim_fidelity": trim(vals),
            "median_fidelity": statistics.median(vals),
        }
        # 指标行（SWAP/深度/2Q 深度）：QTrial 定向重跑后与 pytket 新版
        # 基线都需要同步（2026-11-13）
        def _m3(v):
            return [statistics.mean(v), trim(v), statistics.median(v)]
        # 指标行沿用旧表口径：QTrial 全部完成线路（127/258/199），
        # 不因基线可比子集而缩小；保真度行才用共同子集。
        qt_all = [c for c in qt if all(
            f"qt_{k}" in qt[c] for k in ("swaps", "depth", "twoq_depth"))]
        stats[bucket]["qt_metrics"] = {
            k: _m3([qt[c][f"qt_{k}"] for c in qt_all])
            for k in ("swaps", "depth", "twoq_depth")}
        if names_t:
            stats[bucket]["tket_metrics"] = {
                k: _m3([ptk[c][f"tket_{k}_rr"] for c in names_t])
                for k in ("swaps", "depth", "twoq_depth")}
    (OUT / "summary_v8_corrected.md").write_text(
        "".join(lines), encoding="utf-8")
    (OUT / "corrected_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'summary_v8_corrected.md'}")


if __name__ == "__main__":
    main()
