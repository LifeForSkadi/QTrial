# -*- coding: utf-8 -*-
"""生成 v8 分层报告表 tables/final_report_v8/report.md（供出图脚本读取）。

QTrial 列 = v8 行数据（y 修复 + mid-circuit 剔除后）；
基线列（感知 O1/O3、盲目 O1/O3、pytket）读 tables/final_report 旧行，
但只统计 v8 同线路子集，保证同集对比。

输出格式与 tables/final_report/report.md 一致。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

V8 = Path("tables/final_report_v8")
OLD = Path("tables/final_report")
TITLES = {"small": ("≤10 比特", "≤10"),
          "medium": ("11-50 比特", "11-50"),
          "large": ("51-105 比特", "51-105")}
METHODS = [("QTrial", "qt"), ("感知O3", "aw_o3"), ("感知O1", "aw_o1"),
           ("盲目O3", "b_o3"), ("盲目O1", "b_o1"), ("pytket", "tket")]


def load(p: Path) -> dict:
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["circuit"]] = r
    return out


def trimmed(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * 0.1))
    return (statistics.mean(vals[k:len(vals) - k])
            if 2 * k < len(vals) else statistics.mean(vals))


def stat(rows, prefix, key):
    vals = [r[f"{prefix}_{key}"] for r in rows
            if r.get(f"{prefix}_{key}") is not None]
    if not vals:
        return None
    return (statistics.mean(vals), trimmed(vals), statistics.median(vals))


def fmt(t):
    return "-" if not t else f"{t[0]:.4f}/{t[1]:.4f}/{t[2]:.4f}"


def main():
    lines = ["# 最终方案分层性能报告 v8（天衍-287，105 比特）\n\n",
             "QTrial = v8 管线（自研 VW 合成 + VF2++ + 帧类型化）；"
             "基线列读旧表但在 v8 同线路子集上重算；统计：均值/截尾(去10%)/中位数。\n"]
    for bucket, (title, key_prefix) in TITLES.items():
        v8 = list(load(V8 / f"{bucket}_fidelity_rows.jsonl").values())
        ok = [r for r in v8 if r.get("qt_fidelity") is not None]
        old = load(OLD / f"{bucket}_fidelity_rows.jsonl")
        names = {r["circuit"] for r in ok}
        merged = []
        for r in ok:
            row = dict(r)
            o = old.get(r["circuit"], {})
            for k, v in o.items():
                row.setdefault(k, v)
            merged.append(row)
        n_err = len(v8) - len(ok)
        lines.append(f"\n## {title} · 规则=fidelity（{len(ok)} 条完成，"
                     f"{n_err} 条错误）\n\n")
        lines.append("| 指标（均值/截尾/中位） | QTrial | 感知O3 | 感知O1 "
                     "| 盲目O3 | 盲目O1 | pytket |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for label, key in (("SWAP 数", "swaps"), ("深度", "depth"),
                           ("2Q 深度", "twoq_depth"),
                           ("估计保真度", "fidelity")):
            cells = []
            for _, prefix in METHODS:
                cells.append(fmt(stat(merged, prefix, key)))
            lines.append(f"| {label} | " + " | ".join(cells) + " |\n")
        # 胜场
        wins = {}
        for label, prefix in (("感知O3", "aw_o3"), ("pytket", "tket"),
                              ("盲目O1", "b_o1")):
            w = sum(1 for r in merged
                    if r.get("qt_fidelity") is not None
                    and r.get(f"{prefix}_fidelity") is not None
                    and r["qt_fidelity"] > r[f"{prefix}_fidelity"])
            n = sum(1 for r in merged
                    if r.get(f"{prefix}_fidelity") is not None)
            wins[label] = (w, n)
        lines.append(
            f"\nQTrial 保真度胜场（{len(ok)} 条）：vs 感知O3 "
            f"**{wins['感知O3'][0]}**、vs pytket **{wins['pytket'][0]}**、"
            f"vs 盲目O1 **{wins['盲目O1'][0]}**\n")
    (V8 / "report.md").write_text("".join(lines), encoding="utf-8")
    print("written", V8 / "report.md")


if __name__ == "__main__":
    main()
