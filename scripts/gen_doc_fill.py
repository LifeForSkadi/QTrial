"""从 final_report jsonl 生成文档填充块，替换两篇文档中的【BENCH_FILL】占位。

用法（bench 全部完成后）：
    python scripts/gen_doc_fill.py

数据源：tables/final_report/{small,medium,large}_fidelity_rows.jsonl
输出：
    1. tables/final_report/summary_for_docs.md —— 独立汇总（人工可读）
    2. 就地替换 docs/technical_report.md 与 docs/competition_report.md 中
       所有含【BENCH_FILL】的占位行。
统计口径与 final_report_bench.py 完全一致：均值/截尾(去10%)/中位数。
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tables" / "final_report"
DOCS = [ROOT / "docs" / "technical_report.md",
        ROOT / "docs" / "competition_report.md"]
COLS = [("QTrial", "qt"), ("感知O3", "aw_o3"), ("感知O1", "aw_o1"),
        ("盲目O3", "b_o3"), ("盲目O1", "b_o1"), ("pytket", "tket")]
TITLES = {"small": "≤10 比特", "medium": "11-50 比特", "large": "51-105 比特"}
METRICS = [("swaps", "SWAP 数", 2), ("depth", "深度", 2),
           ("twoq_depth", "2Q 深度", 2), ("fidelity", "估计保真度", 4)]


def trimmed(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * 0.1))
    if 2 * k >= len(vals):
        return statistics.mean(vals)
    return statistics.mean(vals[k:len(vals) - k])


def fmt_cell(vals, nd):
    if not vals:
        return "-"
    if nd == 2:
        return (f"{statistics.mean(vals):.2f}/{trimmed(vals):.2f}/"
                f"{statistics.median(vals):.2f}")
    return (f"{statistics.mean(vals):.4f}/{trimmed(vals):.4f}/"
            f"{statistics.median(vals):.4f}")


def bucket_table(rows):
    lines = ["| 指标（均值/截尾/中位） | " + " | ".join(c for c, _ in COLS) + " |",
             "|---|---|---|---|---|---|---|"]
    for metric, name, nd in METRICS:
        cells = []
        for _, prefix in COLS:
            vals = [r.get(f"{prefix}_{metric}") for r in rows
                    if isinstance(r.get(f"{prefix}_{metric}"), (int, float))]
            cells.append(fmt_cell(vals, nd))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def win_stats(rows):
    """QTrial 保真度胜场：与全完成子集逐线路比较。"""
    ok = [r for r in rows if "qt_fidelity" in r and "aw_o3_fidelity" in r
          and "tket_fidelity" in r and "b_o1_fidelity" in r]
    if not ok:
        return "", 0
    w3 = sum(1 for r in ok if r["qt_fidelity"] > r["aw_o3_fidelity"])
    wt = sum(1 for r in ok if r["qt_fidelity"] > r["tket_fidelity"])
    wb = sum(1 for r in ok if r["qt_fidelity"] > r["b_o1_fidelity"])
    s = (f"QTrial 保真度胜场（{len(ok)} 条全完成）：vs 感知O3 **{w3}**、"
         f"vs pytket **{wt}**、vs 盲目O1 **{wb}**")
    return s, len(ok)


def build_all():
    buckets = {}
    for b in ("small", "medium", "large"):
        p = OUT / f"{b}_fidelity_rows.jsonl"
        if not p.exists():
            raise SystemExit(f"[x] missing {p} — bench 未完成")
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        buckets[b] = rows
    n_total = sum(len(v) for v in buckets.values())
    if n_total < 580:
        raise SystemExit(f"[x] 数据不完整：仅 {n_total} 条（预期 587）")

    # ---- 独立汇总文件
    sec = ["# 最终分层基准汇总（自动生成，供文档引用）\n",
           f"数据源：{n_total} 条线路（≤10 / 11-50 / 51-105 比特），"
           "QTrial pure fidelity 规则 vs 五基线。\n"]
    all_wins = {}
    for b, rows in buckets.items():
        sec.append(f"\n## {TITLES[b]}（{len(rows)} 条线路）\n")
        sec.append(bucket_table(rows))
        ws, _ = win_stats(rows)
        sec.append(f"\n{ws}")
        all_wins[b] = win_stats(rows)
    summary = "\n".join(sec)
    (OUT / "summary_for_docs.md").write_text(summary + "\n", encoding="utf-8")
    print(f"summary -> {OUT / 'summary_for_docs.md'}")

    # ---- 替换文档占位
    fill = {}
    for b, rows in buckets.items():
        block = [f"**{TITLES[b]}**（{len(rows)} 条线路）：\n"]
        block.append(bucket_table(rows))
        ws, _ = win_stats(rows)
        block.append(f"\n{ws}")
        fill[b] = "\n".join(block)

    tables_fill = (f"\n### 分层结果（均值/截尾/中位）\n\n{fill['small']}\n\n"
                   f"{fill['medium']}\n\n{fill['large']}\n\n"
                   f"共 {n_total} 条线路；原始数据逐线路见 tables/final_report/*.jsonl，"
                   f"汇总口径见 tables/final_report/summary_for_docs.md。\n")

    # 结论句：跨桶聚合
    agg = {p: [] for _, p in COLS}
    for rows in buckets.values():
        for _, prefix in COLS:
            agg[prefix].extend(r[prefix + "_fidelity"] for r in rows
                               if isinstance(r.get(prefix + "_fidelity"),
                                             (int, float)))
    agg = {k: v for k, v in agg.items() if v}
    mq, mo3, mt, mb1 = (statistics.mean(agg["qt"]),
                        statistics.mean(agg["aw_o3"]),
                        statistics.mean(agg["tket"]),
                        statistics.mean(agg["b_o1"]))
    all_rows = [r for rows in buckets.values() for r in rows]
    ok = [r for r in all_rows if all(f"{p}_fidelity" in r
                                     for p in ("qt", "aw_o3", "tket", "b_o1"))]
    w3 = sum(1 for r in ok if r["qt_fidelity"] > r["aw_o3_fidelity"])
    wt = sum(1 for r in ok if r["qt_fidelity"] > r["tket_fidelity"])
    wb = sum(1 for r in ok if r["qt_fidelity"] > r["b_o1_fidelity"])
    summary_fill = (
        f"**最终基准（{n_total} 条线路，QTrial pure fidelity 规则）**："
        f"平均估计保真度 QTrial **{mq:.4f}**、感知 O3 **{mo3:.4f}**"
        f"（QTrial 为其 {mq / max(mo3, 1e-12):.2f}×）、pytket **{mt:.4f}**、"
        f"盲目 O1 **{mb1:.4f}**；逐线路胜场（{len(ok)} 条全完成）："
        f"vs 感知 O3 **{w3}**、vs pytket **{wt}**、vs 盲目 O1 **{wb}**。"
        f"分层明细见 §8.2 与 tables/final_report/。")
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        n1 = text.count("【BENCH_FILL_TABLES】")
        n2 = text.count("【BENCH_FILL_SUMMARY】")
        text = text.replace("【BENCH_FILL_TABLES】", tables_fill)
        text = text.replace("【BENCH_FILL_SUMMARY】", summary_fill)
        if n1 + n2 == 0:
            print(f"[!] {doc.name}: no placeholder found")
        else:
            doc.write_text(text, encoding="utf-8")
            print(f"{doc.name}: tables x{n1}, summary x{n2}")


if __name__ == "__main__":
    build_all()
