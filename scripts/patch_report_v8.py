# -*- coding: utf-8 -*-
"""把修正口径（基线补回读出）写回 v8 分层报告表。

只改 `估计保真度` 行与胜场行里的感知O3/感知O1/盲目O3/盲目O1 四列，
QTrial 与 pytket 列不动（pytket 布局未落盘、未修正，保持保守）。
原文件备份为 `report_pre_readout_fix.md`（只备份一次）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path("tables/final_report_v8")
TITLES = {"small": "≤10 比特", "medium": "11-50 比特",
          "large": "51-105 比特"}
COLS = ["b_o1", "b_o3", "aw_o1", "aw_o3"]  # 表中列序：感知O3 感知O1 盲目O3 盲目O1
COL_ORDER = ["aw_o3", "aw_o1", "b_o3", "b_o1"]
LABEL2KEY = {"感知O3": "aw_o3", "感知O1": "aw_o1",
             "盲目O3": "b_o3", "盲目O1": "b_o1"}


def fmt(st):
    return (f"{st['mean_fidelity']:.4f}/{st['trim_fidelity']:.4f}/"
            f"{st['median_fidelity']:.4f}")


def main():
    stats = json.loads((OUT / "corrected_stats.json").read_text(
        encoding="utf-8"))
    path = OUT / "report.md"
    text = path.read_text(encoding="utf-8")
    backup = OUT / "report_pre_readout_fix.md"
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    sections = re.split(r"(?=\n## )", text)
    out = [sections[0]]
    for sec in sections[1:]:
        bucket = next((b for b, t in TITLES.items()
                       if sec.startswith(f"\n## {t}")), None)
        if bucket is None or bucket not in stats:
            out.append(sec)
            continue
        st = stats[bucket]
        # 1) 保真度行：保留 QTrial 与 pytket 列
        def repl_row(m):
            cells = m.group(0).split("|")
            # cells: ['', ' 估计保真度 ', ' qt ', ' aw_o3 ', ... ' pytket ', '']
            for j, key in enumerate(COL_ORDER):
                cells[3 + j] = f" {fmt(st[key])} "
            return "|".join(cells)
        sec = re.sub(r"\| 估计保真度 \|[^\n]*\|", repl_row, sec)
        # 2) 胜场行：vs 感知O3 / vs 盲目O1（pytket 不动）
        def repl_wins(m):
            line = m.group(0)
            for label, key in LABEL2KEY.items():
                if label == "pytket":
                    continue
                line = re.sub(
                    rf"(vs {label} \*\*)\d+(\*\*)",
                    rf"\g<1>{st[key]['wins']}\g<2>", line)
            return line
        sec = re.sub(r"QTrial 保真度胜场[^\n]*", repl_wins, sec)
        out.append(sec)
    text = "".join(out)
    note = (
        "\n> **口径修正（2026-11-13）**：此前基线在 transpile 前剥离了 "
        "measure，乘积保真度未计读出项；QTrial 产物带 measure、计了读出。"
        "本表基线列已用 `scripts/fix_baseline_readout.py` 按原线路实际"
        "测量的逻辑比特补回读出（门级保真度逐条复现旧表）。无 measure "
        "的线路两侧均只计门误差。pytket 列未修正（布局未落盘），"
        "其对比仍偏保守。详见 `summary_v8_corrected.md` 与台账 #24。\n")
    if "口径修正（2026-11-13）" not in text:
        text = text.replace(
            "统计：均值/截尾(去10%)/中位数。",
            "统计：均值/截尾(去10%)/中位数。" + note)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path} (backup {backup})")


if __name__ == "__main__":
    main()
