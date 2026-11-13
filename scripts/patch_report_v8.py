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
             "盲目O3": "b_o3", "盲目O1": "b_o1", "pytket": "tket"}


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
            cells[2] = f" {fmt(st['qt'])} "
            for j, key in enumerate(COL_ORDER):
                cells[3 + j] = f" {fmt(st[key])} "
            if "tket" in st:  # 第 7 列 pytket（2026-11-13 已补读出）
                cells[7] = f" {fmt(st['tket'])} "
            return "|".join(cells)
        sec = re.sub(r"\| 估计保真度 \|[^\n]*\|", repl_row, sec)

        # 指标行（SWAP/深度/2Q 深度）：QTrial 与 pytket 列同步
        def repl_metric(m):
            cells = m.group(0).split("|")
            label = cells[1].strip()
            key = {"SWAP 数": "swaps", "深度": "depth",
                   "2Q 深度": "twoq_depth"}.get(label)
            if key is None:
                return m.group(0)
            qv = st.get("qt_metrics", {}).get(key)
            if qv:
                cells[2] = " " + "/".join(f"{x:.4f}" for x in qv) + " "
            tv = st.get("tket_metrics", {}).get(key)
            if tv:
                cells[7] = " " + "/".join(f"{x:.4f}" for x in tv) + " "
            return "|".join(cells)
        sec = re.sub(r"\| (?:SWAP 数|2Q 深度|深度) \|[^\n]*\|",
                     repl_metric, sec)
        # 2) 胜场行：vs 感知O3 / vs 盲目O1（pytket 不动）
        def repl_wins(m):
            line = m.group(0)
            for label, key in LABEL2KEY.items():
                if key not in st:
                    continue
                line = re.sub(
                    rf"(vs {label} \*\*)\d+(\*\*)",
                    rf"\g<1>{st[key]['wins']}\g<2>", line)
            return line
        sec = re.sub(r"QTrial 保真度胜场[^\n]*", repl_wins, sec)
        out.append(sec)
    text = "".join(out)
    note = (
        "> **口径修正（2026-11-13）**：此前基线在 transpile 前剥离了 "
        "measure，乘积保真度未计读出项；QTrial 产物带 measure、计了读出。"
        "本表基线列已用 `scripts/fix_baseline_readout.py`（qiskit 四条，"
        "门级逐条复现）与 `scripts/fix_pytket_readout.py`（pytket 保留 "
        "measure 映射、当前版本重跑；旧表部分线路不可复现，见台账 #27）"
        "补回读出。无 measure 的线路两侧均只计门误差。详见 "
        "`summary_v8_corrected.md` 与台账 #24/#27。")
    if "口径修正（2026-11-13）" in text:
        text = re.sub(r"^> \*\*口径修正（2026-11-13）\*\*.*$", note,
                      text, flags=re.M)
    else:
        text = text.replace(
            "统计：均值/截尾(去10%)/中位数。",
            "统计：均值/截尾(去10%)/中位数。\n\n" + note)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path} (backup {backup})")


if __name__ == "__main__":
    main()
