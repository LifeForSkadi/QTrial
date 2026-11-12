# -*- coding: utf-8 -*-
"""真机结果判定：x5 / ghz5 / qft5（tianyan176，2000 shots）。

判据：
  x5   理想单态 (10101) 命中率（含读出误差）
  ghz5 理想两态 {00000,11111} 命中率 + TVD
  qft5 均匀分布（32 态）的 TVD / 经典保真度
结果位序 = QCIS M 指令顺序 = 原线路 measure 顺序（x5 已验证）。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path("tables/real_machine_v8")
IDEAL = {
    "x5": {(1, 0, 1, 0, 1)},
    "ghz5": {(0, 0, 0, 0, 0), (1, 1, 1, 1, 1)},
    "ghz10": {tuple([0] * 10), tuple([1] * 10)},
}


def load_counts(name: str):
    raw = json.loads((OUT / f"{name}_raw.json").read_text(encoding="utf-8"))
    item = raw[0] if isinstance(raw, list) else raw
    rows = item["resultStatus"]
    labels = rows[0]
    counts = Counter(tuple(int(x) for x in r) for r in rows[1:])
    return labels, counts


def metrics(counts: Counter, ideal, n_states: int):
    total = sum(counts.values())
    hit = sum(counts.get(k, 0) for k in ideal) / max(total, 1)
    # TVD vs 理想（ideal 上均匀，其余为 0）
    tvd = 0.5 * sum(abs(counts.get(k, 0) / max(total, 1) -
                        (1.0 / n_states if k in ideal else 0.0))
                    for k in set(counts) | set(ideal))
    # 经典保真度（对理想均匀分布）
    p = np.array([counts.get(k, 0) / max(total, 1) for k in ideal])
    q = np.full(len(ideal), 1.0 / n_states)
    fid = float((np.sqrt(p * q).sum()) ** 2) if len(ideal) else 0.0
    return total, hit, tvd, fid


def main():
    table, diag = [], []
    lines = ["# 天衍真机验证（tianyan176，2026-11-13）\n\n",
             "自研 v8 管线映射 → QCIS → 平台 `submit_job(is_verify=False)` "
             "真机执行。tianyan-287 因账号机时不足（paid）未能执行；"
             "tianyan176 为免费 running 真机（66 比特）。\n\n",
             "| 线路 | 测量物理位 | shots | 理想态命中率 | TVD | 经典保真度 |\n",
             "|---|---|---|---|---|---|\n"]
    for name in ("x5", "ghz5", "ghz10", "qft5"):
        labels, counts = load_counts(name)
        if name == "qft5":
            ideal = {tuple((i >> b) & 1 for b in range(5)) for i in range(32)}
            n_states = 32
        else:
            ideal = IDEAL[name]
            n_states = len(ideal)
        total, hit, tvd, fid = metrics(counts, ideal, n_states)
        print(f"{name}: labels={labels} shots={total} hit={hit:.4f} "
              f"tvd={tvd:.4f} fid={fid:.4f}", flush=True)
        hit_s = "—(全态)" if name == "qft5" else f"{hit:.4f}"
        table.append(f"| {name} | {labels} | {total} | {hit_s} | "
                     f"{tvd:.4f} | {fid:.4f} |\n")
        diag.append(f"\n{name} top: {counts.most_common(5)}\n")
    lines += table + diag
    lines.append("\n注：x5 理想单态命中率 ≈ 0.9^5（读出误差约 10%/比特），"
                 "GHZ 命中率受读出+退相干共同限制；QFT5 以 TVD/经典保真度"
                 "对照均匀分布。287 真机待账号机时到位后按同一脚本补跑。\n")
    (OUT / "summary.md").write_text("".join(lines), encoding="utf-8")
    print("written", OUT / "summary.md")


if __name__ == "__main__":
    main()
