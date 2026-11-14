# -*- coding: utf-8 -*-
"""报告图5.9：583 条完成线路的总体保真度与逐线路胜场。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tables" / "submission" / "figs"
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False, "font.size": 10,
    "axes.edgecolor": "#999999", "axes.linewidth": 0.8, "savefig.dpi": 300,
})
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "yellow": "#eda100", "violet": "#6250d6", "red": "#e34948",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e5e5e2"}


def load(p):
    out = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["circuit"]] = r
    return out


def main():
    S = json.loads((ROOT / "tables/submission/summary.json").read_text(encoding="utf-8"))
    methods = [("本方案 v8", C["blue"]), ("噪声感知 O3", C["orange"]),
               ("噪声感知 O1", C["violet"]), ("盲目 O3", C["yellow"]),
               ("盲目 O1", C["red"]), ("pytket", C["aqua"])]
    means = []
    for name, _ in methods:
        num = den = 0
        for b in ("small", "medium", "large"):
            e = S["buckets"][b]["stats"].get(name)
            if e and e["fidelity"]:
                n = e["fidelity"]["n"]
                num += e["fidelity"]["mean"] * n
                den += n
        means.append(num / den if den else 0)

    # 胜场：v8 逐线路 vs 各基线
    v8 = {}
    for b in ("small", "medium", "large"):
        v8.update(load(ROOT / f"tables/final_report_v8/{b}_fidelity_rows.jsonl"))
    base = {}
    for b in ("small", "medium", "large"):
        p = ROOT / f"tables/final_report/{b}_fidelity_rows.jsonl"
        if p.exists():
            base.update(load(p))
    ok = {k: r for k, r in v8.items() if "qt_fidelity" in r}
    wins = []
    for pref in ("aw_o3", "tket", "b_o1"):
        w = n = 0
        for k, r in ok.items():
            b = base.get(k)
            if b and b.get(f"{pref}_fidelity") is not None:
                n += 1
                w += r["qt_fidelity"] > b[f"{pref}_fidelity"]
        wins.append((pref, w, n))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4),
                             gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.3})
    ax = axes[0]
    labels = [m[0] for m in methods]
    bars = ax.bar(range(len(means)), means, width=0.62,
                  color=[m[1] for m in methods], edgecolor="white", linewidth=1.0)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.5, color=C["ink"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("总体估计保真度（583 条加权均值）", fontsize=9.5)
    ax.set_ylim(0, 0.30)
    ax.yaxis.grid(True, color=C["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("本方案超过 pytket 与盲目基线，为感知 O3 的 0.924 倍",
                 fontsize=10.5, loc="left", color=C["ink"])

    ax = axes[1]
    names = {"aw_o3": "vs 感知 O3", "tket": "vs pytket", "b_o1": "vs 盲目 O1"}
    ys = list(range(len(wins)))
    bars = ax.barh(ys, [w for _, w, _ in wins],
                   color=[C["orange"], C["aqua"], C["red"]], height=0.55,
                   edgecolor="white", linewidth=1.0)
    for y, (pref, w, n) in zip(ys, wins):
        ax.text(w + 8, y, f"{w} / {n}（{100*w/n:.0f}%）", va="center",
                fontsize=9, color=C["ink"])
    ax.set_yticks(ys)
    ax.set_yticklabels([names[p] for p, _, _ in wins], fontsize=9.5)
    ax.set_xlim(0, 650)
    ax.set_xlabel("逐线路胜场数", fontsize=9.5)
    ax.xaxis.grid(True, color=C["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("逐线路胜场：对 pytket 469 场、对盲目 O1 531 场",
                 fontsize=10.5, loc="left", color=C["ink"])
    fig.savefig(OUT / "fig5_9_overall.png", bbox_inches="tight", facecolor="white")
    print("written", OUT / "fig5_9_overall.png")


if __name__ == "__main__":
    main()
