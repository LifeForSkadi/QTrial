# -*- coding: utf-8 -*-
"""真机对比图（v8 vs Qiskit O3，2×2）：实测保真度 / GHZ 规模 / 2Q 门数 /
预测-实测校验。风格与 scripts/make_report_figures.py 一致。

数据：tables/real_machine_v9/rows.json
输出：tables/report_figures/图13_真机对比.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
COLORS = {"v8": "#2a78d6", "o3": "#eb6834"}
LABELS = {"v8": "QTrial v8", "o3": "Qiskit 感知 O3"}
OUT = Path("tables/report_figures")
DATA = Path("tables/real_machine_v9/rows.json")


def setup():
    for name in ("Microsoft YaHei", "SimHei", "PingFang SC"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    })


def nice(name: str) -> str:
    if name.startswith("qft"):
        return f"QFT-{name[3:]}"
    if name.startswith("ghz"):
        return f"GHZ-{name[3:]}"
    if name.startswith("qaoa"):
        return f"QAOA-{name[4:]}"
    return name.upper()


def grouped(ax, rows, names, key, ylabel, title, fmt="{:.3f}"):
    x = np.arange(len(names))
    w = 0.36
    for k, arm in enumerate(("v8", "o3")):
        vals = []
        for nm in names:
            r = next((r for r in rows if r["circuit"] == nm
                      and r["arm"] == arm), None)
            vals.append(r[key] if r else 0.0)
        bars = ax.bar(x + (k - 0.5) * w, vals, w, color=COLORS[arm],
                      label=LABELS[arm], edgecolor="white", linewidth=0.6,
                      zorder=3)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        fmt.format(v), ha="center", va="bottom",
                        fontsize=7.5, color=INK2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([nice(n) for n in names], fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, color=INK, pad=8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def main():
    setup()
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    names = sorted({r["circuit"] for r in rows},
                   key=lambda n: (0 if n.startswith("qft") else
                                  1 if n.startswith("ghz") else 2, n))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))

    # (a) 实测经典保真度
    grouped(axes[0][0], rows, names, "meas_fid",
            "经典保真度（实测）", "（a）真机实测经典保真度")
    axes[0][0].set_ylim(0, 1.08)
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper right")

    # (b) GHZ 规模扩展：命中率 vs n + 预测曲线
    ax = axes[0][1]
    ghz = sorted([r for r in rows if r["circuit"].startswith("ghz")
                  and r["arm"] == "v8"], key=lambda r: r["n"])
    xs = [r["n"] for r in ghz]
    ys = [r["hit"] if r["hit"] is not None else 0.0 for r in ghz]
    ps = [r["pred_fid"] for r in ghz]
    ax.plot(xs, ps, color=MUTED, linestyle="--", linewidth=1.2,
            marker="", label="乘积模型预测", zorder=2)
    ax.plot(xs, ys, color=COLORS["v8"], linewidth=1.6, marker="o",
            markersize=7, markerfacecolor="white", markeredgewidth=1.6,
            label="实测（理想态命中率）", zorder=3)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.03, f"{y:.3f}", ha="center", va="bottom",
                fontsize=8, color=INK2)
    ax.set_xlabel("GHZ 比特数", fontsize=9)
    ax.set_ylabel("概率", fontsize=9)
    ax.set_title("（b）GHZ 规模扩展：预测 vs 实测", fontsize=10,
                 color=INK, pad=8)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # (c) 2Q 门数
    grouped(axes[1][0], rows, names, "2q", "2Q 门数（CZ）",
            "（c）映射后 2Q 门数", "{:.0f}")
    top = max(r["2q"] for r in rows) if rows else 1
    axes[1][0].set_ylim(0, top * 1.3 + 1)

    # (d) 预测 vs 实测（峰态电路）
    ax = axes[1][1]
    for arm in ("v8", "o3"):
        pts = [r for r in rows if r["arm"] == arm and r["hit"] is not None]
        if not pts:
            continue
        ax.scatter([r["pred_fid"] for r in pts],
                   [r["hit"] for r in pts], s=52, color=COLORS[arm],
                   label=LABELS[arm], edgecolor="white", linewidth=0.9,
                   zorder=3)
    lim = max([r["pred_fid"] for r in rows if r["hit"] is not None] +
              [r["hit"] for r in rows if r["hit"] is not None] + [0.1]) * 1.18
    ax.plot([0, lim], [0, lim], color=MUTED, linestyle="--", linewidth=1.0,
            zorder=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, min(1.05, lim))
    ax.set_xlabel("乘积模型预测保真度", fontsize=9)
    ax.set_ylabel("实测理想态命中率", fontsize=9)
    ax.set_title("（d）预测 vs 实测（峰态电路）", fontsize=10, color=INK,
                 pad=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("图 13  天衍真机对比：QTrial v8 vs 感知 O3（tianyan176）",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "图13_真机对比.png"
    fig.savefig(dst, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print("written", dst)


if __name__ == "__main__":
    main()
