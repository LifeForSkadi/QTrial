# -*- coding: utf-8 -*-
"""真机对比图（v8 vs Qiskit O3）：实测保真度 / 2Q 门数 / 预测-实测。

风格与 scripts/make_report_figures.py 一致（固定配色、单纵轴、
柱顶标值、中文微软雅黑）。数据：tables/real_machine_v9/rows.json。

用法：python scripts/make_real_machine_v9_figure.py
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
    plt.rcParams["figure.facecolor"] = SURFACE
    plt.rcParams["axes.facecolor"] = SURFACE
    plt.rcParams["axes.edgecolor"] = BASELINE
    plt.rcParams["axes.labelcolor"] = INK2
    plt.rcParams["xtick.color"] = INK2
    plt.rcParams["ytick.color"] = INK2
    plt.rcParams["text.color"] = INK
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = GRID
    plt.rcParams["grid.linewidth"] = 0.6


def nice(name: str) -> str:
    if name.startswith("qft"):
        return f"QFT-{name[3:]}"
    if name.startswith("ghz"):
        return f"GHZ-{name[3:]}"
    return name


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
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=8, color=INK2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([nice(n) for n in names], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, color=INK, pad=8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def main():
    setup()
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    names = sorted({r["circuit"] for r in rows},
                   key=lambda n: (0 if n.startswith("qft") else 1, n))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    grouped(axes[0], rows, names, "meas_fid", "经典保真度（实测）",
            "（a）真机实测经典保真度", "{:.3f}")
    axes[0].set_ylim(0, 1.05)
    grouped(axes[1], rows, names, "2q", "2Q 门数（CZ）",
            "（b）映射后 2Q 门数", "{:.0f}")
    top = max(r["2q"] for r in rows) if rows else 1
    axes[1].set_ylim(0, top * 1.25 + 1)

    ax = axes[2]
    for arm in ("v8", "o3"):
        pts = [r for r in rows if r["arm"] == arm]
        ax.scatter([r["pred_fid"] for r in pts], [r["meas_fid"] for r in pts],
                   s=46, color=COLORS[arm], label=LABELS[arm],
                   edgecolor="white", linewidth=0.8, zorder=3)
    lim = max([r["pred_fid"] for r in rows] +
              [r["meas_fid"] for r in rows] + [0.05]) * 1.15
    ax.plot([0, lim], [0, lim], color=MUTED, linestyle="--", linewidth=1.0,
            zorder=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("乘积模型预测保真度", fontsize=9)
    ax.set_ylabel("实测经典保真度", fontsize=9)
    ax.set_title("（c）预测 vs 实测", fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "图13_真机对比.png"
    fig.savefig(dst, dpi=200, bbox_inches="tight",
                facecolor=SURFACE)
    print("written", dst)


if __name__ == "__main__":
    main()
