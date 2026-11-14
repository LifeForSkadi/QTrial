# -*- coding: utf-8 -*-
"""Generate figures for the QTrial competition PPT.

All figures use the visual palette of the supplied template
(navy #002060, red #C00000, yellow #FFC000, blue #006CC4).
Outputs go to assets/ppt_figs/.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "ppt_figs"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#002060"
RED = "#C00000"
YELLOW = "#FFC000"
BLUE = "#006CC4"
LIGHT = "#F3F7FB"
GRID = "#D9E2F3"
GRAY = "#595959"
MID = "#7F7F7F"
TEAL = "#2A9D8F"
TAN = "#C9A66B"
MAUVE = "#A67C8A"
WHITE = "#FFFFFF"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def rounded(ax, x, y, w, h, fc=LIGHT, ec=GRID, lw=1.2, rad=0.04, z=1):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(p)
    return p


# ---------------------------------------------------------------- problem
def fig_problem():
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=300)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # left: logical circuit
    rounded(ax, 0.05, 0.25, 3.35, 3.45)
    ax.text(1.72, 3.48, "逻辑线路", ha="center", va="center",
            fontsize=12, color=NAVY, fontweight="bold")
    ys = [2.85, 2.15, 1.45]
    for i, y in enumerate(ys):
        ax.plot([0.35, 3.10], [y, y], color=NAVY, lw=1.1, zorder=2)
        ax.text(0.24, y, f"$q_{i}$", ha="right", va="center",
                fontsize=11, color=NAVY)
    # H on q1
    ax.add_patch(Rectangle((0.78, 1.99), 0.34, 0.32, facecolor=WHITE,
                           edgecolor=NAVY, lw=1.1, zorder=3))
    ax.text(0.95, 2.15, "H", ha="center", va="center",
            fontsize=10, color=NAVY, zorder=4)
    # long-range CX q0-q2
    ax.plot([1.95, 1.95], [1.45, 2.85], color=RED, lw=1.6, zorder=3)
    ax.scatter([1.95], [2.85], s=46, color=RED, zorder=4)
    ax.scatter([1.95], [1.45], s=46, facecolors=WHITE, edgecolors=RED,
               linewidth=1.6, zorder=4)
    ax.text(2.12, 2.15, "CX(0,2)", color=RED, fontsize=9.5, va="center")
    ax.text(1.72, 0.55, "远距离双比特门", ha="center", va="center",
            fontsize=9.5, color=RED)

    # arrow
    ax.add_patch(FancyArrowPatch((3.55, 2.15), (4.35, 2.15),
                                 arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color=GRAY))
    ax.text(3.95, 2.32, "布局 + 路由", ha="center", va="bottom",
            fontsize=9.5, color=GRAY)

    # right: 3x3 device grid
    rounded(ax, 4.45, 0.25, 3.45, 3.45)
    ax.text(6.18, 3.48, "物理芯片（3×3 网格）", ha="center", va="center",
            fontsize=11.5, color=NAVY, fontweight="bold")
    xs = [4.95, 5.75, 6.55]
    ys2 = [1.05, 1.85, 2.65]
    for y in ys2:
        ax.plot([4.95, 6.55], [y, y], color=GRID, lw=2.0, zorder=1)
    for x in xs:
        ax.plot([x, x], [1.05, 2.65], color=GRID, lw=2.0, zorder=1)
    for r, y in enumerate(ys2):
        for c, x in enumerate(xs):
            ax.add_patch(Circle((x, y), 0.13, facecolor=WHITE,
                                edgecolor="#8EA9C1", lw=1.1, zorder=3))
    # mapped logical qubits
    ax.add_patch(Circle((4.95, 1.05), 0.16, facecolor=BLUE,
                        edgecolor=BLUE, zorder=4))
    ax.text(4.95, 1.05, "$q_0$", ha="center", va="center",
            fontsize=8.5, color=WHITE, zorder=5)
    ax.add_patch(Circle((6.55, 2.65), 0.16, facecolor=RED,
                        edgecolor=RED, zorder=4))
    ax.text(6.55, 2.65, "$q_2$", ha="center", va="center",
            fontsize=8.5, color=WHITE, zorder=5)
    # path q0 -> q2 (4 hops)
    path_x = [4.95, 5.75, 5.75, 5.75, 6.55]
    path_y = [1.05, 1.05, 1.85, 2.65, 2.65]
    ax.plot(path_x, path_y, color=RED, lw=1.6, ls=(0, (4, 2)), zorder=2)
    ax.text(5.55, 1.65, "距离 = 4", color=RED, fontsize=9.5,
            rotation=90, va="center", ha="center")

    # bottom note
    ax.text(4.0, 0.02, "SWAP = 3×CX；每多一跳，多 3 个双比特门",
            ha="center", va="bottom", fontsize=9, color=GRAY)
    save(fig, "fig_problem.png")


# ---------------------------------------------------------------- QAP
def fig_qap():
    fig, ax = plt.subplots(figsize=(7.8, 3.9), dpi=300)
    ax.set_xlim(0, 7.8)
    ax.set_ylim(0, 3.9)
    ax.axis("off")

    # left graph
    rounded(ax, 0.05, 0.25, 3.15, 3.45)
    ax.text(1.62, 3.48, "逻辑交互图 $G_C$", ha="center", va="center",
            fontsize=11.5, color=NAVY, fontweight="bold")
    pos = {
        "A": (1.05, 2.55), "B": (2.15, 2.85), "C": (2.65, 1.95),
        "D": (1.75, 1.25), "E": (0.85, 1.65),
    }
    edges = [
        ("A", "B", 8), ("A", "C", 5), ("B", "C", 3),
        ("C", "D", 2), ("D", "E", 1), ("A", "D", 4),
    ]
    for u, v, w in edges:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="#8EA9C1", lw=1.2 + 0.12 * w, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.07, str(w),
                ha="center", va="bottom", fontsize=8.5, color=GRAY,
                bbox=dict(boxstyle="round,pad=0.12", fc=WHITE, ec="none"))
    for k, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.19, facecolor=NAVY,
                            edgecolor=NAVY, zorder=3))
        ax.text(x, y, k, ha="center", va="center", fontsize=10,
                color=WHITE, zorder=4)

    # arrow pi
    ax.add_patch(FancyArrowPatch((3.35, 1.95), (4.20, 1.95),
                                 arrowstyle="-|>", mutation_scale=15,
                                 lw=1.6, color=GRAY))
    ax.text(3.78, 2.12, r"$\pi$", ha="center", va="bottom",
            fontsize=15, color=RED)

    # right grid
    rounded(ax, 4.35, 0.25, 3.35, 3.45)
    ax.text(6.02, 3.48, "设备耦合图 $G_D$", ha="center", va="center",
            fontsize=11.5, color=NAVY, fontweight="bold")
    xs = [4.85, 5.65, 6.45]
    ys = [1.05, 1.85, 2.65]
    for y in ys:
        ax.plot([4.85, 6.45], [y, y], color="#8EA9C1", lw=2.0, zorder=1)
    for x in xs:
        ax.plot([x, x], [1.05, 2.65], color="#8EA9C1", lw=2.0, zorder=1)
    for y in ys:
        for x in xs:
            ax.add_patch(Circle((x, y), 0.13, facecolor=WHITE,
                                edgecolor="#8EA9C1", lw=1.1, zorder=3))
    # occupied nodes
    occupied = [(4.85, 1.05), (5.65, 1.05), (5.65, 1.85), (5.65, 2.65), (6.45, 2.65)]
    for (x, y) in occupied:
        ax.add_patch(Circle((x, y), 0.15, facecolor=BLUE,
                            edgecolor=BLUE, zorder=4))
    ax.text(5.65, 0.72, "5 个逻辑比特 → 5 个物理位置",
            ha="center", va="center", fontsize=9, color=GRAY)

    # formula box
    rounded(ax, 1.15, 0.02, 5.5, 0.62, fc="#FFF7E6", ec=YELLOW, lw=1.4)
    ax.text(3.9, 0.33,
            r"$C(\pi)=2\sum_{i<j} w_{ij}\,d(\pi(i),\pi(j))$",
            ha="center", va="center", fontsize=14, color=NAVY)
    save(fig, "fig_qap.png")


# ---------------------------------------------------------------- overall
def fig_overall():
    fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=300)
    methods = ["QTrial\n(本方案)", "噪声感知\nO3", "pytket", "噪声感知\nO1", "盲目\nO3", "盲目\nO1"]
    vals = [0.226, 0.226, 0.180, 0.158, 0.123, 0.107]
    colors = [NAVY, "#4472C4", MID, TEAL, TAN, MAUVE]
    bars = ax.bar(range(len(vals)), vals, color=colors,
                  width=0.62, zorder=3)
    bars[0].set_edgecolor(RED)
    bars[0].set_linewidth(1.6)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10,
                color=NAVY if i == 0 else GRAY,
                fontweight="bold" if i == 0 else "normal")
    ax.set_ylim(0, 0.265)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9.5, color="#333333")
    ax.set_ylabel("估计保真度（均值）", fontsize=10, color=GRAY)
    ax.tick_params(axis="y", labelsize=9, colors=GRAY)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#BFBFBF")
    ax.set_title("584 条线路 · 统一口径", fontsize=10.5, color=GRAY, pad=8)
    ax.text(0.99, 0.96, "越高越好", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=GRAY)
    save(fig, "fig_overall.png")


# ---------------------------------------------------------------- cover thumbs
def cover_thumbs():
    from PIL import Image, ImageOps, ImageDraw

    # source images already generated / extracted
    srcs = [
        OUT / "fig_problem.png",
        ROOT / "assets" / "existing_ppt" / "slide06_1.png",
        ROOT / "assets" / "existing_ppt" / "slide08_1.png",
        ROOT / "assets" / "existing_ppt" / "slide09_2.png",
        OUT / "fig_overall.png",
        ROOT / "assets" / "existing_ppt" / "slide13_1.png",
    ]
    # widths in inches from the template cover; use 4:3-ish crops
    sizes = [(2.25, 1.63), (2.20, 1.63), (2.46, 1.65),
             (1.35, 1.65), (2.44, 1.64), (2.47, 1.64)]
    for i, (src, (w, h)) in enumerate(zip(srcs, sizes), 1):
        if not src.exists():
            continue
        im = Image.open(src).convert("RGB")
        # card canvas with light blue background and navy border
        W, H = int(w * 300), int(h * 300)
        card = Image.new("RGB", (W, H), "#E9F0F8")
        pad = int(min(W, H) * 0.035)
        inner = ImageOps.contain(im, (W - 2 * pad, H - 2 * pad), Image.LANCZOS)
        x = (W - inner.width) // 2
        y = (H - inner.height) // 2
        card.paste(inner, (x, y))
        draw = ImageDraw.Draw(card)
        draw.rectangle([0, 0, W - 1, H - 1], outline="#002060", width=6)
        card.save(OUT / f"cover_{i}.png")
        print("wrote", OUT / f"cover_{i}.png", card.size)


if __name__ == "__main__":
    fig_problem()
    fig_qap()
    fig_overall()
    cover_thumbs()
