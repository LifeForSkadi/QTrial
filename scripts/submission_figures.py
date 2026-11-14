# -*- coding: utf-8 -*-
"""决赛提交版图件：管线示意图 + 分层结果图（PPT 简版 + 报告详版）。

数据来源：tables/submission/summary.json
输出：tables/submission/figs/*.png / *.svg
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tables" / "submission" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.edgecolor": "#999999",
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
})

# dataviz 验证调色板（前 4 槽，相邻对通过 CVD/常规视觉门槛）
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_VIOLET, C_RED = "#6250d6", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e5e2"

# Okabe-Ito（示意图）
OI_BLUE, OI_ORANGE, OI_GREEN, OI_VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
OI_GREY = "#4D4D4D"

S = json.loads((ROOT / "tables" / "submission" / "summary.json").read_text(encoding="utf-8"))
B = S["buckets"]


def val(bucket, method, metric):
    m = B[bucket]["stats"][method][metric]
    return m["mean"] if m else None


# ---------------------------------------------------------------- 管线示意图
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(13.33, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    def box(x, y, w, h, text, color, fs=10, fc="white", lw=1.6, bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                           facecolor=fc, edgecolor=color, linewidth=lw)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, fontweight="bold" if bold else "normal",
                linespacing=1.5)

    def arrow(x1, y1, x2, y2, color=OI_GREY, label=None, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, color=color, linewidth=1.3,
                                     linestyle=ls))

    # 输入
    box(1.2, 21, 14, 8, "QASM 2.0 / 量子线路", OI_GREY, fs=11)
    box(1.2, 9, 14, 8, "天衍-287 拓扑+校准", OI_GREY, fs=11)

    # 阶段 1
    ax.text(28, 36.0, "阶段一  强化学习布局", ha="center", fontsize=11.5,
            color=OI_BLUE, fontweight="bold")
    box(19, 17, 18, 13.5, "", OI_BLUE, lw=1.8)
    ax.text(28, 27.0, "GAT 程序图编码", ha="center", fontsize=9.5, color=INK)
    ax.text(28, 24.0, "设备图 + 噪声特征", ha="center", fontsize=9.5, color=INK)
    ax.text(28, 21.0, "指针解码 → 候选布局", ha="center", fontsize=9.5, color=INK)
    ax.text(28, 18.6, "REINFORCE · 多起点搜索", ha="center", fontsize=8.5,
            color=INK2, style="italic")

    # 阶段 2
    ax.text(54, 36.0, "阶段二  SWIFT 路由内核", ha="center", fontsize=11.5,
            color=OI_ORANGE, fontweight="bold")
    box(45, 17, 18, 13.5, "", OI_ORANGE, lw=1.8)
    ax.text(54, 27.0, "前沿增量维护", ha="center", fontsize=9.5, color=INK)
    ax.text(54, 24.0, "噪声感知候选评分", ha="center", fontsize=9.5, color=INK)
    ax.text(54, 21.0, "SWAP 插入 / 门级调度", ha="center", fontsize=9.5, color=INK)
    ax.text(54, 18.6, "任意拓扑 · 增量内核", ha="center", fontsize=8.5,
            color=INK2, style="italic")

    # 阶段 3
    ax.text(80, 36.0, "阶段三  置换重写后处理", ha="center", fontsize=11.5,
            color=OI_GREEN, fontweight="bold")
    box(71, 17, 18, 13.5, "", OI_GREEN, lw=1.8)
    ax.text(80, 27.0, "VW 闭式块合成", ha="center", fontsize=9.5, color=INK)
    ax.text(80, 24.0, "1Q 游程合并 / 消解", ha="center", fontsize=9.5, color=INK)
    ax.text(80, 21.0, "连通性感知推挤", ha="center", fontsize=9.5, color=INK)
    ax.text(80, 18.6, "零残留 SWAP · 硬件可执行", ha="center", fontsize=8.5,
            color=INK2, style="italic")

    # 输出
    box(91.5, 17, 7.5, 13.5, "QCIS" + chr(10) + "可执行" + chr(10) + "线路", OI_VERM, fs=10.5, bold=True)
    arrow(15.4, 13, 18.6, 20)
    arrow(37.3, 23.7, 44.6, 23.7)
    arrow(63.3, 23.7, 70.6, 23.7)
    arrow(89.3, 23.7, 91.2, 23.7)
    ax.text(41, 25.1, "候选布局", fontsize=8.5, color=INK2, ha="center")
    ax.text(67, 25.1, "路由后线路", fontsize=8.5, color=INK2, ha="center")

    # 底部闭环：验证
    ax.plot([1.2, 98], [5.2, 5.2], color=GRID, lw=1.2)
    ax.text(50, 3.0, "验证闭环：232 项测试  ·  31/31 真机合法（门集/相邻/禁用比特）  ·  "
                     "measure 语义 31/31  ·  端到端语义保真度 1.000000",
            ha="center", fontsize=9.5, color=INK2)
    ax.text(50, 1.0, "映射内核与 CLI 零 qiskit 依赖 · 任意设备拓扑可执行 · 全流程可复现",
            ha="center", fontsize=9, color=OI_BLUE)

    fig.savefig(OUT / "fig_pipeline.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_pipeline.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------ PPT 简版：3 方法 2 面板
def draw_ppt_results():
    methods = [("本方案 v8", C_BLUE), ("噪声感知 O3", C_ORANGE), ("pytket", C_AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.2), gridspec_kw={"wspace": 0.22})

    # 面板 A：保真度（≤10 / 11-50）
    ax = axes[0]
    labels = ["≤10 比特", "11-50 比特"]
    keys = ["small", "medium"]
    x = range(len(labels))
    w = 0.24
    for i, (name, color) in enumerate(methods):
        vals = [val(k, name, "fidelity") for k in keys]
        bars = ax.bar([xi + (i - 1) * w for xi in x], vals, width=w * 0.92,
                      color=color, label=name, edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("估计保真度（逐门乘积代理）", fontsize=10)
    ax.set_ylim(0, 0.92)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_title("小规模保真度接近感知 O3，中规模仍有差距", fontsize=11.5,
                 loc="left", color=INK)
    ax.text(0.0, -0.17, "注：51-105 比特层乘积代理下溢，见右图 2Q 深度。",
            transform=ax.transAxes, fontsize=8.5, color=INK2)

    # 面板 B：2Q 深度（三层）
    ax = axes[1]
    labels = ["≤10", "11-50", "51-105"]
    keys = ["small", "medium", "large"]
    x = range(len(labels))
    for i, (name, color) in enumerate(methods):
        vals = [val(k, name, "twoq_depth") for k in keys]
        bars = ax.bar([xi + (i - 1) * w for xi in x], vals, width=w * 0.92,
                      color=color, label=name, edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.04, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("2Q 深度（均值，越低越好）", fontsize=10)
    ax.set_ylim(0, 3500)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("大线路 2Q 深度与 pytket 持平", fontsize=11.5,
                 loc="left", color=INK)
    ax.text(0.0, -0.17, "注：均值口径；感知 O3 仍是大线路最强基线（差距如实披露）。",
            transform=ax.transAxes, fontsize=8.5, color=INK2)

    fig.savefig(OUT / "fig_results_ppt.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_results_ppt.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------- 视频/PPT 深色版（暗色表面）
def draw_ppt_results_dark():
    """深色背景版（用于视频/深色 PPT 页）；色槽取 dataviz 暗色验证值。"""
    d_blue, d_orange, d_aqua = "#3987e5", "#d95926", "#199e70"
    d_ink, d_ink2, d_grid, d_surface = "#f0efec", "#c3c2b7", "#2a3342", "#0e1117"
    methods = [("本方案 v8", d_blue), ("噪声感知 O3", d_orange), ("pytket", d_aqua)]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.2), gridspec_kw={"wspace": 0.22})
    fig.patch.set_facecolor(d_surface)
    for ax in axes:
        ax.set_facecolor(d_surface)
        ax.tick_params(colors=d_ink2, labelsize=11.5)
        ax.yaxis.label.set_color(d_ink2)
        ax.xaxis.label.set_color(d_ink2)
        for sp in ax.spines.values():
            sp.set_color(d_grid)

    ax = axes[0]
    labels, keys = ["≤10 比特", "11-50 比特"], ["small", "medium"]
    x = range(len(labels))
    w = 0.24
    for i, (name, color) in enumerate(methods):
        vals = [val(k, name, "fidelity") for k in keys]
        bars = ax.bar([xi + (i - 1) * w for xi in x], vals, width=w * 0.92,
                      color=color, label=name, edgecolor=d_surface, linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10.5, color=d_ink)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("估计保真度（逐门乘积代理）", fontsize=10)
    ax.set_ylim(0, 0.92)
    ax.yaxis.grid(True, color=d_grid, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    leg = ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(d_ink)
    ax.set_title("小规模保真度接近感知 O3，中规模仍有差距", fontsize=12.5,
                 loc="left", color=d_ink)
    ax.text(0.0, -0.17, "注：51-105 比特层乘积代理下溢，见右图 2Q 深度。",
            transform=ax.transAxes, fontsize=9.5, color=d_ink2)

    ax = axes[1]
    labels, keys = ["≤10", "11-50", "51-105"], ["small", "medium", "large"]
    x = range(len(labels))
    for i, (name, color) in enumerate(methods):
        vals = [val(k, name, "twoq_depth") for k in keys]
        bars = ax.bar([xi + (i - 1) * w for xi in x], vals, width=w * 0.92,
                      color=color, label=name, edgecolor=d_surface, linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.04, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=10.5, color=d_ink)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("2Q 深度（均值，越低越好）", fontsize=10)
    ax.set_ylim(0, 3500)
    ax.yaxis.grid(True, color=d_grid, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("大线路 2Q 深度与 pytket 持平", fontsize=12.5, loc="left", color=d_ink)
    ax.text(0.0, -0.17, "注：均值口径；感知 O3 仍是大线路最强基线（差距如实披露）。",
            transform=ax.transAxes, fontsize=9.5, color=d_ink2)
    fig.savefig(OUT / "fig_results_ppt_dark.png", bbox_inches="tight",
                facecolor=d_surface)
    plt.close(fig)


# ---------------------------------------------------- 报告详版：6 方法 2 面板
def draw_report_results():
    methods = [("本方案 v8", C_BLUE), ("噪声感知 O3", C_ORANGE),
               ("盲目 O3", C_YELLOW), ("pytket", C_AQUA),
               ("噪声感知 O1", C_VIOLET), ("盲目 O1", C_RED)]
    keys = ["small", "medium", "large"]
    labels = ["≤10 比特", "11-50 比特", "51-105 比特"]

    for metric, ylabel, fname, title, ymax in (
        ("fidelity", "估计保真度（逐门乘积代理）", "fig5_2_fidelity.png",
         "分层估计保真度对比（均值；51-105 层乘积下溢为 0）", 0.85),
        ("twoq_depth", "2Q 深度（均值，越低越好）", "fig5_3_twoq_depth.png",
         "分层 2Q 深度对比（均值）", 3500),
        ("depth", "线路深度（均值，越低越好）", "fig5_4_depth.png",
         "分层线路深度对比（均值）", 9500),
    ):
        fig, ax = plt.subplots(figsize=(11.5, 4.6))
        x = range(len(keys))
        w = 0.135
        for i, (name, color) in enumerate(methods):
            vals = [val(k, name, metric) for k in keys]
            bars = ax.bar([xi + (i - 2.5) * w for xi in x], vals, width=w * 0.9,
                          color=color, label=name, edgecolor="white", linewidth=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + ymax * 0.015, f"{v:.2f}" if metric == "fidelity" else f"{v:.0f}",
                        rotation=90, ha="center", va="bottom", fontsize=6.3, color=INK)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, ymax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, ncol=6, loc="upper center",
                  bbox_to_anchor=(0.5, -0.14))
        ax.set_title(title, fontsize=11, loc="left", color=INK, pad=10)
        fig.savefig(OUT / fname, bbox_inches="tight", facecolor="white")
        fig.savefig(OUT / fname.replace(".png", ".svg"), bbox_inches="tight", facecolor="white")
        plt.close(fig)


# ---------------------------------------------------------------- 审计数字卡
def draw_audit_tiles():
    tiles = [
        ("31 / 31", "抽样线路真机合法", "门集·相邻·禁用比特·无 swap 残留"),
        ("31 / 31", "measure 语义一致", "测量位与 final_layout 对齐"),
        ("1.000000", "端到端语义保真度", "k≤16 紧凑子空间态矢量，15 条"),
        ("232", "自动化测试通过", "另 2 项为已知环境性断言"),
    ]
    fig, ax = plt.subplots(figsize=(13.33, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.axis("off")
    for i, (big, mid, small) in enumerate(tiles):
        x = 2 + i * 24.5
        ax.add_patch(FancyBboxPatch((x, 2), 22, 16, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    facecolor="#F7F7F7", edgecolor="#CCCCCC", linewidth=1.0))
        ax.text(x + 11, 13.2, big, ha="center", fontsize=22, color=OI_BLUE, fontweight="bold")
        ax.text(x + 11, 7.6, mid, ha="center", fontsize=11, color=INK)
        ax.text(x + 11, 4.2, small, ha="center", fontsize=8, color=INK2)
    fig.savefig(OUT / "fig_audit_tiles.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_audit_tiles.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)




# --------------------------------------------------- 管线图（视频深色版）
def draw_pipeline_dark():
    """与 draw_pipeline 同版式，深色表面 + 亮色文字，供视频/深色页使用。"""
    surface, panel, border = "#0e1117", "#161b24", "#2a3342"
    ink, ink2 = "#f0efec", "#8b98ab"
    blue, orange, green, verm = "#4cc2ff", "#ffb454", "#7ee081", "#ff6b6b"
    fig, ax = plt.subplots(figsize=(13.33, 5.6))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    def box(x, y, w, h, text, color, fs=10, bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                           facecolor=panel, edgecolor=color, linewidth=1.8)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=ink, fontweight="bold" if bold else "normal",
                linespacing=1.5)

    def arrow(x1, y1, x2, y2, color=ink2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, color=color, linewidth=1.3))

    box(1.2, 21, 14, 8, "QASM 2.0 / 量子线路", ink2, fs=11)
    box(1.2, 9, 14, 8, "天衍-287 拓扑+校准", ink2, fs=11)
    ax.text(28, 36.0, "阶段一  强化学习布局", ha="center", fontsize=11.5,
            color=blue, fontweight="bold")
    box(19, 17, 18, 13.5, "", blue)
    for yy, tx in ((27.0, "GAT 程序图编码"), (24.0, "设备图 + 噪声特征"),
                   (21.0, "指针解码 → 候选布局")):
        ax.text(28, yy, tx, ha="center", fontsize=9.5, color=ink)
    ax.text(28, 18.6, "REINFORCE · 多起点搜索", ha="center", fontsize=8.5,
            color=ink2, style="italic")
    ax.text(54, 36.0, "阶段二  SWIFT 路由内核", ha="center", fontsize=11.5,
            color=orange, fontweight="bold")
    box(45, 17, 18, 13.5, "", orange)
    for yy, tx in ((27.0, "前沿增量维护"), (24.0, "噪声感知候选评分"),
                   (21.0, "SWAP 插入 / 门级调度")):
        ax.text(54, yy, tx, ha="center", fontsize=9.5, color=ink)
    ax.text(54, 18.6, "任意拓扑 · 增量内核", ha="center", fontsize=8.5,
            color=ink2, style="italic")
    ax.text(80, 36.0, "阶段三  置换重写后处理", ha="center", fontsize=11.5,
            color=green, fontweight="bold")
    box(71, 17, 18, 13.5, "", green)
    for yy, tx in ((27.0, "VW 闭式块合成"), (24.0, "1Q 游程合并 / 消解"),
                   (21.0, "连通性感知推挤")):
        ax.text(80, yy, tx, ha="center", fontsize=9.5, color=ink)
    ax.text(80, 18.6, "零残留 SWAP · 硬件可执行", ha="center", fontsize=8.5,
            color=ink2, style="italic")
    box(91.5, 17, 7.5, 13.5, "QCIS" + chr(10) + "可执行" + chr(10) + "线路", verm, fs=10.5, bold=True)
    arrow(15.4, 13, 18.6, 20)
    arrow(37.3, 23.7, 44.6, 23.7)
    arrow(63.3, 23.7, 70.6, 23.7)
    arrow(89.3, 23.7, 91.2, 23.7)
    ax.text(41, 25.1, "候选布局", fontsize=8.5, color=ink2, ha="center")
    ax.text(67, 25.1, "路由后线路", fontsize=8.5, color=ink2, ha="center")
    ax.plot([1.2, 98], [5.2, 5.2], color=border, lw=1.2)
    ax.text(50, 3.0, "验证闭环：232 项测试  ·  31/31 真机合法  ·  measure 语义 31/31  ·  端到端保真度 1.000000",
            ha="center", fontsize=9.5, color=ink2)
    ax.text(50, 1.0, "映射内核与 CLI 零 qiskit 依赖 · 任意设备拓扑可执行 · 全流程可复现",
            ha="center", fontsize=9, color=blue)
    fig.savefig(OUT / "fig_pipeline_dark.png", bbox_inches="tight",
                facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    draw_pipeline()
    draw_pipeline_dark()
    draw_ppt_results()
    draw_ppt_results_dark()
    draw_report_results()
    draw_audit_tiles()
    print("figures written to", OUT)
