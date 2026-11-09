# -*- coding: utf-8 -*-
"""运行时间折线对比图：本方案布局 + O1 混合 / O3 筛选混合 vs 感知 O1/O3、pytket。

数据源（均为逐线路端到端运行时间，秒）：
  - tables/compare_models/layout_contribution_<bench>_rows.jsonl
    wall_s（本方案布局+SABRE 路由，含 RL+路由+分解）、aw-o1_wall/aw-o3_wall、
    tket_wall
  - tables/compare_models/layout_inject_<bench>_rows.jsonl
    inj_wall（本方案布局+O3 筛选管线，含候选池+SABRE 预筛+O3 试探+红线兜底）

风格与报告图表系列一致（固定配色、单纵轴、对数刻度）；盲目 O1/O3 与感知档
共用同一编译管线（仅布局评分输入不同），耗时相当，不再单列。

用法：
  python scripts/make_runtime_compare.py                  # 图11：按比特数排序
  python scripts/make_runtime_compare.py --sort swaps     # 图12：按 SWAP 门数排序
  python scripts/make_runtime_compare.py --bench scaling  # 90 条规模-难度网格单面板
输出：tables/report_figures/图11_运行时间折线对比.png（或 图12_…）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_report_figures import COLORS, INK, INK2, MUTED, GRID, SURFACE, setup_fonts

BENCHES = [("benchpress", "Benchpress（30 条）", 30),
           ("mqtbench_n15", "MQTBench n15（24 条）", 24),
           ("queko_BIGD", "QUEKO BIGD（30 条）", 30)]

# 系列：标签 / (文件名, 字段) / 颜色
SERIES = [
    ("自研方案（QTrial 全管线）", ("layout_pure", "wall_s"), "#1d4e89"),
    ("本方案布局 + O1 路由", ("layout_contribution", "wall_s"),
     COLORS["QTrial"]),
    ("本方案布局 + O3 路由（筛选）", ("layout_inject", "inj_wall"), "#7ba8e0"),
    ("噪声感知 O3", ("layout_contribution", "aw-o3_wall"), COLORS["感知O3"]),
    ("噪声感知 O1", ("layout_contribution", "aw-o1_wall"), COLORS["感知O1"]),
    ("pytket", ("layout_contribution", "tket_wall"), COLORS["pytket"]),
]


def load(file_key: str, bench: str) -> dict:
    p = Path(f"tables/compare_models/{file_key}_{bench}_rows.jsonl")
    if not p.exists():
        return {}
    return {json.loads(l)["circuit"]: json.loads(l)
            for l in open(p, encoding="utf-8") if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", choices=("qubits", "swaps"), default="qubits",
                    help="x 轴排序键：比特数（图11）或 SWAP 门数（图12）")
    ap.add_argument("--bench", choices=("three", "scaling"), default="three",
                    help="three=原三基准三面板；scaling=90 条规模-难度网格单面板")
    ap.add_argument("--fig-no", default=None,
                    help="图中编号（默认 11/12；thesis 版可传 5.6/5.7）")
    ap.add_argument("--out", default=None, help="输出路径（默认按图号）")
    args = ap.parse_args()
    setup_fonts()
    if args.bench == "scaling":
        benches = [("scaling", "规模-难度网格测试集（90 条，天衍-287）", 90)]
        # 竖版文档适配：缩短横轴、拉高纵轴
        fig, axes = plt.subplots(1, 1, figsize=(8, 5.2))
        axes = [axes]
    else:
        benches = BENCHES
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    for ax, (bench, blabel, _n) in zip(axes, benches):
        rows = {}
        for fk in {fk for _, (fk, _), _ in SERIES} | {"layout_trivial"}:
            rows[fk] = load(fk, bench)
        # x 轴线路：以 layout_contribution 的线路为准（inject 同名对齐）
        names = list(rows["layout_contribution"].keys())
        if not names:
            names = list(rows["layout_inject"].keys())
        if args.sort == "swaps":
            # 排序键 = 优化前 SWAP 门数（平凡布局 + SABRE 路由，零布局
            # 优化——线路固有路由难度；缺省回退比特数）
            names.sort(key=lambda n: (rows["layout_trivial"].get(
                n, {}).get("trivial_swaps", 10 ** 9),
                rows["layout_trivial"].get(n, {}).get("n_qubits", 0), n))
            xlabel = "线路（按优化前 SWAP 门数升序排列）"
        else:
            names.sort(key=lambda n: (rows["layout_contribution"].get(
                n, rows["layout_inject"].get(n, {})).get("n_qubits", 0), n))
            xlabel = "线路（按比特数升序排列）"
        ends = []
        for label, (fk, field), color in SERIES:
            xs, ys = [], []
            for i, name in enumerate(names):
                r = rows[fk].get(name)
                v = r.get(field) if r else None
                if isinstance(v, (int, float)) and v > 0:
                    xs.append(i)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.6,
                        color=color, zorder=3)
                ends.append((label, xs[-1], ys[-1], color))
        # 尾端直接标注（替代底部图例）：标签置于各折线最后一个数据点
        # 右侧，文字颜色与折线一致；相近末端按对数间距错开防重叠
        ends.sort(key=lambda t: t[2])
        placed = []
        max_pushed = 0.0
        for label, x, y, color in ends:
            for ly, px in placed:
                if abs(px - x) < 12 and y / ly < 1.6:
                    y = ly * 1.6
            ax.annotate(label, (x, y), xytext=(5, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8.5, color=color)
            placed.append((y, x))
            max_pushed = max(max_pushed, y)
        ax.set_xlim(-1.5, len(names) - 1 + 18)
        ax.set_yscale("log")
        # 统一量程：10^-2 ~ 10^2 秒（数据上限约 20s，10^2 以上无数据）；
        # 尾端标注错位推高时相应扩展上界
        ax.set_ylim(10 ** -2.2, max(10 ** 2, max_pushed * 1.2))
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([], fontsize=7)
        ax.set_title(blabel, fontsize=11, color=INK)
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
        if ax is axes[0]:
            ax.set_ylabel("端到端运行时间（秒，对数刻度）", fontsize=9,
                          color=INK2)
    fig_no = (args.fig_no or
              ("11" if args.sort == "qubits" else "12"))
    sort_note = ("按比特数升序" if args.sort == "qubits"
                 else "按优化前 SWAP 门数升序")
    fig.suptitle(f"图 {fig_no}  运行时间折线对比\n"
                 f"本方案布局 + O1 路由 / O3 路由（筛选） vs "
                 f"噪声感知 O1/O3 与 pytket\n"
                 f"（逐线路端到端运行时间，{sort_note}；盲目档与感知档共用"
                 f"同一编译管线，耗时相当，未单列）",
                 fontsize=11.5, y=1.01)
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    out = (args.out or
           f"tables/report_figures/图{fig_no}_运行时间折线对比.png")
    fig.savefig(Path(out), dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
