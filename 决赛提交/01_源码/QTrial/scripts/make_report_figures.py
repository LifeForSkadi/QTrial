"""竞赛作品报告图表生成（中文图表系列，输出 tables/report_figures/）。

数据源全部为仓库内实验产物：
  - 训练过程：logs/train_*.jsonl（学习曲线）、configs/train_qtrail_laurel.yaml
  - 性能展示：tables/final_report/*.jsonl（587 条分层）、tables/speed_large_vs_o3.jsonl、
    tables/defect_sweep/f*.jsonl、tables/cross_topo/*.jsonl
  - 性能对比：tables/final_report、tables/compare_models/*.jsonl

设计规范：固定实体配色（QTrial=蓝、感知O3=橙、感知O1=青、盲目O3=黄、
盲目O1=品红、pytket=绿，全系列图一致）；单纵轴原则；小面板代替双轴；
关键柱体直接标值；中文字体微软雅黑。
用法：python scripts/make_report_figures.py [--out tables/report_figures]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------- 设计常量
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# 固定实体配色（顺序经 CVD 校验，全系列一致）
COLORS = {
    "QTrial": "#2a78d6",
    "感知O3": "#eb6834",
    "感知O1": "#1baf7a",
    "盲目O3": "#eda100",
    "盲目O1": "#e87ba4",
    "pytket": "#008300",
}
METHOD_ORDER = ["QTrial", "感知O3", "感知O1", "盲目O3", "盲目O1", "pytket"]
METHOD_LABELS = {"QTrial": "QTrial（本方案）", "感知O3": "噪声感知 O3",
                 "感知O1": "噪声感知 O1", "盲目O3": "盲目 O3",
                 "盲目O1": "盲目 O1", "pytket": "pytket"}

_plt_ready = False


def setup_fonts():
    global _plt_ready
    if _plt_ready:
        return
    for name in ("Microsoft YaHei", "SimHei", "PingFang SC"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["text.color"] = INK
    plt.rcParams["axes.edgecolor"] = BASELINE
    plt.rcParams["axes.labelcolor"] = INK2
    plt.rcParams["xtick.color"] = INK2
    plt.rcParams["ytick.color"] = INK2
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["figure.facecolor"] = SURFACE
    plt.rcParams["axes.facecolor"] = SURFACE
    _plt_ready = True


def style_ax(ax, xlabel="", ylabel="", grid="y"):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis=grid, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def value_labels(ax, rects, fmt="{:.3f}", dy=0.02, fontsize=9):
    """关键柱体直接标值（缓解色弱的 relief 手段）。"""
    for r in rects:
        ax.annotate(fmt.format(r.get_height()),
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize, color=INK2)


def grouped_bars(ax, groups, series, values, width=0.8, log=False):
    """通用分组柱。groups/values 为列表对齐；series 按 METHOD_ORDER。"""
    n_g = len(groups)
    n_s = len(series)
    xs = np.arange(n_g)
    w = width / n_s
    rects = []
    for i, s in enumerate(series):
        vs = [values[j][i] for j in range(n_g)]
        if log:
            vs = [max(v, 1e-12) for v in vs]
        r = ax.bar(xs + (i - (n_s - 1) / 2) * w, vs, w,
                   color=COLORS[s], label=METHOD_LABELS.get(s, s), zorder=3)
        rects.append(r)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=10)
    return rects


# ---------------------------------------------------------------- 数据加载
def load_final_report():
    """分层数据 → {stratum: {method: {fidelity_mean, depth_trim, depth_median}}}

    数据源 = tables/final_report/report.md（**v4 定稿口径**，与 README/汇报
    正文一致）。注意：tables/final_report/*.jsonl 是 gitignore 的草稿文件，
    后来被 bench6（硬件可执行口径）重跑覆盖，数字与定稿表不一致——图表
    必须用定稿表，不能用 jsonl 现场重算。
    """
    out = {}
    # 2026-11-13：优先使用 v8 分层表（y 修复 + mid-circuit 剔除后）；
    # 不存在时回退 v4 定稿表。
    v8_report = "tables/final_report_v8/report.md"
    path = v8_report if Path(v8_report).exists() else "tables/final_report/report.md"
    text = open(path, encoding="utf-8").read()
    sections = re.split(r"\n## ", text)
    strata_map = {"≤10": "≤10 比特", "11-50": "11-50 比特", "51-105": "51-105 比特"}
    wins = {"感知O3": 0, "pytket": 0, "盲目O1": 0}
    for sec in sections:
        title = sec.split("\n", 1)[0]
        key = next((k for k in strata_map if title.startswith(k)), None)
        if key is None:
            continue
        # 胜场行累计（图 8 用）
        for mth, pat in (("感知O3", r"vs 感知O3 \*\*(\d+)\*\*"),
                         ("pytket", r"vs pytket \*\*(\d+)\*\*"),
                         ("盲目O1", r"vs 盲目O1 \*\*(\d+)\*\*")):
            mm = re.search(pat, sec)
            if mm:
                wins[mth] += int(mm.group(1))
        # 指标表行解析：| 估计保真度 | a/b/c | ...；| 深度 | a/b/c | ...
        m = {}
        for line in sec.splitlines():
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if not cells or not cells[0].startswith(("估计保真度", "深度")):
                continue
            metric = cells[0]
            for j, method in enumerate(METHOD_ORDER):
                if j + 1 >= len(cells) or not cells[j + 1] or cells[j + 1] == "-":
                    continue
                parts = cells[j + 1].split("/")
                if metric.startswith("估计保真度"):
                    m.setdefault(method, {})["fidelity_mean"] = float(parts[0])
                elif metric.startswith("深度"):
                    m.setdefault(method, {}).update(
                        depth_trim=float(parts[1]), depth_median=float(parts[2]))
        out[strata_map[key]] = m
    out["_total_wins"] = wins            # 三层累计的总体胜场（图 8 用）
    return out


def load_defect_sweep():
    """缺陷扫描 → [(倍数, {method: mean2q})]，4 个强度文件。"""
    mults = {"f0": 0, "f2": 2, "f3": 3, "f5": 5}
    out = []
    for fname, mult in sorted(mults.items(), key=lambda kv: kv[1]):
        rows = [json.loads(l) for l in
                open(f"tables/defect_sweep/{fname}_rows.jsonl", encoding="utf-8")]
        m = {}
        for method in ("QTrial", "感知O1", "感知O3", "盲目O1"):
            pref = {"QTrial": "os", "感知O1": "aw_o1",
                    "感知O3": "aw_o3", "盲目O1": "blind"}[method]
            vals = [r[f"{pref}_mean2q"] for r in rows
                    if r.get(f"{pref}_mean2q") is not None]
            m[method] = float(np.mean(vals)) if vals else None
        out.append((mult, m))
    return out


def load_cross_topo():
    """跨拓扑 → [(拓扑名, {method: mean2q})]"""
    out = []
    for fname, label in (("grid-6x6", "6×6 网格（36 比特）"),
                         ("heavy-hex-115", "heavy-hex（115 比特）"),
                         ("sycamore-53", "Sycamore（53 比特）")):
        rows = [json.loads(l) for l in
                open(f"tables/cross_topo/{fname}_rows.jsonl", encoding="utf-8")]
        m = {}
        for method in ("QTrial", "感知O1", "感知O3", "盲目O1", "pytket"):
            pref = {"QTrial": "os", "感知O1": "aw_o1", "感知O3": "aw_o3",
                    "盲目O1": "blind", "pytket": "tket"}[method]
            vals = [r[f"{pref}_mean2q"] for r in rows
                    if r.get(f"{pref}_mean2q") is not None]
            m[method] = float(np.mean(vals)) if vals else None
        out.append((label, m))
    return out


def load_training_curves():
    """训练日志 → [(标签, epochs, val_cost, wall_s)]；缺文件返回空。"""
    out = []
    for fname, label in (("train_tianyan-287_gat_combined_calib_dep0.1_t0.5_"
                          "20260820_174731.jsonl", "训练实例二（247 轮）"),
                         ("train_tianyan-287_gat_combined_calib_dep0.1_t0.5_"
                          "20260820_041629.jsonl", "训练实例一（135 轮）")):
        p = Path("logs") / fname
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        out.append((label, [r["epoch"] for r in rows],
                    [r["val_cost"] for r in rows],
                    [r["wall_s"] for r in rows]))
    return out


def load_compare_models():
    """LAUREL vs 400ep 逐线路 → [(线路, f_laurel, f_orig)]"""
    def rows(name):
        p = Path("tables/compare_models") / name
        return [json.loads(l) for l in open(p, encoding="utf-8")]
    laurel = {r["circuit"]: r["fidelity"]
              for r in rows("LAUREL(89ep)_rows.jsonl")}
    orig = {r["circuit"]: r["fidelity"]
            for r in rows("源项目(400ep)_rows.jsonl")}
    common = sorted(set(laurel) & set(orig))
    return [(c.rsplit("/", 1)[-1], laurel[c], orig[c]) for c in common]


# ---------------------------------------------------------------- 各图
def fig1_training(curves, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, curve in zip(axes, curves):
        lab, ep, vc, wall = curve
        ax.plot(ep, vc, color=COLORS["QTrial"], linewidth=1.8, label=lab, zorder=3)
        style_ax(ax, "训练轮次", "验证集平均代价")
        ax.legend(frameon=False, loc="upper right", fontsize=10)
    # 右面板换成耗时（覆盖上面的实现：改为一个面板画两条曲线 + 一个面板画耗时）
    fig.clear()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for i, (lab, ep, vc, wall) in enumerate(curves):
        c = [COLORS["QTrial"], "#86b6ef"][i]
        ax.plot(ep, vc, color=c, linewidth=1.8, label=lab, zorder=3)
    style_ax(ax, "训练轮次", "验证集平均代价")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax = axes[1]
    if curves:
        lab, ep, vc, wall = curves[0]
        ax.plot(ep, wall, color=COLORS["感知O1"], linewidth=1.8, zorder=3)
        ax.annotate(f"单轮平均耗时 ≈ {np.mean(wall):.2f} 秒",
                    xy=(0.62, 0.9), xycoords="axes fraction",
                    fontsize=10, color=INK2)
        style_ax(ax, "训练轮次", "每轮训练耗时（秒）")
        ax.set_ylim(0, max(wall) * 1.25)
    fig.suptitle("图 1  布局策略训练过程（验证集平均代价收敛 + 单轮耗时）",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "图1_训练学习曲线.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig3_stratified_fidelity(data, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, stratum in zip(axes, ["≤10 比特", "11-50 比特", "51-105 比特"]):
        m = data[stratum]
        series = [s for s in METHOD_ORDER if m[s]["fidelity_mean"] is not None]
        vals = [m[s]["fidelity_mean"] for s in series]
        n_s = len(series)
        w = 0.8 / n_s
        rects = []
        for i, s in enumerate(series):
            r = ax.bar([(i - (n_s - 1) / 2) * w], [vals[i]], w,
                       color=COLORS[s], label=METHOD_LABELS.get(s, s),
                       zorder=3)
            rects.append(r[0])
        ax.set_xticks([])
        ax.set_xticklabels([])
        # 全柱标值：统一偏移、标签直接置顶于各柱（柱间距足够容纳
        # 标签宽度，不重叠）；白底框防与网格混淆
        for i, s in enumerate(series):
            r = rects[i]
            ax.annotate(f"{vals[i]:.4f}",
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK,
                        zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12",
                                  facecolor=SURFACE, edgecolor="none"))
        ax.set_ylim(0, max(vals) * 1.28)
        style_ax(ax, "", "估计保真度（均值）" if stratum == "≤10 比特" else "")
        ax.set_title(stratum, fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper right",
                  ncols=2, columnspacing=0.6)
    fig.suptitle("图 3  分层估计保真度对比（587 条线路，均值口径）",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "图3_分层保真度对比.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig4_stratified_depth(data, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, stratum, logy in zip(axes, ["≤10 比特", "11-50 比特", "51-105 比特"],
                                 [False, True, True]):
        m = data[stratum]
        series = [s for s in METHOD_ORDER if m[s]["depth_trim"] is not None]
        vals = [m[s]["depth_trim"] for s in series]
        n_s = len(series)
        w = 0.8 / n_s
        xs = [(i - (n_s - 1) / 2) * w for i in range(n_s)]
        if logy:
            # 对数轴必须先于柱体设置；柱底锚定在轴下限 10²，柱高 =
            # 真实值 − 10²，顶部即真实截尾均值
            ax.set_yscale("log")
            ax.set_ylim(1e2, 1e6)
            bottoms = [1e2] * n_s
            heights = [max(v - 1e2, 1.0) for v in vals]
        else:
            bottoms = [0] * n_s
            heights = vals
        for i, s in enumerate(series):
            ax.bar([xs[i]], [heights[i]], w, bottom=bottoms[i],
                   color=COLORS[s], label=METHOD_LABELS.get(s, s), zorder=3)
        ax.set_xticks([])
        # 全柱标值：统一偏移、标签直接置顶于各柱；白底框防止被邻近
        # 高柱吞掉
        for i, s in enumerate(series):
            ax.annotate(f"{vals[i]:.0f}",
                        (xs[i], vals[i]),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8,
                        color=INK, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor=SURFACE, edgecolor="none"))
        if not logy:
            ax.set_ylim(0, max(vals) * 1.25)
        style_ax(ax, "", "线路深度（截尾均值）" if stratum == "≤10 比特" else "")
        ax.set_title(f"{stratum}" + ("（对数刻度）" if logy else ""),
                     fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper right",
                  ncols=2, columnspacing=0.6)
    fig.suptitle("图 4  分层线路深度对比（截尾均值口径，去两端 10% 极值）",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "图4_分层深度对比.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig6_defect_sweep(sweep, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    mults = [m for m, _ in sweep]
    # 末端标签按数值排序后错开纵偏移，避免数值接近时重叠
    series = []
    for method in ("QTrial", "感知O1", "感知O3", "盲目O1"):
        vals = [mm[method] for _, mm in sweep]
        ax.plot(mults, vals, marker="o", markersize=5, linewidth=1.8,
                color=COLORS[method], label=METHOD_LABELS[method], zorder=3)
        series.append((method, vals))
    series_sorted = sorted(series, key=lambda sv: sv[1][-1])
    for rank, (method, vals) in enumerate(series_sorted):
        ax.annotate(f"{vals[-1]:.4f}",
                    (mults[-1], vals[-1]),
                    xytext=(6, 2 + 11 * (rank - len(series) / 2)),
                    textcoords="offset points",
                    fontsize=9, color=INK2)
    style_ax(ax, "缺陷误差放大倍数", "2Q 误差加权均值")
    ax.set_xticks(mults)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("图 6  缺陷敏感性扫描（噪声感知的鲁棒性证据）", fontsize=13,
                 y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "图6_缺陷敏感性扫描.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig7_cross_topo(ct, outdir):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    groups = [g for g, _ in ct]
    series = ("QTrial", "感知O1", "感知O3", "盲目O1", "pytket")
    vals = [[mm[s] if mm.get(s) is not None else None for _, mm in ct]
            for s in series]
    xs = np.arange(len(groups))
    w = 0.8 / len(series)
    for i, s in enumerate(series):
        vs = [v if v is not None else 0 for v in vals[i]]
        ax.bar(xs + (i - (len(series) - 1) / 2) * w, vs, w,
               color=COLORS[s], label=METHOD_LABELS[s], zorder=3)
    # 全柱标值：统一偏移、标签直接置顶于各柱
    for j in range(len(groups)):
        for i in range(len(series)):
            v = vals[i][j]
            if v is None:
                continue
            ax.annotate(f"{v:.4f}",
                        (xs[j] + (i - (len(series) - 1) / 2) * w, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK,
                        zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12",
                                  facecolor=SURFACE, edgecolor="none"))
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=10)
    style_ax(ax, "", "2Q 误差加权均值（越低越好）")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("图 7  跨拓扑噪声规避（同一策略零重训迁移）", fontsize=13,
                 y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "图7_跨拓扑噪声规避.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig8_overall(data, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    # 总体保真度 = 分层均值按层内线路数加权（129/259/199 → 587）
    counts = {"≤10 比特": 129, "11-50 比特": 259, "51-105 比特": 199}
    series = [s for s in METHOD_ORDER
              if all(data[strt][s]["fidelity_mean"] is not None
                     for strt in counts)]
    vals = [sum(data[strt][s]["fidelity_mean"] * counts[strt]
                for strt in counts) / sum(counts.values())
            for s in series]
    rects = ax.bar(range(len(series)), vals, 0.62,
                   color=[COLORS[s] for s in series], zorder=3)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([METHOD_LABELS[s].replace("（本方案）", "") for s in series],
                       fontsize=9)
    value_labels(ax, rects, fmt="{:.3f}", fontsize=9)
    style_ax(ax, "", "总体估计保真度（587 条加权均值）")
    ax.set_ylim(0, max(vals) * 1.2)
    ax = axes[1]
    wins = data.get("_total_wins", {})
    order = [("对 感知O3", "感知O3"), ("对 pytket", "pytket"),
             ("对 盲目O1", "盲目O1")]
    r2 = ax.bar([0, 1, 2], [wins.get(mth, 0) for _, mth in order], 0.55,
                color=[COLORS[mth] for _, mth in order], zorder=3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([t for t, _ in order], fontsize=10)
    for r, (_, mth) in zip(r2, order):
        ax.annotate(f"{wins.get(mth, 0)}",
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color=INK2)
    style_ax(ax, "", "逐线路胜场数")
    ax.set_ylim(0, max(wins.values()) * 1.25)
    fig.suptitle("图 8  与成熟映射方案总体对比", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "图8_总体对比.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig9_laurel_detail(rows, outdir):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    names = [r[0] for r in rows]
    diffs = [r[1] - r[2] for r in rows]
    order = np.argsort(diffs)
    names = [names[i] for i in order]
    diffs = [diffs[i] for i in order]
    colors = [COLORS["QTrial"] if d >= 0 else "#e34948" for d in diffs]
    ax.bar(range(len(diffs)), diffs, 0.7, color=colors, zorder=3)
    ax.axhline(0, color=BASELINE, linewidth=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6.5)
    style_ax(ax, "", "保真度差（LAUREL − 原配置）")
    n_win = sum(1 for d in diffs if d > 0)
    ax.annotate(f"LAUREL 更优 {n_win} 条 / 共 {len(diffs)} 条",
                xy=(0.02, 0.9), xycoords="axes fraction", fontsize=10,
                color=INK2)
    fig.suptitle("图 9  LAUREL 与原配置逐线路保真度差（QUEKO BIGD）",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "图9_LAUREL逐线路对比.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig10_layout_contribution(outdir, fig_no="10", out_name=None):
    """图 10：布局贡献对比——本方案布局 + qiskit SABRE 路由 vs
    pytket / 盲目 O1、O3 / 噪声感知 O1、O3，另给出本方案布局注入
    qiskit O3 管线的组合；在三个基准上（Benchpress 为 Qiskit 官方测评
    自己的测试集），3×2 小面板 = 基准 × 指标（保真度/深度），截尾均值
    口径（去两端 10% 极值）。SWAP 数因各方法在该测试集上差异极小
    （同路由器口径）不作展示。"""
    INJ = "#7ba8e0"  # QTrial 蓝的浅调：本方案布局 + O3 管线组合
    benches = [("benchpress", "Benchpress\n（Qiskit 官方）"),
               ("mqtbench_n15", "MQTBench n15"),
               ("queko_BIGD", "QUEKO BIGD")]
    entities = [("本方案布局\n+O1 路由", "rl", COLORS["QTrial"]),
                ("本方案布局\n+O3 路由", "inj", INJ),
                ("噪声感知 O3", "aw-o3", COLORS["感知O3"]),
                ("噪声感知 O1", "aw-o1", COLORS["感知O1"]),
                ("盲目 O3", "qiskit-o3", COLORS["盲目O3"]),
                ("盲目 O1", "qiskit-o1", COLORS["盲目O1"]),
                ("pytket", "tket", COLORS["pytket"])]

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.5))
    for row_i, (bench, blabel) in enumerate(benches):
        src = Path(f"tables/compare_models/"
                   f"layout_contribution_{bench}_rows.jsonl")
        if not src.exists():
            print(f"图10 [{bench}] 跳过（无数据，先跑 "
                  f"scripts/bench_layout_contribution.py --bench {bench}）")
            continue
        rows = [json.loads(l) for l in open(src, encoding="utf-8")]
        blabel = f"{blabel}\n（{len(rows)} 条）"
        inj_src = Path(f"tables/compare_models/"
                       f"layout_inject_{bench}_rows.jsonl")
        inj_rows = ([json.loads(l) for l in open(inj_src, encoding="utf-8")]
                    if inj_src.exists() else [])

        def mean_of(field, rs):
            vals = sorted(r[field] for r in rs
                          if isinstance(r.get(field), (int, float)))
            if not vals:
                return None
            k = max(1, int(len(vals) * 0.1))
            if 2 * k >= len(vals):
                return float(np.mean(vals))
            return float(np.mean(vals[k:len(vals) - k]))

        for col_i, (mname, title, fmt) in enumerate(
                (("fidelity", "估计保真度（截尾均值）", "{:.4f}"),
                 ("depth", "线路深度（截尾均值）", "{:.0f}"))):
            ax = axes[row_i][col_i]
            vals = []
            for label, pref, color in entities:
                field = mname if pref == "rl" else f"{pref}_{mname}"
                rs = inj_rows if pref == "inj" else rows
                vals.append((label, mean_of(field, rs), color))
            xs = range(len(vals))
            rects = ax.bar(xs, [v[1] if v[1] else 0 for v in vals], 0.58,
                           color=[v[2] for v in vals], zorder=3)
            ax.set_xticks(list(xs))
            ax.set_xticklabels([v[0] for v in vals], fontsize=7)
            hi = max([v[1] for v in vals if v[1]] or [1])
            ax.set_ylim(0, hi * 1.28)
            for r, (label, v, _) in zip(rects, vals):
                if not v:
                    continue
                txt = fmt.format(v)
                if v < 0.12 * hi:
                    # 极小值：标注置于柱顶右侧，避免遮挡 x 轴刻度标签
                    ax.annotate(txt,
                                (r.get_x() + r.get_width(), v),
                                xytext=(5, 0), textcoords="offset points",
                                ha="left", va="center", fontsize=7.5,
                                color=INK, zorder=5,
                                bbox=dict(boxstyle="round,pad=0.12",
                                          facecolor=SURFACE,
                                          edgecolor="none"))
                else:
                    ax.annotate(txt,
                                (r.get_x() + r.get_width() / 2, v),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7.5,
                                color=INK, zorder=5,
                                bbox=dict(boxstyle="round,pad=0.12",
                                          facecolor=SURFACE,
                                          edgecolor="none"))
            style_ax(ax, "", "")
            if row_i == 0:
                # 列标题置于顶行面板顶部（不再占用 ylabel——第一列的
                # ylabel 是基准名，曾因此丢失列标题）
                ax.set_title(title, fontsize=11, color=INK)
            if col_i == 0:
                ax.set_ylabel(blabel, fontsize=9, rotation=0,
                              labelpad=46, va="center")
    fig.suptitle(f"图 {fig_no}  布局贡献对比：本方案布局 + O1 路由 / O3 路由 vs "
                 f"pytket 与盲目/噪声感知 O1、O3（三基准 × 保真度/深度，截尾均值口径）",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / (out_name or f"图{fig_no}_布局贡献对比.png"), dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def table1_training_settings(outdir):
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off")
    rows = [
        ["项目", "设置"],
        ["训练数据集", "ER 随机图（p=0.3）50% + 线路图池 50%（MQTBench / "
         "QUEKO / 天衍-287 原生池 2 万条）"],
        ["规模分桶", "5-40 比特 50% / 41-80 比特 30% / 81-105 比特 20%"],
        ["布局模型", "GAT 编码器 4 层 × 8 头（d=128，GraphNorm）"
         "+ 16 头指针解码器（tanh 截断 C=10）+ LAUREL 可学习残差（rw_lr，秩 4）"],
        ["训练算法", "REINFORCE + 自批评贪心基线；batch 512；每轮 4096 实例"],
        ["奖励函数", "终局稀疏奖励 R = −Σ w_ij · 2 · D_eff(π_i, π_j) / Σw"
         "（噪声加权距离，λ_n = 0.5）"],
        ["优化器", "Adam，学习率 3e-4 余弦退火至 3e-5，梯度截断 1.0"],
        ["训练预算", "总预算 400 轮，早停 patience 50；LAUREL 配置 89 轮收敛"],
        ["硬件与耗时", "RTX 5090；单轮约 3 秒，89 轮总耗时约 5 分钟"],
        ["复现命令", "python -m qtrail.train --config "
         "configs/train_qtrail_laurel.yaml --dev cuda"],
    ]
    tb = ax.table(cellText=rows, loc="center", cellLoc="left")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.6)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if i == 0:
            cell.set_text_props(color=INK, fontweight="bold")
            cell.set_facecolor("#f0efec")
        elif j == 0:
            cell.set_facecolor("#f7f6f3")
            cell.set_text_props(color=INK2)
            cell.set_width(0.22)
        else:
            cell.set_facecolor(SURFACE)
            cell.set_text_props(color=INK)
    ax.set_title("表 1  布局策略训练设置（可复现口径）", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(outdir / "表1_训练设置.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables/report_figures")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_fonts()

    curves = load_training_curves()
    if curves:
        fig1_training(curves, outdir)
        print("图1 训练学习曲线: OK")
    else:
        print("图1 跳过（无训练日志）")

    data = load_final_report()
    fig3_stratified_fidelity(data, outdir)
    print("图3 分层保真度: OK")
    fig4_stratified_depth(data, outdir)
    print("图4 分层深度: OK")

    fig6_defect_sweep(load_defect_sweep(), outdir)
    print("图6 缺陷敏感性扫描: OK")

    fig7_cross_topo(load_cross_topo(), outdir)
    print("图7 跨拓扑噪声规避: OK")

    fig8_overall(data, outdir)
    print("图8 总体对比: OK")

    cm = load_compare_models()
    if cm:
        fig9_laurel_detail(cm, outdir)
        print("图9 LAUREL 逐线路对比: OK")

    fig10_layout_contribution(outdir)
    print("图10 布局贡献对比: OK")
    # thesis 版（内部标题与第 5 章编号一致：图5.6）
    fig10_layout_contribution(outdir / "thesis", fig_no="5.6",
                               out_name="图5.6_布局贡献对比.png")

    table1_training_settings(outdir)
    print("表1 训练设置: OK")
    print(f"全部输出 -> {outdir}")


if __name__ == "__main__":
    main()
