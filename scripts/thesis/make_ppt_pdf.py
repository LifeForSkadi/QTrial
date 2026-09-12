# -*- coding: utf-8 -*-
"""方案展示：matplotlib 渲染 16:9 页面 → 多页 PDF + 每页 PNG → 图片版 PPTX。

公式以 mathtext 渲染（含式编号，与报告书一致）；图表直接复用
tables/report_figures 的成品 PNG。图片版 PPT 保证可打开（整页图片）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent.parent
FIG = ROOT / "tables/report_figures/thesis"
OUT_PDF = ROOT / "方案展示.pdf"
OUT_PPT = ROOT / "方案展示.pptx"
PNG_DIR = ROOT / "tables/report_figures/slides"
PNG_DIR.mkdir(parents=True, exist_ok=True)

W, H = 13.333, 7.5
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"


def wrap_cjk(text, n):
    out, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= n:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def new_canvas():
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return fig, ax


def header(ax, title, page):
    ax.add_patch(plt.Rectangle((0, H - 0.72), W, 0.72, facecolor=SURFACE,
                               edgecolor="none", zorder=2))
    ax.add_patch(plt.Rectangle((0, H - 0.72), 0.12, 0.72, facecolor=BLUE,
                               edgecolor="none", zorder=3))
    ax.text(0.38, H - 0.36, title, fontsize=19, color=INK, va="center",
            fontweight="bold", zorder=4)
    ax.text(W - 0.35, 0.25, str(page), fontsize=9, color=MUTED,
            ha="right", va="center", zorder=4)


def bullets(ax, items, x=0.55, y=6.35, w=6.4, size=12.5, gap=0.62):
    yy = y
    for text, bold in items:
        for i, line in enumerate(wrap_cjk(text, 24)):
            yy -= gap if i == 0 else gap * 0.62
            ax.text(x, yy, ("▪ " if i == 0 else "   ") + line,
                    fontsize=size, color=INK, va="top",
                    fontweight="bold" if bold else "normal")
            yy -= 0.34
        yy -= 0.18
    return yy


def add_figure(ax, path, x, y, w, caption):
    img = mpimg.imread(str(path))
    ih, iw = img.shape[:2]
    h = w * ih / iw * (W / H)          # 保持纵横比（数据坐标）
    ax.imshow(img, extent=(x, x + w, y - h, y), zorder=2)
    ax.text(x + w / 2, y - h - 0.16, caption, fontsize=8.2, color=MUTED,
            ha="center", va="top", zorder=3)
    return y - h - 0.5


def add_formula(ax, x, y, w, label, tex, size=15):
    ax.text(x, y, label, fontsize=11, color=INK2, va="center",
            fontweight="bold")
    ax.text(x, y - 0.55, f"${tex}$", fontsize=size, color=INK,
            va="center", wrap=False)


def cover():
    fig, ax = new_canvas()
    ax.add_patch(plt.Rectangle((0, H - 2.65), W, 0.09, facecolor=BLUE))
    ax.text(W / 2, H - 3.1,
            "QTrail：基于稀疏奖励强化学习的\n自适应量子线路映射优化方案",
            fontsize=22, color=INK, ha="center", va="top", fontweight="bold",
            linespacing=1.6)
    ax.text(W / 2, H - 4.7, "面向天衍-287 超导处理器（105 比特）｜量子+AI 双向赋能榜题",
            fontsize=13, color=INK2, ha="center", va="top")
    ax.text(W / 2, 1.15, "单位：　　　　姓名：　　　　2026 年 9 月",
            fontsize=11, color=MUTED, ha="center", va="center")
    return fig


def toc():
    fig, ax = new_canvas()
    header(ax, "目录", 0)
    items = ["01 问题、方案与价值", "02 系统总体架构",
             "03 布局：QAP 形式化与强化学习策略", "04 路由：自研 SABRE",
             "05 后处理栈与平台基分解", "06 噪声感知与跨设备零重训",
             "07 训练过程与模型侧改进", "08 数据集与评测体系",
             "09 分层性能", "10 深度与运行时间", "11 噪声感知证据",
             "12 与成熟方案对比", "13 布局贡献与结论展望"]
    for i, t in enumerate(items):
        col, row = i % 2, i // 2
        x = 0.7 + col * 6.4
        y = 6.0 - row * 0.78
        ax.text(x, y, t, fontsize=12.5,
                color=BLUE if col == 0 else INK2, va="top")
    return fig


def thanks():
    fig, ax = new_canvas()
    ax.add_patch(plt.Rectangle((0, H - 3.6), W, 0.09, facecolor=BLUE))
    ax.text(W / 2, H - 4.1, "谢谢聆听，欢迎提问", fontsize=26, color=INK,
            ha="center", fontweight="bold")
    ax.text(W / 2, H - 5.2, "全部实验数据与代码可复现｜详见作品报告书与仓库",
            fontsize=12, color=INK2, ha="center")
    return fig


def s1():
    fig, ax = new_canvas()
    header(ax, "问题、方案与价值", 1)
    bullets(ax, [
        ("问题：线路必须匹配芯片拓扑——布局是 NP 难问题，路由要插 SWAP 门", True),
        ("方案：强化学习布局 + 自研 SABRE 路由 + 置换重写后处理，全链路零 qiskit 依赖", True),
        ("价值：天衍-287 真机可执行；587 条线路评测中总体保真度超过 pytket 与盲目基线", True),
        ("噪声感知布局在三拓扑上零重训迁移，训练一次处处可用", False),
    ])
    boxes = [("问题", "拓扑约束 + 噪声约束", "#cde2fb"),
             ("方案", "RL 布局｜自研 SABRE｜置换重写", "#ffffff"),
             ("价值", "真机可执行 · 保真度领先", "#cde2fb")]
    yy = 5.9
    for t1, t2, fc in boxes:
        ax.add_patch(FancyBboxPatch((7.9, yy - 1.05), 4.9, 1.05,
                                    boxstyle="round,pad=0.06",
                                    facecolor=fc, edgecolor=BLUE,
                                    linewidth=1.2))
        ax.text(10.35, yy - 0.5, t1, fontsize=14, color=INK, ha="center",
                va="center", fontweight="bold")
        ax.text(10.35, yy - 0.9, t2, fontsize=9.5, color=INK2, ha="center",
                va="center")
        if yy > 2.0:
            ax.text(10.35, yy - 1.35, "▼", fontsize=11, color=BLUE,
                    ha="center", va="center")
        yy -= 2.0
    return fig


def s2():
    fig, ax = new_canvas()
    header(ax, "系统总体架构", 2)
    bullets(ax, [
        ("五段管线：解析 → 图构建 → 布局 → 路由 → 后处理", True),
        ("程序图（6 维特征）与设备规格（7 维噪声特征）双输入", False),
        ("噪声感知机制（有效距离 D_eff）贯穿全链路", False),
        ("三层真机验证链：态矢量等价 / 平台校验 / 真机对照", False),
        ("失败阶梯保证任何输入都有输出", False),
    ], size=11.5)
    add_figure(ax, FIG / "图3.1_系统架构图.png", 6.35, 6.55, 6.45,
               "图3.1 系统总体架构：五段主链路 + 两条支撑链路")
    return fig


def s3():
    fig, ax = new_canvas()
    header(ax, "布局：QAP 形式化与强化学习策略", 3)
    bullets(ax, [
        ("布局 = 二次分配问题（NP 难），式(2-1) 为噪声感知代价", True),
        ("策略：GAT（4 层 8 头）+ 指针网络（16 头），REINFORCE 稀疏终局奖励", True),
        ("多起点解码（10 起点）+ 局部搜索精化", False),
        ("失败阶梯：策略 → 谱启发式 → 平凡布局", False),
    ])
    add_formula(ax, 7.8, 6.3, 5.5, "式(2-1) 布局静态代价",
                r"C(\pi)=\sum_{i<j} w_{ij}\cdot 2\cdot d(\pi(i),\,\pi(j))")
    add_formula(ax, 7.8, 4.4, 5.5, "式(4-2) 终局奖励（稀疏）",
                r"R=-\sum_{i<j} w_{ij}\cdot 2\cdot D_{\mathrm{eff}}(\pi(i),\pi(j))\,/\,\Sigma w",
                size=13)
    return fig


def s4():
    fig, ax = new_canvas()
    header(ax, "路由：自研 SABRE", 4)
    bullets(ax, [
        ("完整复现：前沿层并行执行 + 扩展集前瞻", True),
        ("逐门衰减权重（0.5^t）+ 交换历史惩罚", False),
        ("工程加速：增量维护 + Numba 融合内核，百比特级线路秒级", True),
        ("逐位 A/B 验证保证加速改造零行为变化", False),
    ])
    add_formula(ax, 7.8, 5.6, 5.6, "式(4-1) 交换评分",
                r"\mathrm{score}(a,b)=\sum_g 0.5^{t_g}\,d'(g)"
                r"+0.5\sum_e 0.5^{t_e}\,d'_e+0.5\,\mathrm{hist}[a,b]",
                size=14)
    return fig


def s5():
    fig, ax = new_canvas()
    header(ax, "后处理栈与平台基分解", 5)
    bullets(ax, [
        ("四层消化：共轭推挤 → 对消解 → 尾部吸收 → 平台基分解", True),
        ("每一步均为保语义等价变换，态矢量测试常驻守护", True),
        ("连通性红线：重标记后比特对必须在耦合图上", False),
        ("U3 再合成误差 2.5e-16（式(4-3)），落平台基 [rz, sx, x, cz]", False),
    ])
    add_formula(ax, 7.8, 5.8, 5.5, "式(4-3) U3 分解模板",
                r"U3(\theta,\phi,\lambda)=RZ(\lambda)\,SX\,RZ(\theta+\pi)\,SX\,RZ(\phi+3\pi)",
                size=15)
    return fig


def s6():
    fig, ax = new_canvas()
    header(ax, "噪声感知与跨设备零重训", 6)
    bullets(ax, [
        ("噪声揉进距离：式(2-2) 有效距离进奖励与评分", True),
        ("设备特征按自身统计归一化 → 学的是相对质量差异", True),
        ("同一权重覆盖 6×6 网格 / heavy-hex / Sycamore，零微调", False),
        ("图5.7：三拓扑 2Q 误差加权均值均显著低于盲目 O1", False),
    ], size=11.5)
    add_formula(ax, 0.55, 1.9, 6.0, "式(2-2) 有效距离",
                r"D_{\mathrm{eff}}=D_{\mathrm{topo}}+\lambda\,(D_{\mathrm{noise}}-D_{\mathrm{topo}})",
                size=14)
    add_figure(ax, FIG / "图5.7_跨拓扑噪声规避.png", 7.1, 6.5, 5.85,
               "图5.7 跨拓扑噪声规避（同一策略零重训迁移）")
    return fig


def s7():
    fig, ax = new_canvas()
    header(ax, "训练过程与模型侧改进", 7)
    bullets(ax, [
        ("两次独立训练一致收敛；单轮约 3 秒（RTX 5090）", True),
        ("LAUREL 可学习残差：收敛 400 轮 → 89 轮，约 5 分钟", True),
        ("效果不降反升：保真度 +3.1%，21/30 条更优", False),
        ("QuEst 风格预测器：T1/T2 退相干纳入保真度估计", False),
    ], size=11.5)
    add_figure(ax, FIG / "图5.2_LAUREL收敛效率.png", 6.9, 6.5, 6.0,
               "图5.2 LAUREL 训练效率与效果对比")
    return fig


def s8():
    fig, ax = new_canvas()
    header(ax, "数据集与评测体系", 8)
    bullets(ax, [
        ("Benchpress（Qiskit 官方测评集，656 条）+ MQTBench + QUEKO", True),
        ("587 条分层评测：≤10 / 11-50 / 51-105 三层", True),
        ("三种统计口径：均值 / 截尾 / 中位，原始数据逐线路落盘", False),
        ("自建 2 万条 287 原生池（含已知最优映射标注）", False),
    ], size=11.5)
    rows = [("数据集", "规模", "用途"),
            ("Benchpress", "656 条", "主评测集"),
            ("MQTBench", "36 类", "评测+训练图池"),
            ("QUEKO", "900 条", "评测+训练图池"),
            ("287 原生池", "20020 条", "训练图池")]
    tx, ty = 7.6, 5.9
    colw = [2.1, 1.3, 2.0]
    for i, row in enumerate(rows):
        x = tx
        for j, cell in enumerate(row):
            fc = "#f0efec" if i == 0 else ("#fcfcfb" if j else "#f7f6f3")
            ax.add_patch(plt.Rectangle((x, ty - i * 0.72), colw[j], 0.7,
                                       facecolor=fc, edgecolor="#c3c2b7",
                                       linewidth=0.6))
            ax.text(x + colw[j] / 2, ty - i * 0.72 - 0.35, cell,
                    fontsize=9.5 if i else 10, color=INK if i else INK2,
                    ha="center", va="center",
                    fontweight="bold" if i == 0 else "normal")
            x += colw[j] + 0.06
    ax.text(10.3, 1.35, "表5.1（节选）评测数据集构成", fontsize=8.2,
            color=MUTED, ha="center")
    return fig


def s9():
    fig, ax = new_canvas()
    header(ax, "分层性能", 9)
    bullets(ax, [
        ("≤10 层：保真度 0.619，显著高于盲目 O1（0.454）与 O3（0.505）", True),
        ("11-50 层：0.129 超过 pytket（0.105）与全部盲目基线", True),
        ("51-105 层为短板（0.0089），差距根因已定位（见第 13 页）", False),
    ], size=11.5)
    add_figure(ax, FIG / "图5.3_分层保真度对比.png", 6.55, 6.55, 6.35,
               "图5.3 分层估计保真度对比（587 条，均值口径）")
    return fig


def s10():
    fig, ax = new_canvas()
    header(ax, "深度与运行时间", 10)
    bullets(ax, [
        ("深度差距 = 残余 SWAP 展开：大层约 8.7 倍（截尾口径）", True),
        ("单条映射耗时 2.5-27.7 秒，满足分钟级目标", True),
        ("耗时比约 51 倍为工程差距，如实呈现", False),
    ], size=11.5)
    add_figure(ax, FIG / "图5.5_运行时间对比.png", 6.9, 6.5, 6.0,
               "图5.5 运行时间对比（51-105 比特，对数刻度）")
    return fig


def s11():
    fig, ax = new_canvas()
    header(ax, "噪声感知证据", 11)
    bullets(ax, [
        ("缺陷放大 ×5：盲目 O1 的 2Q 误差从 0.0035 恶化到 0.0110", True),
        ("本方案稳定在 0.0030-0.0033，与感知 O3（0.0027）接近", True),
        ("机制：奖励塑形 + 设备特征自归一化，缺一不可", False),
    ], size=11.5)
    add_figure(ax, FIG / "图5.6_缺陷敏感性扫描.png", 6.9, 6.5, 5.7,
               "图5.6 缺陷敏感性扫描（噪声感知鲁棒性证据）")
    return fig


def s12():
    fig, ax = new_canvas()
    header(ax, "与成熟方案对比", 12)
    bullets(ax, [
        ("总体保真度 0.196：超过 pytket（0.187）与全部盲目基线", True),
        ("胜场：对感知 O3 胜 29，对 pytket 胜 134，对盲目 O1 胜 226", True),
        ("为感知 O3 的 0.80 倍——差距集中在大线路层，如实呈现", False),
    ], size=11.5)
    add_figure(ax, FIG / "图5.8_总体对比.png", 6.55, 6.5, 6.3,
               "图5.8 与成熟映射方案总体对比（587 条线路）")
    return fig


def s13():
    fig, ax = new_canvas()
    header(ax, "布局贡献与结论展望", 13)
    bullets(ax, [
        ("路由器相同条件下，本方案布局保真度三基准全面第一", True),
        ("Benchpress：0.068 vs O3 0.003（约 21 倍）；MQTBench 0.279；QUEKO 0.410", True),
        ("结论：噪声感知布局路线成立；展望：门级再合成 + 学习化路由", True),
    ], size=11)
    add_figure(ax, FIG / "图5.10_布局贡献对比.png", 6.55, 6.55, 6.35,
               "图5.10 布局贡献对比（三基准，截尾均值）")
    return fig


def main():
    builders = [cover, toc, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11,
                s12, s13, thanks]
    with PdfPages(str(OUT_PDF)) as pdf:
        for i, fn in enumerate(builders):
            fig = fn()
            fig.savefig(PNG_DIR / f"slide_{i:02d}.png", dpi=200,
                        facecolor=SURFACE)
            pdf.savefig(fig, facecolor=SURFACE)
            plt.close(fig)
            print(f"页面 {i} OK: {fn.__name__}")
    print(f"saved pdf -> {OUT_PDF}")

    # 图片版 PPT（每页一张全幅图，保证可打开）
    from pptx import Presentation
    from pptx.util import Emu
    prs = Presentation()
    prs.slide_width = Emu(12192000)
    prs.slide_height = Emu(6858000)
    blank = prs.slide_layouts[6]
    for i in range(len(builders)):
        s = prs.slides.add_slide(blank)
        s.shapes.add_picture(str(PNG_DIR / f"slide_{i:02d}.png"), 0, 0,
                             width=Emu(12192000), height=Emu(6858000))
    prs.save(str(OUT_PPT))
    print(f"saved pptx -> {OUT_PPT}")


if __name__ == "__main__":
    main()
