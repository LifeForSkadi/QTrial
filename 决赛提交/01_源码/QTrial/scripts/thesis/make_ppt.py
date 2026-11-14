# -*- coding: utf-8 -*-
"""生成方案展示 PPT（16:9，OMML 原生公式，图注，三段式版面）。

编号与术语与作品报告书完全一致：式(2-1)/式(2-2)/式(4-1)/式(4-3)，
图 3.1/5.2/5.3/5.4/5.5/5.6/5.7/5.8/5.10，表 5.1。
正文 13 页（不含封面/目录/致谢），总时长目标 ≤ 600 秒。
"""
from __future__ import annotations

import copy
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Cm, Emu, Pt

ROOT = Path(__file__).resolve().parent.parent.parent
FIG = ROOT / "tables/report_figures/thesis"
OUT = ROOT / "方案展示.pptx"

# 调色板（与图表系列一致）
BLUE = RGBColor(0x2A, 0x78, 0xD6)
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
LIGHT = RGBColor(0xCD, 0xE2, 0xFB)
FONT = "微软雅黑"

EMU_W, EMU_H = 12192000, 6858000   # 13.333 x 7.5 in


def _set_font(run, size, bold=False, color=INK, italic=False):
    f = run.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = etree.SubElement(rPr, qn("a:ea"))
    ea.set("typeface", FONT)


def add_rect(slide, x, y, w, h, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
    sh.shadow.inherit = False
    return sh


def add_text(slide, x, y, w, h, lines, size=16, bold=False, color=INK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.15):
    """lines = [(text, size_override, bold_override, color_override), ...] 或 str"""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    if isinstance(lines, str):
        lines = [(lines, None, None, None)]
    first = True
    for text, sz, bd, cl in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = text
        _set_font(run, sz or size, bold if bd is None else bd,
                  cl or color)
    return tb


def title_bar(slide, title, subtitle=""):
    add_rect(slide, 0, 0, Emu(EMU_W), Cm(1.15), SURFACE)
    add_rect(slide, 0, 0, Cm(0.18), Cm(1.15), BLUE)
    add_text(slide, Cm(0.55), Cm(0.18), Cm(12.0), Cm(0.9), title,
             size=24, bold=True, color=INK)
    if subtitle:
        add_text(slide, Cm(9.5), Cm(0.42), Cm(3.6), Cm(0.6), subtitle,
                 size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    # 页码
    add_text(slide, Cm(12.55), Cm(7.05), Cm(0.7), Cm(0.4),
             str(slide.slide_id // 256 if hasattr(slide, "slide_id") else ""),
             size=9, color=MUTED, align=PP_ALIGN.RIGHT)


def bullets(slide, x, y, w, h, items, size=15, gap=6):
    """items = [(text, bold)] 列表。"""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for text, bold in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.line_spacing = 1.18
        p.space_after = Pt(gap)
        run = p.add_run()
        run.text = "▪ " + text
        _set_font(run, size, bold)
    return tb


def figure_with_caption(slide, path, x, y, w, caption):
    slide.shapes.add_picture(str(path), x, y, width=w)
    add_text(slide, x, y + Emu(int(w * 0.062) * 100) if False else
             Emu(int(w)),  # 占位避免误用
             None, None, None)
    return


def add_figure(slide, path, x, y, w, caption, caption_w=None):
    pic = slide.shapes.add_picture(str(path), x, y, width=w)
    ch = int(w / pic.width * pic.height)
    add_text(slide, x, y + Emu(ch) + Emu(40000), caption_w or w, Cm(0.75),
             caption, size=10, color=MUTED, align=PP_ALIGN.CENTER)
    return ch


# ---------------------------------------------------------------- OMML
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _m(tag):
    return f"{{{M}}}{tag}"


def _r(parent, text):
    r = etree.SubElement(parent, _m("r"))
    t = etree.SubElement(r, _m("t"))
    t.text = text
    return r


def _ssub(parent, base, sub):
    s = etree.SubElement(parent, _m("sSub"))
    e = etree.SubElement(s, _m("e"))
    _r(e, base)
    sb = etree.SubElement(s, _m("sub"))
    _r(sb, sub)
    return s


def _ssup(parent, base, sup):
    s = etree.SubElement(parent, _m("sSup"))
    e = etree.SubElement(s, _m("e"))
    _r(e, base)
    sb = etree.SubElement(s, _m("sup"))
    _r(sb, sup)
    return s


def _nary(parent, sub_text, e_items):
    n = etree.SubElement(parent, _m("nary"))
    pr = etree.SubElement(n, _m("naryPr")
                          )
    chr_ = etree.SubElement(pr, _m("chr"))
    chr_.set(_m("val"), "∑")
    lim = etree.SubElement(pr, _m("limLoc"))
    lim.set(_m("val"), "undOvr")
    sub = etree.SubElement(n, _m("sub"))
    _r(sub, sub_text)
    sup = etree.SubElement(n, _m("sup"))
    e = etree.SubElement(n, _m("e"))
    _items(e, e_items)
    return n


def _items(parent, items):
    for it in items:
        kind = it[0]
        if kind == "r":
            _r(parent, it[1])
        elif kind == "ssub":
            _ssub(parent, it[1], it[2])
        elif kind == "ssup":
            _ssup(parent, it[1], it[2])
        elif kind == "nary":
            _nary(parent, it[1], it[2])


def add_omml(slide, x, y, w, label, items, size_hint=20):
    """插入 OMML 原生公式（可双击编辑）+ 公式编号标签。

    PowerPoint 文本框内公式的规范结构为
    <a:r><a14:m><m:oMathPara><m:oMath>…，必须经 a14 扩展命名空间包装，
    且 slide 根元素声明 mc:Ignorable 含 a14 与 m。
    """
    A14 = "http://schemas.microsoft.com/office/drawing/2010/main"
    tb = slide.shapes.add_textbox(x, y, w, Cm(2.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = label
    _set_font(run, 14, bold=True, color=INK2)
    # 公式 run：a:r > a14:m > m:oMathPara > m:oMath
    r = etree.SubElement(p._p, qn("a:r"))
    a14m = etree.SubElement(r, f"{{{A14}}}m")
    omp = etree.SubElement(a14m, _m("oMathPara"))
    om = etree.SubElement(omp, _m("oMath"))
    _items(om, items)
    # slide 根元素声明
    MC_IGN = ("{http://schemas.openxmlformats.org/markup-compatibility/"
              "2006}Ignorable")
    sld = slide._element
    ignorable = sld.get(MC_IGN) or ""
    for tok in ("a14", "m"):
        if tok not in ignorable.split():
            ignorable += (" " + tok)
    sld.set(MC_IGN, ignorable.strip())
    return tb


# ---------------------------------------------------------------- 公式定义
F21 = [("r", "C("), ("r", "π"), ("r", ") = "),
       ("nary", "i<j", [("ssub", "w", "ij"), ("r", " · 2 · d("),
                        ("r", "π"), ("r", "(i), "), ("r", "π"), ("r", "(j))")])]

F22 = [("r", "D"), ("ssub", "eff", ""), ("r", " = D"), ("ssub", "topo", ""),
       ("r", " + λ(D"), ("ssub", "noise", ""), ("r", " − D"),
       ("ssub", "topo", ""), ("r", ")")]

F41 = [("r", "score(a,b) = "),
       ("nary", "g", [("ssup", "0.5", "t"), ("r", " · d'(g)")]),
       ("r", " + 0.5 · "),
       ("nary", "e", [("ssup", "0.5", "t"), ("r", " · d'")]),
       ("r", " + 0.5 · hist[a,b]")]

F43 = [("r", "U3("), ("r", "θ,φ,λ"), ("r", ") = RZ("), ("r", "λ"),
       ("r", ")·SX·RZ("), ("r", "θ"), ("r", "+"), ("r", "π"),
       ("r", ")·SX·RZ("), ("r", "φ"), ("r", "+3"), ("r", "π"), ("r", ")")]


def new_slide(prs, title, subtitle=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_bar(slide, title, subtitle)
    return slide


def build():
    prs = Presentation()
    prs.slide_width = Emu(EMU_W)
    prs.slide_height = Emu(EMU_H)

    # ---------------- 封面（不计时）
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(s, 0, 0, Emu(EMU_W), Emu(EMU_H), SURFACE)
    add_rect(s, 0, Cm(2.4), Emu(EMU_W), Cm(0.12), BLUE)
    add_text(s, Cm(1.5), Cm(2.9), Cm(10.3), Cm(1.6),
             "QTrail：基于稀疏奖励强化学习的自适应量子线路映射优化方案",
             size=28, bold=True, color=INK, align=PP_ALIGN.CENTER)
    add_text(s, Cm(1.5), Cm(4.5), Cm(10.3), Cm(1.0),
             "面向天衍-287 超导处理器（105 比特）｜量子+AI 双向赋能榜题",
             size=16, color=INK2, align=PP_ALIGN.CENTER)
    add_text(s, Cm(1.5), Cm(6.2), Cm(10.3), Cm(0.8),
             "单位：　　　　姓名：　　　　2026 年 9 月",
             size=13, color=MUTED, align=PP_ALIGN.CENTER)

    # ---------------- 目录（不计时）
    s = new_slide(prs, "目录")
    toc = [("01", "问题、方案与价值"), ("02", "系统总体架构"),
           ("03", "布局：QAP 形式化与强化学习策略"), ("04", "路由：自研 SABRE"),
           ("05", "后处理栈与平台基分解"), ("06", "噪声感知与跨设备零重训"),
           ("07", "训练过程与模型侧改进"), ("08", "数据集与评测体系"),
           ("09", "分层性能"), ("10", "深度与运行时间"),
           ("11", "噪声感知证据"), ("12", "与成熟方案对比"),
           ("13", "布局贡献与结论展望")]
    tb = s.shapes.add_textbox(Cm(1.2), Cm(1.7), Cm(11.0), Cm(5.4))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (num, title) in enumerate(toc):
        col = i % 2
        row = i // 2
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if col == 0:
            pass
        run = p.add_run()
        run.text = f"{num}  {title}"
        _set_font(run, 14, bold=(col == 0), color=BLUE if col == 0 else INK2)
        p.space_after = Pt(14)

    # ================= 正文 13 页 =================
    # 第 1 页
    s = new_slide(prs, "问题、方案与价值")
    bullets(s, Cm(0.8), Cm(1.7), Cm(6.4), Cm(4.8), [
        ("问题：线路必须匹配芯片拓扑——布局是 NP 难问题，路由要插 SWAP 门", True),
        ("方案：强化学习布局 + 自研 SABRE 路由 + 置换重写后处理，全链路零 qiskit 依赖", True),
        ("价值：天衍-287 真机可执行；587 条线路评测中总体保真度超过 pytket 与盲目基线", True),
        ("噪声感知布局在三拓扑上零重训迁移，训练一次处处可用", False),
    ])
    # 右侧流程图：问题 → 方案 → 价值
    from pptx.enum.shapes import MSO_SHAPE
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    boxes = [("问题", "拓扑约束 + 噪声约束", LIGHT),
             ("方案", "RL 布局｜自研 SABRE｜置换重写", WHITE),
             ("价值", "真机可执行 · 保真度领先", LIGHT)]
    yy = Cm(2.0)
    for i, (t1, t2, fill) in enumerate(boxes):
        sh = add_rect(s, Cm(8.0), yy, Cm(4.6), Cm(1.25), fill, line=BLUE if fill == WHITE else None)
        tf = sh.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = t1
        _set_font(run, 16, bold=True, color=INK)
        p2 = tf.add_paragraph()
        run2 = p2.add_run()
        run2.text = t2
        _set_font(run2, 11, color=INK2)
        if i < 2:
            add_text(s, Cm(10.2), yy + Cm(1.3), Cm(0.6), Cm(0.7), "▼",
                     size=14, color=BLUE, align=PP_ALIGN.CENTER)
        yy += Cm(1.9)

    # 第 2 页
    s = new_slide(prs, "系统总体架构")
    bullets(s, Cm(0.8), Cm(1.6), Cm(5.4), Cm(5.2), [
        ("五段管线：解析 → 图构建 → 布局 → 路由 → 后处理", True),
        ("程序图（6 维特征）与设备规格（7 维噪声特征）双输入", False),
        ("噪声感知机制（有效距离 D_eff）贯穿全链路", False),
        ("三层真机验证链：态矢量等价 / 平台校验 / 真机对照", False),
        ("失败阶梯保证任何输入都有输出", False),
    ])
    add_figure(s, FIG / "图3.1_系统架构图.png", Cm(6.7), Cm(1.6), Cm(6.0),
               "图3.1 系统总体架构：五段主链路 + 两条支撑链路")

    # 第 3 页
    s = new_slide(prs, "布局：QAP 形式化与强化学习策略")
    bullets(s, Cm(0.8), Cm(1.6), Cm(6.6), Cm(4.6), [
        ("布局 = 二次分配问题（NP 难），式(2-1) 为噪声感知代价", True),
        ("策略：GAT（4 层 8 头）+ 指针网络（16 头），REINFORCE 稀疏终局奖励", True),
        ("多起点解码（10 起点）+ 局部搜索精化", False),
        ("失败阶梯：策略 → 谱启发式 → 平凡布局", False),
    ])
    add_omml(s, Cm(8.2), Cm(1.9), Cm(4.6), "式(2-1) 布局静态代价", F21)
    add_omml(s, Cm(8.2), Cm(4.3), Cm(4.6), "式(4-2) 终局奖励（稀疏）", [
        ("r", "R = −("), ("nary", "i<j", [("ssub", "w", "ij"), ("r", "·2·D"),
                                            ("ssub", "eff", ""), ("r", "(π(i),π(j))")]),
        ("r", ") / Σw")])

    # 第 4 页
    s = new_slide(prs, "路由：自研 SABRE")
    bullets(s, Cm(0.8), Cm(1.6), Cm(6.6), Cm(4.6), [
        ("完整复现：前沿层并行执行 + 扩展集前瞻", True),
        ("逐门衰减权重（0.5^t）+ 交换历史惩罚", False),
        ("工程加速：增量维护 + Numba 融合内核，百比特级线路秒级", True),
        ("逐位 A/B 验证保证加速改造零行为变化", False),
    ])
    add_omml(s, Cm(8.2), Cm(2.2), Cm(4.6), "式(4-1) 交换评分", F41)

    # 第 5 页
    s = new_slide(prs, "后处理栈与平台基分解")
    bullets(s, Cm(0.8), Cm(1.6), Cm(6.6), Cm(4.6), [
        ("四层消化：共轭推挤 → 对消解 → 尾部吸收 → 平台基分解", True),
        ("每一步均为保语义等价变换，态矢量测试常驻守护", True),
        ("连通性红线：重标记后比特对必须在耦合图上", False),
        ("U3 再合成误差 2.5e-16（式(4-3)），落平台基 [rz, sx, x, cz]", False),
    ])
    add_omml(s, Cm(8.2), Cm(2.4), Cm(4.6), "式(4-3) U3 分解模板", F43)

    # 第 6 页
    s = new_slide(prs, "噪声感知与跨设备零重训")
    bullets(s, Cm(0.8), Cm(1.6), Cm(6.4), Cm(4.6), [
        ("噪声揉进距离：式(2-2) 有效距离进奖励与评分", True),
        ("设备特征按自身统计归一化 → 学的是相对质量差异", True),
        ("同一权重覆盖 6×6 网格 / heavy-hex / Sycamore，零微调", False),
        ("图5.7：三拓扑 2Q 误差加权均值均显著低于盲目 O1", False),
    ])
    add_omml(s, Cm(0.8), Cm(4.9), Cm(6.4), "式(2-2) 有效距离", F22)
    add_figure(s, FIG / "图5.7_跨拓扑噪声规避.png", Cm(7.5), Cm(1.6), Cm(5.3),
               "图5.7 跨拓扑噪声规避（同一策略零重训迁移）")

    # 第 7 页
    s = new_slide(prs, "训练过程与模型侧改进")
    bullets(s, Cm(0.8), Cm(1.6), Cm(6.2), Cm(4.8), [
        ("两次独立训练一致收敛；单轮约 3 秒（RTX 5090）", True),
        ("LAUREL 可学习残差：收敛 400 轮 → 89 轮，约 5 分钟", True),
        ("效果不降反升：保真度 +3.1%，21/30 条更优", False),
        ("QuEst 风格预测器：T1/T2 退相干纳入保真度估计", False),
    ])
    add_figure(s, FIG / "图5.2_LAUREL收敛效率.png", Cm(7.3), Cm(1.6), Cm(5.5),
               "图5.2 LAUREL 训练效率与效果对比")

    # 第 8 页
    s = new_slide(prs, "数据集与评测体系")
    bullets(s, Cm(0.8), Cm(1.6), Cm(6.2), Cm(4.8), [
        ("Benchpress（Qiskit 官方测评集，656 条）+ MQTBench + QUEKO", True),
        ("587 条分层评测：≤10 / 11-50 / 51-105 三层", True),
        ("三种统计口径：均值 / 截尾 / 中位，原始数据逐线路落盘", False),
        ("自建 2 万条 287 原生池（含已知最优映射标注）", False),
    ])
    # 数据集构成小表（视觉元素）
    rows = [("数据集", "规模", "用途"),
            ("Benchpress", "656 条", "主评测集"),
            ("MQTBench", "36 类", "评测 + 训练图池"),
            ("QUEKO", "900 条", "评测 + 训练图池"),
            ("287 原生池", "20020 条", "训练图池")]
    from pptx.util import Inches
    tbl = s.shapes.add_table(5, 3, Cm(7.3), Cm(1.8), Cm(5.2), Cm(4.4)).table
    tbl.columns[0].width = Cm(2.2)
    tbl.columns[1].width = Cm(1.4)
    tbl.columns[2].width = Cm(1.6)
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = cell_text
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    _set_font(run, 11, bold=(i == 0), color=INK if i else INK2)
    add_text(s, Cm(7.3), Cm(6.35), Cm(5.2), Cm(0.6),
             "表5.1（节选）评测数据集构成", size=10, color=MUTED,
             align=PP_ALIGN.CENTER)

    # 第 9 页
    s = new_slide(prs, "分层性能")
    bullets(s, Cm(0.8), Cm(1.6), Cm(5.8), Cm(4.6), [
        ("≤10 层：保真度 0.619，显著高于盲目 O1（0.454）与 O3（0.505）", True),
        ("11-50 层：0.129 超过 pytket（0.105）与全部盲目基线", True),
        ("51-105 层为短板（0.0089），差距根因已定位（见第 13 页）", False),
    ])
    add_figure(s, FIG / "图5.3_分层保真度对比.png", Cm(6.9), Cm(1.6), Cm(5.9),
               "图5.3 分层估计保真度对比（587 条，均值口径）")

    # 第 10 页
    s = new_slide(prs, "深度与运行时间")
    bullets(s, Cm(0.8), Cm(1.6), Cm(6.2), Cm(4.6), [
        ("深度差距 = 残余 SWAP 展开：大层约 8.7 倍（截尾口径）", True),
        ("单条映射耗时 2.5-27.7 秒，满足分钟级目标", True),
        ("耗时比约 51 倍为工程差距，如实呈现", False),
    ])
    add_figure(s, FIG / "图5.5_运行时间对比.png", Cm(7.3), Cm(1.6), Cm(5.5),
               "图5.5 运行时间对比（51-105 比特，对数刻度）")

    # 第 11 页
    s = new_slide(prs, "噪声感知证据")
    bullets(s, Cm(0.8), Cm(1.6), Cm(6.2), Cm(4.6), [
        ("缺陷放大 ×5：盲目 O1 的 2Q 误差从 0.0035 恶化到 0.0110", True),
        ("本方案稳定在 0.0030-0.0033，与感知 O3（0.0027）接近", True),
        ("机制：奖励塑形 + 设备特征自归一化，缺一不可", False),
    ])
    add_figure(s, FIG / "图5.6_缺陷敏感性扫描.png", Cm(7.3), Cm(1.6), Cm(5.3),
               "图5.6 缺陷敏感性扫描（噪声感知鲁棒性证据）")

    # 第 12 页
    s = new_slide(prs, "与成熟方案对比")
    bullets(s, Cm(0.8), Cm(1.6), Cm(5.8), Cm(4.6), [
        ("总体保真度 0.196：超过 pytket（0.187）与全部盲目基线", True),
        ("胜场：对感知 O3 胜 29，对 pytket 胜 134，对盲目 O1 胜 226", True),
        ("为感知 O3 的 0.80 倍——差距集中在大线路层，如实呈现", False),
    ])
    add_figure(s, FIG / "图5.8_总体对比.png", Cm(6.9), Cm(1.6), Cm(5.9),
               "图5.8 与成熟映射方案总体对比（587 条线路）")

    # 第 13 页
    s = new_slide(prs, "布局贡献与结论展望")
    bullets(s, Cm(0.8), Cm(1.6), Cm(5.8), Cm(4.6), [
        ("路由器相同条件下，本方案布局保真度三基准全面第一", True),
        ("Benchpress：0.068 vs O3 0.003（约 21 倍）；MQTBench 0.279；QUEKO 0.410", True),
        ("结论：噪声感知布局路线成立；展望：门级再合成 + 学习化路由", True),
    ])
    add_figure(s, FIG / "图5.10_布局贡献对比.png", Cm(6.9), Cm(1.6), Cm(5.9),
               "图5.10 布局贡献对比（三基准，截尾均值）")

    # ---------------- 致谢（不计时）
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(s, 0, 0, Emu(EMU_W), Emu(EMU_H), SURFACE)
    add_rect(s, 0, Cm(3.0), Emu(EMU_W), Cm(0.12), BLUE)
    add_text(s, Cm(1.5), Cm(3.5), Cm(10.3), Cm(1.6),
             "谢谢聆听，欢迎提问",
             size=30, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, Cm(1.5), Cm(5.2), Cm(10.3), Cm(0.8),
             "全部实验数据与代码可复现｜详见作品报告书与仓库",
             size=14, color=INK2, align=PP_ALIGN.CENTER)

    prs.save(str(OUT))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    build()
