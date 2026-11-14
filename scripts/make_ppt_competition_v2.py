# -*- coding: utf-8 -*-
"""QTrial competition deck v2 — template-matched, text-dense.

Changes vs v1:
- 21 slides (the template's 20 minus 2 duplicated TOC pages, plus 3
  duplicated content pages for the RL-layout deep dive).
- Every content page carries 300-450 Chinese characters, close to the
  reference deck's density.
- The sparse-reward RL layout is promoted to a four-slide core section:
  2.2 why sparse reward, 2.3 MDP, 2.4 GAT+pointer, 2.5 training/search.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "delivery" / "作品展示PPT_案例.pptx"
BASE21 = ROOT / "out" / "_ppt_base21.pptx"
OUT = ROOT / "delivery" / "QTrail_竞赛展示_模板版v2.pptx"
FIG = ROOT / "assets" / "ppt_figs"
EX = ROOT / "assets" / "existing_ppt"

NAVY = "002060"
RED = "C00000"
YELLOW = "FFC000"
BLUE = "006CC4"
LIGHT = "F3F7FB"
GRID = "D9E2F3"
GRAY = "595959"
MID = "7F7F7F"
DARK = "333333"
WHITE = "FFFFFF"
GREEN = "2A9D8F"
GOLD = "B8860B"

TOTAL = 21


# ---------------------------------------------------------------- helpers
def rgb(hexstr: str) -> RGBColor:
    return RGBColor.from_string(hexstr.replace("#", "").upper())


def set_font(run, size=12, bold=False, color=DARK, name="微软雅黑",
             italic=False):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = rgb(color)
    f.name = name
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = OxmlElement("a:ea")
        rPr.append(ea)
    ea.set("typeface", name)


def add_text(slide, x, y, w, h, text, size=12, bold=False, color=DARK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             line_spacing=1.08, space_after=0, name="微软雅黑",
             margins=(0.02, 0.02, 0.02, 0.02), italic=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(margins[0])
    tf.margin_right = Inches(margins[1])
    tf.margin_top = Inches(margins[2])
    tf.margin_bottom = Inches(margins[3])
    for i, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        r = p.add_run()
        r.text = line
        set_font(r, size, bold, color, name, italic)
    return box


def add_runs(slide, x, y, w, h, paragraphs, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, line_spacing=1.08, space_after=0,
             margins=(0.02, 0.02, 0.02, 0.02)):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(margins[0])
    tf.margin_right = Inches(margins[1])
    tf.margin_top = Inches(margins[2])
    tf.margin_bottom = Inches(margins[3])
    for i, runs in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        for rd in runs:
            r = p.add_run()
            r.text = rd["t"]
            set_font(r, rd.get("size", 11), rd.get("bold", False),
                     rd.get("color", DARK), rd.get("name", "微软雅黑"),
                     rd.get("italic", False))
    return box


def add_bullets(slide, x, y, w, h, items, size=10.6, lead_color=NAVY,
                body_color=DARK, line_spacing=1.12, space_after=4,
                bullet="●"):
    """items: list of (bold lead, body) or plain strings."""
    paras = []
    for item in items:
        if isinstance(item, tuple):
            lead, body = item
            paras.append([
                {"t": bullet + " ", "size": size, "color": lead_color},
                {"t": lead, "size": size, "bold": True, "color": lead_color},
                {"t": body, "size": size, "color": body_color},
            ])
        else:
            paras.append([
                {"t": bullet + " ", "size": size, "color": lead_color},
                {"t": item, "size": size, "color": body_color},
            ])
    return add_runs(slide, x, y, w, h, paras,
                    line_spacing=line_spacing, space_after=space_after)


def add_rect(slide, x, y, w, h, fill=None, line=None, line_width=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            shp.adjustments[0] = radius
        except Exception:
            pass
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(line_width)
    shp.shadow.inherit = False
    shp.text_frame.word_wrap = True
    shp.text_frame.margin_left = Inches(0.04)
    shp.text_frame.margin_right = Inches(0.04)
    shp.text_frame.margin_top = Inches(0.02)
    shp.text_frame.margin_bottom = Inches(0.02)
    return shp


def add_line(slide, x1, y1, x2, y2, color=GRAY, width=1.0, dash=None,
             arrow=False):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                      Inches(x1), Inches(y1),
                                      Inches(x2), Inches(y2))
    conn.line.color.rgb = rgb(color)
    conn.line.width = Pt(width)
    if dash is not None:
        conn.line.dash_style = dash
    if arrow:
        ln = conn.line._get_or_add_ln()
        tail = OxmlElement("a:tailEnd")
        tail.set("type", "triangle")
        tail.set("w", "med")
        tail.set("len", "med")
        ln.append(tail)
    return conn


def add_circle_num(slide, x, y, d, num, fill=NAVY, color=WHITE, size=10):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y),
                                 Inches(d), Inches(d))
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(num)
    set_font(r, size, True, color)
    return shp


def add_picture_fit(slide, path, x, y, w, h, border=None):
    path = Path(path)
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    nw, nh = iw * scale, ih * scale
    left = x + (w - nw) / 2
    top = y + (h - nh) / 2
    pic = slide.shapes.add_picture(str(path), Inches(left), Inches(top),
                                   Inches(nw), Inches(nh))
    if border:
        pic.line.color.rgb = rgb(border)
        pic.line.width = Pt(0.75)
    return pic


def add_table(slide, x, y, w, h, data, col_widths, font_size=9.8,
              header_fill=NAVY):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, Inches(x), Inches(y),
                                   Inches(w), Inches(h))
    table = shape.table
    table.first_row = True
    table.horz_banding = False
    for i, cw in enumerate(col_widths):
        table.columns[i].width = Inches(cw)
    for r, row in enumerate(data):
        table.rows[r].height = Inches(h / rows)
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(header_fill if r == 0 else
                                           (LIGHT if r % 2 == 0 else WHITE))
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 1 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(val)
            set_font(run, font_size, bold=(r == 0 or c == 0),
                     color=(WHITE if r == 0 else DARK))
    return table


def add_footer(slide, page):
    add_line(slide, 0.55, 6.98, 12.78, 6.98, color="D9D9D9", width=0.75)
    add_text(slide, 0.55, 7.04, 8.0, 0.24,
             "QTrial · 稀疏奖励强化学习布局 · 零 qiskit 交付",
             size=8, color="808080")
    add_text(slide, 11.55, 7.04, 1.25, 0.24, f"{page:02d} / {TOTAL}",
             size=8, color="808080", align=PP_ALIGN.RIGHT)


def clear_content(slide):
    for sh in list(slide.shapes):
        top = sh.top / 914400
        h = sh.height / 914400
        w = sh.width / 914400
        left = sh.left / 914400
        keep = False
        if top < 1.0 and h < 1.0:
            keep = True
        elif 1.1 <= top <= 1.62 and h <= 0.65:
            if w > 12 or (left < 0.35 and w < 0.25):
                keep = True
        if not keep:
            sh._element.getparent().remove(sh._element)
    from lxml import etree
    sp_tree = slide.shapes._spTree
    for child in list(sp_tree):
        if etree.QName(child).localname == "AlternateContent":
            sp_tree.remove(child)


def set_header(slide, section, number):
    section_no = str(number).split(".")[0]
    for sh in slide.shapes:
        if sh.shape_type == 6 and sh.top / 914400 < 1.0:
            for ch in sh.shapes:
                if ch.has_text_frame:
                    runs = [r for p in ch.text_frame.paragraphs for r in p.runs]
                    if len(runs) >= 2:
                        runs[0].text = f"{section_no}. "
                        runs[1].text = section
                        return True
    return False


def set_subheader(slide, number, title):
    for sh in slide.shapes:
        if sh.has_text_frame and 1.1 <= sh.top / 914400 <= 1.62 \
                and sh.width / 914400 > 12:
            runs = [r for p in sh.text_frame.paragraphs for r in p.runs]
            if len(runs) >= 2:
                runs[0].text = f" {number} "
                runs[1].text = title
                return True
    add_rect(slide, 0.14, 1.21, 13.06, 0.44, fill=NAVY, line=None,
             shape=MSO_SHAPE.RECTANGLE)
    add_rect(slide, 0.14, 1.21, 0.10, 0.44, fill=YELLOW, line=None,
             shape=MSO_SHAPE.RECTANGLE)
    add_text(slide, 0.24, 1.21, 12.8, 0.44, f" {number}  {title}",
             size=18, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    return True


def content_slide(slide, section, number, subheader, title, subtitle, page,
                  title_size=27, title_y=1.68, subtitle_y=2.18):
    clear_content(slide)
    set_header(slide, section, number)
    set_subheader(slide, number, subheader)
    add_text(slide, 0.55, title_y, 12.2, 0.52, title, size=title_size,
             bold=True, color=NAVY)
    if subtitle:
        add_text(slide, 0.55, subtitle_y, 12.2, 0.38, subtitle, size=13.5,
                 color=GRAY)
    add_footer(slide, page)


def takeaway(slide, text, y=6.42, h=0.40, size=11.5):
    add_rect(slide, 0.55, y, 12.23, h, fill="FFF7E6", line=YELLOW,
             line_width=0.9)
    add_text(slide, 0.72, y + 0.02, 11.9, h - 0.04, text, size=size,
             bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)


def label_box(slide, x, y, w, h, text, fill=NAVY, color=WHITE, size=9):
    add_rect(slide, x, y, w, h, fill=fill, line=None,
             shape=MSO_SHAPE.RECTANGLE)
    add_text(slide, x, y, w, h, text, size=size, bold=True, color=color,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def prepare_base():
    """Delete the two extra TOC pages, then clone slide 7 three times."""
    BASE21.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, BASE21)
    import win32com.client
    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = 1
    pres = app.Presentations.Open(str(BASE21), WithWindow=False)
    pres.Slides(16).Delete()
    pres.Slides(5).Delete()
    for _ in range(3):
        pres.Slides(6).Duplicate()
    pres.Save()
    pres.Close()
    app.Quit()
    return BASE21


# ---------------------------------------------------------------- slides
def slide_cover(s):
    title_box = author_box = None
    for sh in s.shapes:
        if sh.has_text_frame and "博士资格考试答辩" in sh.text_frame.text:
            title_box = sh
        if sh.has_text_frame and "答辩人" in sh.text_frame.text:
            author_box = sh
    if title_box is not None:
        tf = title_box.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = "2026 年“揭榜挂帅”擂台赛 · 榜题：量子 + AI 双向赋能"
        set_font(r, 14, False, BLUE)
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = "QTrial"
        set_font(r2, 52, True, RED)
        p3 = tf.add_paragraph()
        p3.alignment = PP_ALIGN.CENTER
        r3 = p3.add_run()
        r3.text = "自适应量子线路映射优化系统"
        set_font(r3, 28, True, NAVY)
        p4 = tf.add_paragraph()
        p4.alignment = PP_ALIGN.CENTER
        r4 = p4.add_run()
        r4.text = "核心创新：稀疏奖励强化学习布局 · SWIFT 路由 · 零 qiskit 交付"
        set_font(r4, 13, False, GOLD)
    if author_box is not None:
        tf = author_box.text_frame
        tf.clear()
        for i, line in enumerate([
            "团队成员：何欣航  夏聆心  袁一茗",
            "指导教师：尚涛",
            "2026 年 11 月",
        ]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = line
            set_font(r, 19, False, NAVY)
    photo_names = {"图片 1", "图片 12", "图片 13", "图片 18", "图片 20",
                   "图片 21"}
    positions = [
        (-0.01, 5.87, 2.25, 1.63), (2.27, 5.87, 2.20, 1.63),
        (4.50, 5.85, 2.46, 1.65), (6.99, 5.85, 1.35, 1.65),
        (8.37, 5.86, 2.44, 1.64), (10.84, 5.86, 2.47, 1.64),
    ]
    for sh in list(s.shapes):
        if sh.name in photo_names:
            sh._element.getparent().remove(sh._element)
    for i, (x, y, w, h) in enumerate(positions, 1):
        add_picture_fit(s, FIG / f"cover_{i}.png", x, y, w, h)
    notes(s, "各位评委老师好，我们是 QTrial 团队。我们的核心创新是把量子比特布局写成"
             "稀疏终局奖励的序列决策，用 GAT + 指针网络学习；后面接 SWIFT 路由和置换重写，"
             "整条交付管线零 qiskit。")


def slide_toc(s):
    labels = ["研究背景", "核心方法", "实验与价值"]
    originals = {"个人介绍": labels[0], "研究内容": labels[1],
                 "研究进展": labels[2]}
    for sh in s.shapes:
        if sh.shape_type == 6:
            for ch in sh.shapes:
                if ch.has_text_frame:
                    for p in ch.text_frame.paragraphs:
                        for r in p.runs:
                            if r.text in originals:
                                r.text = originals[r.text]
    add_text(s, 1.55, 5.95, 10.25, 0.72,
             "核心方法分四层：稀疏奖励 RL 布局 → 多起点搜索 → SWIFT 路由 → 置换重写。\n"
             "布局层是本项目的核心创新，占方法部分的一半篇幅。",
             size=13.5, color=GRAY, align=PP_ALIGN.CENTER, line_spacing=1.25)
    notes(s, "汇报分三部分：为什么映射难、我们的核心方法、做得怎么样。"
             "核心方法里布局层是重点，会展开讲四页。")


def slide_problem(s):
    content_slide(
        s, "研究背景", "1.1",
        "量子线路映射：NISQ 时代的刚需",
        "一条逻辑线路，不能直接跑在真实芯片上",
        "拓扑把任意两比特交互变成“必须相邻”的硬约束；布局差一点，路由就要多付很多 SWAP。",
        3,
        title_size=27,
    )
    add_bullets(s, 0.55, 2.62, 5.75, 3.55, [
        ("耦合图约束：", "超导芯片只在耦合边两端执行 CX/CZ；逻辑线路里任意两比特都可能有双比特门，"
                    "一旦落到非相邻物理比特，就必须插入 SWAP。一次 SWAP 在硬件上要拆成 3 个 CX"
                    "（3 个 CZ + 6 个 H），是路由成本的主要来源。"),
        ("两个子问题：", "布局（layout）给每个逻辑比特选一个物理位置，是一个单射 π；路由（routing）"
                    "在布局给定后插入 SWAP，使所有双比特门在执行时刻满足邻近性。布局决定“要搬多远”，"
                    "路由决定“怎么搬”。"),
        ("设备口径：", "天衍-287 规格为 105 比特、15×7 网格、182 耦合器；评测模型采用 15×7 完整网格"
                  "（188 条边）+ 合成校准。真机是弯曲排布、168 条可用边、4 个禁用比特，两者口径分开。"),
    ], size=11.0, space_after=7)
    chain = ["布局差", "SWAP 多", "深度大", "保真度低"]
    cx = 0.55
    for i, text in enumerate(chain):
        add_rect(s, cx, 5.58, 1.28, 0.38, fill=LIGHT, line=GRID, line_width=0.7)
        add_text(s, cx, 5.58, 1.28, 0.38, text, size=9.2, bold=True,
                 color=RED if i == 0 else NAVY, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)
        if i < len(chain) - 1:
            add_line(s, cx + 1.29, 5.77, cx + 1.39, 5.77, color=MID,
                     width=1.1, arrow=True)
        cx += 1.42
    add_picture_fit(s, FIG / "fig_problem.png", 6.45, 2.60, 6.35, 3.45)
    takeaway(s, "布局是“因”，路由是“果”：布局阶段多算一步，路由阶段就能少插很多交换。",
             y=6.25, size=11)
    notes(s, "先看左边这条三比特线路：q0 和 q2 之间有一个 CX，放到 3×3 网格上隔了四跳。"
             "芯片只允许相邻比特做双比特门，所以必须插 SWAP。一个 SWAP 是三个 CX，"
             "跳数越多，代价越大。映射因此分成布局和路由两个子问题。")


def slide_qap(s):
    content_slide(
        s, "研究背景", "1.2",
        "问题建模：从线路到 QAP",
        "把布局写成加权图匹配：交互越多，物理距离越要小",
        "QAP 是 NP-难的组合优化；噪声感知把“距离”从拓扑距离换成有效距离，让布局学会避开热区。",
        4,
        title_size=26,
    )
    add_bullets(s, 0.55, 2.60, 6.05, 3.45, [
        ("静态代价：", "程序图 G_C 的节点是逻辑比特，边权 w_ij 是两比特之间的双比特门数；"
                   "设备图 G_D 的 d(·,·) 是 BFS 最短距离。布局 π 的代价 "
                   "C(π) = 2 Σ w_ij · d(π(i), π(j))。"),
        ("数理本质：", "C(π) 是置换群上的高维非凸函数，精确求解是 QAP，NP-难；105 比特状态空间爆炸。"
                   "传统谱方法/模拟退火只利用拓扑结构，本文用神经网络学习“程序图结构 → 好置换”的映射，"
                   "把组合搜索压进前向推理。"),
        ("噪声感知：", "每条耦合器的 ε2q、每个比特的 T1/T2 都不同。直接把误差塞进代价会让奖励景观过粗，"
                   "所以用距离插值 D_eff = D_topo + λ(D_noise − D_topo)，λ=0.5；噪声边权 "
                   "w_e = 1 + α(φ_e−1) + β·T1_pen，α=0.5、β=0.1。"),
    ], size=10.7, space_after=6)
    add_rect(s, 0.55, 5.55, 6.05, 0.48, fill=LIGHT, line=GRID, line_width=0.7)
    add_text(s, 0.70, 5.57, 5.80, 0.44,
             "直觉：把交互最频繁的比特对放到最短的边上；噪声感知再把“边”换成“有效距离”。",
             size=9.2, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    add_picture_fit(s, FIG / "fig_qap.png", 6.70, 2.60, 6.10, 3.25)
    add_rect(s, 0.55, 6.15, 12.23, 0.48, fill=LIGHT, line=GRID, line_width=0.7)
    add_text(s, 0.72, 6.18, 11.9, 0.42,
             "静态代价只是路由成本的代理；真正的胜负要在路由后看产物——所以后面还要候选池实测决胜。",
             size=10.5, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "把线路写成加权交互图，把芯片写成耦合图，布局就是一个双射。"
             "目标函数是交互次数乘物理距离。这个模型是 QAP，NP 难。"
             "噪声感知不是把误差直接塞进代价，而是用它扰动距离，保持奖励景观平滑。")


def slide_opportunity(s):
    content_slide(
        s, "研究背景", "1.3",
        "现有方法与本项目定位",
        "现有编译器的规则是静态的，芯片的噪声是动态的",
        "三条路线各有取舍；我们把学习放在最需要数据、也最能迁移的布局层，而不是做端到端黑箱。",
        5,
        title_size=26,
    )
    cols = [
        ("精确解", "ILP / SMT / 分支定界", "可证最优",
         "小规模能给出最优解，适合做离线基准；105 比特的状态空间爆炸，无法在线编译，"
         "对真实线路只能做小规模验证。", MID),
        ("启发式", "SABRE / O3 / tket", "毫秒级",
         "工程成熟、可解释、在线可用；但规则和参数是静态的，换芯片或换校准要重调；"
         "布局层通常只看拓扑或静态代价，路由后的真实成本没有反馈。", BLUE),
        ("学习式", "GAT + 指针 + 稀疏奖励", "自适应",
         "从数据学“程序图结构 → 好置换”的映射；校准特征按设备自身归一化，跨设备零重训；"
         "但纯学习布局可能被代理代价带偏，所以后面用候选池实测决胜。", NAVY),
    ]
    x = 0.55
    for name, tool, tag, desc, color in cols:
        add_rect(s, x, 2.58, 3.85, 3.30, fill=WHITE, line=GRID, line_width=0.9)
        add_rect(s, x, 2.58, 3.85, 0.07, fill=color, line=None,
                 shape=MSO_SHAPE.RECTANGLE)
        add_text(s, x + 0.20, 2.72, 3.45, 0.30, name, size=16, bold=True,
                 color=color)
        add_text(s, x + 0.20, 3.06, 3.45, 0.24, tool, size=9.8, bold=True,
                 color=DARK)
        add_text(s, x + 0.20, 3.38, 3.45, 0.24, tag, size=11, bold=True,
                 color=color)
        add_text(s, x + 0.20, 3.70, 3.45, 1.95, desc, size=9.3, color=GRAY,
                 line_spacing=1.15)
        x += 4.14
    conclusions = [("结论：不可扩展", MID), ("结论：快但静态", BLUE),
                   ("结论：可学习、可迁移", NAVY)]
    cx = 0.55
    for text, color in conclusions:
        add_text(s, cx + 0.20, 5.50, 3.45, 0.28, text, size=10.8,
                 bold=True, color=color)
        cx += 4.14
    add_rect(s, 0.55, 6.02, 12.23, 0.62, fill=LIGHT, line=GRID, line_width=0.8)
    add_text(s, 0.72, 6.05, 11.9, 0.56,
             "本文定位：不替代 SABRE，也不做端到端黑箱；学习只放在布局层，路由仍用可解释的 SWIFT，"
             "后处理用代数重写保证非劣化。",
             size=11, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "三类方法各有取舍。精确解只能做小规模，启发式快但静态，学习式能自适应但可能被代理代价带偏。"
             "我们的做法是把学习限制在布局层，后面用真实路由实测来兜底。")


def slide_pipeline(s):
    content_slide(
        s, "核心方法", "2.1",
        "总体方案：四段管线，三处自研",
        "布局学、路由快、后处理赚：一条从 QASM 到 QCIS 的零 qiskit 管线",
        "解析和图构建复用公开规范；三处自研集中在布局、路由、后处理；训练链与推理链共享同一套布局策略。",
        6,
        title_size=24,
    )
    # compact four-column pipeline
    cols = [
        ("输入与建模", NAVY, [
            ("OpenQASM 2.0", "解析 / 依赖 DAG"),
            ("程序交互图", "6 维节点特征"),
            ("设备耦合图", "7 维校准特征"),
        ]),
        ("稀疏奖励 RL 布局", BLUE, [
            ("GAT 双图编码", "4 层 × 8 头 × 128 维"),
            ("指针逐比特解码", "16 头 + 占用掩码"),
            ("稀疏终局奖励", "−C(π)/Σw + 深度 + 紧凑"),
        ]),
        ("SWIFT 路由", RED, [
            ("候选布局池", "多起点 + 局部搜索"),
            ("前沿 + 扩展集", "前瞻评分，少插交换"),
            ("16 种子取优", "最终路由永不截断"),
        ]),
        ("后处理与交付", NAVY, [
            ("置换重写", "共轭 / 对消 / VF2++"),
            ("平台基分解", "[rz, sx, x, cz]"),
            ("QASM / QCIS", "metrics + 校验"),
        ]),
    ]
    x0, col_w, gap = 0.55, 2.78, 0.37
    for ci, (header, color, boxes) in enumerate(cols):
        x = x0 + ci * (col_w + gap)
        label_box(s, x, 2.52, col_w, 0.32, header, fill=color, size=10.5)
        for bi, (label, sub) in enumerate(boxes):
            y = 2.96 + bi * 0.66
            add_rect(s, x, y, col_w, 0.54, fill=WHITE, line=GRID, line_width=0.8)
            add_text(s, x + 0.06, y + 0.04, col_w - 0.12, 0.24, label,
                     size=9.4, bold=True, color=color, align=PP_ALIGN.CENTER)
            add_text(s, x + 0.06, y + 0.28, col_w - 0.12, 0.22, sub,
                     size=7.4, color=GRAY, align=PP_ALIGN.CENTER)
        if ci < 3:
            add_line(s, x + col_w + 0.03, 4.10, x + col_w + gap - 0.03, 4.10,
                     color=MID, width=1.2, arrow=True)
    add_line(s, 11.45, 5.02, 11.45, 5.30, color=YELLOW, width=1.2)
    add_line(s, 11.45, 5.30, 4.80, 5.30, color=YELLOW, width=1.2)
    add_line(s, 4.80, 5.30, 4.80, 5.02, color=YELLOW, width=1.2, arrow=True)
    add_text(s, 5.0, 5.34, 6.2, 0.22,
             "训练链：REINFORCE + 自批评基线，89 轮收敛",
             size=7.8, color=GOLD, italic=True)
    # two-column explanatory text
    add_bullets(s, 0.55, 5.62, 6.0, 1.15, [
        ("布局段：", "程序图/设备图 → GAT 双图编码 → 指针逐比特解码 → 稀疏终局奖励 → "
                  "多起点采样 + 两阶段局部搜索，输出一组候选布局而不是一个布局。"),
        ("路由段：", "候选逐个交给 SWIFT 实测，按 swap/fidelity/depth 三规则决胜；"
                  "胜者全预算最终路由，永不截断。"),
    ], size=8.8, space_after=3)
    add_bullets(s, 6.75, 5.62, 6.0, 1.15, [
        ("后处理段：", "SWAP 共轭推挤 → 恒等对消 → 尾部吸收 → 平台基 [rz,sx,x,cz]，"
                   "最后用非劣化开关兜底。"),
        ("训练/推理对齐：", "训练看 QAP 代价，推理看路由后产物；候选决胜把两者口径对齐，"
                       "避免“训练指标好、真机产物差”。"),
    ], size=8.8, space_after=3)
    notes(s, "整条管线四段：输入建模、RL 布局、SWIFT 路由、后处理交付。"
             "三处自研是布局、路由、后处理；解析和图构建复用公开算法。"
             "训练链和推理链共享同一套布局策略，用候选决胜对齐口径。")


def slide_core_innovation(s):
    content_slide(
        s, "核心方法", "2.2",
        "核心创新：为什么是稀疏奖励 RL 布局",
        "不给中间步骤打分，只给完整布局一次奖励",
        "布局的输出是一个排列，排列没有可信的“中间标签”；强行加 shaping，奖励设计本身会把策略带偏。",
        7,
        title_size=26,
    )
    add_bullets(s, 0.55, 2.58, 7.0, 3.55, [
        ("问题：", "布局是 QAP，输出是一个排列。放了一半的布局，很难判断好坏；如果给中间步骤加密集奖励，"
                "等于把人工先验塞进策略，模型可能学会“讨好 shaping”而不是最小化真实代价。"),
        ("做法：", "把布局写成 MDP——状态是已放置集合 + 程序图/设备图嵌入，动作是给下一个逻辑比特选位置，"
                "奖励只在终局给一次 R = −C(π)/Σw；深度项、紧凑性项也都是终局一次性项，不破坏稀疏结构。"),
        ("代价与解法：", "稀疏奖励方差大。我们用自批评贪心基线 b 复用策略本身，零额外价值网络；"
                   "再用多起点采样（5 采样 + 4 乱序贪心 + 1 普通贪心）和两阶段局部搜索把方差压住。"),
        ("结果：", "策略学的是“程序图结构 → 好置换”的映射，而不是拟合某个静态中间值；"
                 "同一套权重可以跨芯片迁移，换校准不换权重。"),
    ], size=10.5, space_after=8)
    cards = [
        ("01", "稀疏终局奖励", "只对完整布局打分，避免中间 shaping 偏差；奖励按 Σw 归一化，5–105 比特同量纲。"),
        ("02", "自批评基线", "基线 b 取同批贪心 rollout，复用策略本身，零额外网络、低方差、实现极简。"),
        ("03", "多起点 + 局部搜索", "一次解码变一组候选：采样/乱序贪心去重，再跑两阶段 LS 精化。"),
    ]
    y = 2.58
    for num, title, body in cards:
        add_rect(s, 7.85, y, 4.95, 1.02, fill=LIGHT, line=GRID, line_width=0.8)
        add_circle_num(s, 7.98, y + 0.14, 0.34, num, fill=RED if num == "01" else NAVY, size=8)
        add_text(s, 8.42, y + 0.10, 4.20, 0.26, title, size=11.5, bold=True,
                 color=NAVY)
        add_text(s, 8.42, y + 0.38, 4.20, 0.58, body, size=8.6, color=GRAY,
                 line_spacing=1.05)
        y += 1.12
    add_rect(s, 0.55, 5.35, 7.0, 0.58, fill=LIGHT, line=GRID,
             line_width=0.7)
    add_text(s, 0.70, 5.38, 6.70, 0.52,
             "为什么不用监督学习：QAP 没有大规模最优标签；REINFORCE 直接优化真实代价，"
             "不需要人工标注。",
             size=9.0, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    add_rect(s, 7.85, 5.98, 4.95, 0.46, fill="FFF7E6", line=YELLOW,
             line_width=0.8)
    add_text(s, 7.98, 6.00, 4.70, 0.42,
             "证据：89 轮收敛 · 3.2 s/轮 · ~60 万参数 · 跨拓扑 1.37–4.34×",
             size=8.4, bold=True, color=GOLD, anchor=MSO_ANCHOR.MIDDLE)
    takeaway(s, "核心创新不是“用了 RL”，而是“只给终局奖励，再用基线和多起点把方差压住”——"
                "这让策略学结构，而不是学中间标签。", y=6.55, h=0.34, size=9.6)
    notes(s, "这一页是整场的核心。布局的输出是一个排列，排列没有天然的中间标签。"
             "我们不搞 dense shaping，只在完整布局时给一次 QAP 代价。"
             "稀疏奖励的方差用自批评基线、多起点和局部搜索来压。"
             "这样策略学的是结构，不是人工设计的中间奖励。")


def slide_mdp(s):
    content_slide(
        s, "核心方法", "2.3",
        "核心创新 1/4：把布局写成 MDP",
        "状态是部分布局，动作是选一个位置，奖励只在终点给一次",
        "状态、动作、奖励三件事定义清楚，稀疏奖励才有可能训得动。",
        8,
        title_size=25,
    )
    add_bullets(s, 0.55, 2.58, 6.05, 3.45, [
        ("状态 S_t：", "已放置集合 + 当前逻辑比特的程序图嵌入 + 设备图嵌入；设备节点带 7 维校准特征"
                   "（T1、T2、1Q 误差、读出误差、度、行、列），按设备自身 min-max 归一化。"),
        ("动作 A_t：", "为下一个逻辑比特选一个未占用的物理位置；已占用位置用 −∞ 掩码，保证布局是双射。"
                   "解码顺序固定，但多起点里用乱序贪心打破顺序偏差。"),
        ("奖励 R：", "t<T 时恒为 0；t=T 时给 R = −C(π)/Σw 一次；深度感知变体加 λ_d·Σh²/Σh，"
                  "紧凑性变体加 λ_c·diam，都是终局一次性项，不破坏稀疏结构。"),
        ("为什么有效：", "稀疏奖励把“布局好坏”的定义权交给 QAP 代价本身，不需要人工设计中间奖励；"
                    "代价是方差，用自批评基线和多起点压。"),
    ], size=10.6, space_after=7)
    add_picture_fit(s, EX / "slide07_1.png", 6.75, 2.55, 5.95, 2.95)
    add_rect(s, 6.75, 5.62, 5.95, 0.62, fill="FFF7E6", line=YELLOW,
             line_width=0.8)
    add_text(s, 6.90, 5.64, 5.65, 0.58,
             "R = − C(π) / Σw  +  λ_d · Σh²/Σh  +  λ_c · diam\n"
             "三项都在终局一次性结算，中间步骤不给任何信号。",
             size=9.2, color=NAVY, line_spacing=1.18)
    takeaway(s, "状态和动作决定“能不能生成排列”，奖励决定“排列好不好”；稀疏奖励把后者交还给 QAP 代价。",
             y=6.38, h=0.38, size=10.2)
    notes(s, "状态是部分布局，动作是给下一个逻辑比特选位置，奖励只在完整布局时给一次。"
             "深度项和紧凑性项也是终局一次性项，不破坏稀疏结构。"
             "中间不给信号，是为了不让策略去讨好人工设计的中间奖励。")


def slide_network(s):
    content_slide(
        s, "核心方法", "2.4",
        "核心创新 2/4：GAT 编码 + 指针解码",
        "两张图进去，一个排列出来：GAT 学结构，指针网络学“指向”",
        "同一套权重处理任意节点数；校准特征按设备自身统计量归一化，换芯片不换权重。",
        9,
        title_size=25,
    )
    add_bullets(s, 0.55, 2.58, 5.95, 3.55, [
        ("程序图编码器：", "4 层 GAT、8 头、d=128；注意力只在图边和自环上计算，头拼接后接残差、"
                     "GraphNorm1d、ReLU、dropout 0.1。GraphNorm 只统计非填充节点，保证变长线路的填充不变性。"),
        ("设备编码器：", "同样 4×8×128，但输入是 7 维校准特征；设备图 N=105 固定，编码一次广播到全批，"
                    "所以换设备只需要换特征，不需要换网络结构。"),
        ("指针解码器：", "16 头指针注意力，u_ij = C·tanh(q_iᵀk_j/√d)，C=10；对已占用位置做 −∞ 掩码后 softmax。"
                    "指针网络解决的是“输出是输入的一个排列”，普通全连接做不到。"),
        ("训练配置：", "REINFORCE + 自批评贪心基线；batch 512、Adam 3e-4 余弦退火、梯度裁剪 1.0、"
                   "早停 patience 20；模型约 60 万参数，RTX 5090 单轮 3.2 s。"),
    ], size=10.3, space_after=5)
    add_rect(s, 0.55, 5.55, 5.95, 0.48, fill=LIGHT, line=GRID, line_width=0.7)
    add_text(s, 0.70, 5.57, 5.70, 0.44,
             "可迁移性：GAT 不依赖固定节点数；指针输出排列；两者都不绑定某台芯片。",
             size=9.0, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    add_picture_fit(s, EX / "slide08_1.png", 6.70, 2.52, 6.05, 2.75)
    add_rect(s, 6.70, 5.42, 6.05, 0.82, fill=LIGHT, line=GRID, line_width=0.8)
    add_text(s, 6.84, 5.45, 5.80, 0.76,
             "α_ij = softmax_j( LeakyReLU( aᵀ[W h_i ‖ W h_j] ) )\n"
             "u_ij = C · tanh( q_iᵀ k_j / √d ),  C = 10",
             size=9.0, color=NAVY, line_spacing=1.22)
    takeaway(s, "GAT 把“图结构”编码成向量，指针把“选哪个物理比特”变成可采样动作；两者相加才是可迁移的布局策略。",
             y=6.36, h=0.38, size=9.8)
    notes(s, "编码器是 GAT，程序图和设备图各一个；解码器是指针网络，每步指向一个物理比特。"
             "模型只有 60 万参数，89 轮收敛，单轮 3.2 秒。"
             "同一套权重可以处理 5 到 105 个比特，因为 GAT 不依赖固定节点数。")


def slide_training(s):
    content_slide(
        s, "核心方法", "2.5",
        "核心创新 3/4：稀疏奖励怎么训得动",
        "自批评基线压方差，多起点搜索把一次解码变成一组候选",
        "稀疏奖励不是“不给奖励”，而是“只在终局给一次”；训练和推理各有一套抗方差手段。",
        10,
        title_size=25,
    )
    add_bullets(s, 0.55, 2.58, 6.55, 3.60, [
        ("梯度：", "REINFORCE 的梯度是 (R−b)∇log π；基线 b 取同批实例的贪心 rollout，复用策略本身，"
                "零额外网络，这就是自批评（Kool et al. 2019）。"),
        ("训练数据：", "每轮现生成 ER 随机图（p=0.3，规模 5–40 / 41–80 / 81–105 按 50/30/20），"
                   "再与预生成线路图池（MQTBench / QUEKO / QASMBench 派生图）按比例混合；"
                   "奖励按总交互权重归一化，5–105 比特同量纲。"),
        ("推理多起点：", "5 个采样解码 + 4 个乱序贪心 + 1 个普通贪心，按 QAP 代价去重取 top-10；"
                    "每个起点再跑两阶段局部搜索。"),
        ("两阶段 LS：", "前 30% 预算做大扰动（热比特 3 连交换、空位重分配），后 70% 做首改善单交换 + "
                   "嵌入定向交换；patience 50，总预算 5000 步/起点；交换只更新受影响项，每步 O(n)。"),
        ("为什么有效：", "学习负责给出好起点，局部搜索负责把起点精化到局部最优；只用学习或只用搜索都不够。"),
    ], size=10.0, space_after=6)
    # right: training loop diagram + stats
    add_rect(s, 7.28, 2.55, 5.45, 0.42, fill=NAVY, line=None,
             shape=MSO_SHAPE.RECTANGLE)
    add_text(s, 7.28, 2.55, 5.45, 0.42, "训练回路：一次轨迹，一次奖励",
             size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
    steps = [
        ("采样完整布局", "按当前策略逐比特放置"),
        ("终局奖励 R", "−C(π)/Σw + 深度 + 紧凑"),
        ("自批评基线 b", "同批贪心 rollout，无梯度"),
        ("策略梯度更新", "(R−b)·∇log π，Adam + 余弦退火"),
    ]
    y = 3.08
    for i, (title, body) in enumerate(steps):
        add_rect(s, 7.28, y, 5.45, 0.52, fill=WHITE, line=GRID, line_width=0.8)
        add_circle_num(s, 7.38, y + 0.11, 0.30, str(i + 1), fill=BLUE, size=8)
        add_text(s, 7.78, y + 0.05, 1.55, 0.22, title, size=9.0, bold=True,
                 color=NAVY)
        add_text(s, 9.35, y + 0.05, 3.25, 0.22, body, size=8.0, color=GRAY)
        if i < len(steps) - 1:
            add_line(s, 9.95, y + 0.53, 9.95, y + 0.60, color=MID, width=1.0,
                     arrow=True)
        y += 0.62
    add_rect(s, 0.55, 5.55, 6.55, 0.52, fill=LIGHT, line=GRID,
             line_width=0.7)
    add_text(s, 0.70, 5.57, 6.25, 0.48,
             "方差来源：整条轨迹只有一个标量——用基线减方差，用多起点增样本，用局部搜索做精化。",
             size=8.8, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    add_rect(s, 7.28, 5.68, 5.45, 0.46, fill=LIGHT, line=GRID, line_width=0.8)
    add_text(s, 7.40, 5.70, 5.20, 0.42,
             "4096 实例/轮 · 512 batch · 89 轮 · 3.2 s/轮 · ~60 万参数",
             size=8.2, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    takeaway(s, "稀疏奖励的代价是方差；解法不是偷偷加 shaping，而是自批评基线 + 多起点 + 局部搜索。",
             y=6.38, h=0.38, size=9.8)
    notes(s, "稀疏奖励的方差来自整条轨迹只有一个标量。我们用同批贪心 rollout 做基线，"
             "这就是自批评。推理时不止一条轨迹，采样和乱序贪心各跑几条，再去重取 top-10，"
             "每个起点跑两阶段局部搜索。学习给起点，搜索做精化。")


def slide_multistart(s):
    content_slide(
        s, "核心方法", "2.6",
        "核心创新 4/4：从布局到产物",
        "候选池 + 路由实测决胜：不让代理代价决定最终赢家",
        "学习布局的输出不是最终答案，而是一组候选；每个候选都要过一遍真实路由，再按三规则决胜。",
        11,
        title_size=25,
    )
    add_bullets(s, 0.55, 2.58, 6.55, 3.55, [
        ("候选池：", "RL 多起点 top-10、λ=0 拓扑候选、Fiedler 序/蛇形启发式、平凡布局；"
                 "全部是自研构造，不引入外部编译器输出。"),
        ("实测决胜：", "每个候选交给 SWIFT 路由器实测，而不是只看 QAP 静态代价；"
                  "swap 规则在 SWAP 数容差 2 内取保真度最高，fidelity 规则在保真度 98% 内取 SWAP 最少，"
                  "depth 规则在深度容差 max(3,5%) 内取 SWAP 最少。"),
        ("为什么：", "静态代价低的布局，路由后不一定好；实测决胜把“学习布局”和“真实产物”对齐，"
                 "避免代理指标自嗨。"),
        ("联合选择：", "≤50 比特还可以先廉价路由预筛 top-6，再跑完整 v5 管线实测，胜者产物缓存复用，"
                  "不重复路由。"),
    ], size=10.2, space_after=7)
    # right: candidate pool flow
    add_rect(s, 7.28, 2.55, 5.45, 0.40, fill=RED, line=None,
             shape=MSO_SHAPE.RECTANGLE)
    add_text(s, 7.28, 2.55, 5.45, 0.40, "候选池 → 实测 → 决胜 → 最终路由",
             size=9.8, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
    rows = [
        ("RL 多起点 top-10", "5 采样 + 4 乱序贪心 + 1 贪心"),
        ("λ=0 拓扑候选", "去掉噪声项，防止链式结构被打散"),
        ("启发式 + 平凡", "Fiedler 序 / 蛇形 / 平凡布局兜底"),
        ("SWIFT 实测", "每个候选跑一遍真实路由"),
        ("三规则决胜", "swap / fidelity / depth"),
    ]
    y = 3.06
    for i, (a, b) in enumerate(rows):
        add_rect(s, 7.28, y, 5.45, 0.48, fill=LIGHT if i < 3 else WHITE,
                 line=GRID, line_width=0.7)
        add_text(s, 7.40, y + 0.04, 2.05, 0.40, a, size=8.6, bold=True,
                 color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, 9.48, y + 0.04, 3.15, 0.40, b, size=7.7, color=GRAY,
                 anchor=MSO_ANCHOR.MIDDLE)
        if i < len(rows) - 1:
            add_line(s, 10.0, y + 0.49, 10.0, y + 0.54, color=MID, width=1.0,
                     arrow=True)
        y += 0.56
    add_rect(s, 0.55, 5.45, 6.55, 0.52, fill=LIGHT, line=GRID,
             line_width=0.7)
    add_text(s, 0.70, 5.47, 6.25, 0.48,
             "为什么候选池有效：同一策略的不同起点会落到不同局部最优；实测决胜把“多样性”变成“可选性”。",
             size=8.8, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    add_rect(s, 7.28, 5.92, 5.45, 0.42, fill="FFF7E6", line=YELLOW,
             line_width=0.8)
    add_text(s, 7.40, 5.94, 5.20, 0.38,
             "胜者：全预算最终路由，永不截断；后处理再用非劣化开关兜底。",
             size=8.4, bold=True, color=GOLD, anchor=MSO_ANCHOR.MIDDLE)
    takeaway(s, "学习只负责“给出好起点”，真实路由负责“选出真赢家”；这一步是代理代价和真机产物之间的桥。",
             y=6.36, h=0.38, size=9.6)
    notes(s, "学习布局输出一组候选，不是一个答案。每个候选都过 SWIFT 实测，"
             "按 swap/fidelity/depth 三规则决胜。这样静态代价低的布局不会自动赢，"
             "必须路由后真的好才算好。")


def slide_swift(s):
    content_slide(
        s, "核心方法", "2.7",
        "SWIFT 路由内核",
        "不只看当前能执行的门，还要看下一步会被解锁的门",
        "SWIFT = Swap-based Incremental Front-layered rouTer；前沿层决定现在，扩展集决定下一步。",
        12,
        title_size=25,
    )
    # left circuit (compact)
    add_rect(s, 0.45, 2.55, 5.55, 3.45, fill=WHITE, line=GRID, line_width=0.8)
    ys = [3.02, 3.56, 4.10, 4.64, 5.18]
    for i, y in enumerate(ys):
        add_line(s, 0.72, y, 5.72, y, color=NAVY, width=1.0)
        add_text(s, 0.50, y - 0.13, 0.20, 0.26, f"q{i}", size=8.6,
                 color=NAVY, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    for a, b, x in [(0, 1, 1.45), (2, 3, 1.45)]:
        add_line(s, x, ys[a], x, ys[b], color=BLUE, width=1.5)
        add_circle_num(s, x - 0.07, ys[a] - 0.07, 0.14, "", fill=BLUE, size=1)
        add_circle_num(s, x - 0.10, ys[b] - 0.10, 0.20, "", fill=WHITE,
                       color=BLUE, size=1)
        ring = s.shapes[-1]
        ring.line.color.rgb = rgb(BLUE)
        ring.line.width = Pt(1.5)
    add_text(s, 1.05, 2.68, 1.5, 0.24, "F 前沿层", size=8.8, bold=True,
             color=BLUE)
    add_line(s, 3.00, ys[1], 3.00, ys[2], color=GOLD, width=1.5)
    add_circle_num(s, 2.93, ys[1] - 0.07, 0.14, "", fill=GOLD, size=1)
    add_circle_num(s, 2.90, ys[2] - 0.10, 0.20, "", fill=WHITE, color=GOLD,
                   size=1)
    ring = s.shapes[-1]
    ring.line.color.rgb = rgb(GOLD)
    ring.line.width = Pt(1.5)
    add_text(s, 2.62, 3.76, 1.6, 0.24, "E 扩展集", size=8.8, bold=True,
             color=GOLD)
    add_line(s, 2.15, ys[1] + 0.05, 2.15, ys[2] - 0.05, color=MID,
             width=1.1, arrow=True)
    add_text(s, 1.95, 3.38, 0.65, 0.20, "SWAP", size=7.2, color=MID)
    add_line(s, 2.15, ys[3] + 0.05, 2.15, ys[4] - 0.05, color=RED,
             width=1.8, arrow=True)
    add_text(s, 1.95, 4.86, 0.65, 0.20, "chosen", size=7.2, color=RED,
             bold=True)
    add_text(s, 0.72, 5.55, 5.05, 0.28,
             "蓝色 = 当前可并行；金色 = 即将解锁；灰/红 = 候选交换。",
             size=7.4, color=GRAY)
    # right text
    add_bullets(s, 6.20, 2.55, 6.55, 3.60, [
        ("前沿层 F：", "依赖已全部满足、端点空闲的 2Q 门；每次交换后重新检查，交换可能让某个前沿门立刻满足邻近性。"),
        ("扩展集 E：", "依赖集 ⊆ F ∪ executed 的 2Q 门，即将进入前沿的“下一层”，用作前瞻信号。"),
        ("评分：", "score = Σ_F w_g d(after_g) + w_ext Σ_E w_g d(after_g) + λ_ms·Δ_ms + δ·h(a,b)；"
                "w_ext=0.5/N，history 抑制来回换位，makespan 项可选。"),
        ("工程：", "候选×前沿距离矩阵向量化，argmin 决胜；候选边排序遍历，同分取最小边序，保证逐位确定性；"
                "最终路由用 16 个种子取 SWAP 最少者，永不截断。"),
        ("数理本质：", "式(7) 是“当前层 + 下一层”的加权距离一步前瞻和；不保证全局最优，但能在依赖顺序约束下"
                   "给出稳定、可解释的交换序列。前瞻启发式单独贡献 15–20% 的交换减少。"),
    ], size=9.9, space_after=4)
    add_rect(s, 6.20, 5.55, 6.55, 0.42, fill=LIGHT, line=GRID, line_width=0.7)
    add_text(s, 6.34, 5.57, 6.28, 0.38,
             "终止性：每步交换要么解锁前沿门，要么降低加权距离和；全预算上限兜底。",
             size=8.4, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    takeaway(s, "路由不是找最短路，而是在依赖顺序里找一条少插交换的路；SWIFT 的“前瞻”就是把下一步的代价提前算进来。",
             y=6.42, h=0.36, size=9.4)
    notes(s, "SWIFT 的核心是前沿层和扩展集。前沿层是现在能执行的门，扩展集是下一批会被解锁的门。"
             "评分同时看两层，再加历史惩罚和可选的 makespan 项。"
             "前瞻启发式单独贡献 15 到 20 个百分点的交换减少。")


def slide_post(s):
    content_slide(
        s, "核心方法", "2.8",
        "置换重写与平台基",
        "SWAP 可以穿过门：把交换推到末端，再整体吸收",
        "后处理不是“再跑一遍优化”，而是用置换群的共轭恒等式，把路由已经付出的交换成本赚回来。",
        13,
        title_size=25,
    )
    add_picture_fit(s, EX / "slide09_2.png", 0.40, 2.52, 5.25, 2.00)
    add_text(s, 0.55, 4.65, 5.10, 1.55,
             "SWAP(a,b); CX(a,c) → CX(b,c); SWAP(a,b)\n"
             "SWAP(a,b); CX(a,b) → CX(b,a); SWAP(a,b)\n"
             "SWAP(a,b); R(a)   → R(b);   SWAP(a,b)",
             size=8.0, color=GRAY, line_spacing=1.35, name="Consolas")
    add_bullets(s, 6.05, 2.52, 6.70, 3.55, [
        ("核心恒等式：", "P_(a,b)·U·P_(a,b) = U^{σ_ab}，即 SWAP 对门的作用是置换群共轭。三条派生规则："
                   "不相交对易、单边触碰重标记、双边触碰控靶翻转。"),
        ("推挤与吸收：", "SWAP 只向后推、不振荡；到末端与逆 SWAP 抵消（SWAP²=I），或进入尾部吸收："
                   "末端 SWAP 对测量无影响，直接删除并更新 final_layout。"),
        ("硬件约束：", "重写后的 2Q 门必须仍在耦合图上，否则 SWAP 阻塞不穿；这条约束保证“语义等价”同时“硬件可执行”。"),
        ("平台基：", "U3(θ,φ,λ) ≅ RZ(λ)·SX·RZ(θ+π)·SX·RZ(φ+3π)；H = RZ(π/2)·SX·RZ(π/2)；"
                 "CX→H-CZ-H；SWAP→3×CX。"),
        ("四层重写：", "A1 共轭推挤、A2 恒等对消、A3 平台基分解、A4 VF2++ 精确嵌入；"
                  "最后非劣化开关：后处理不优于路由产物就回退。"),
    ], size=9.9, space_after=5)
    add_rect(s, 6.05, 5.55, 6.70, 0.42, fill=LIGHT, line=GRID, line_width=0.7)
    add_text(s, 6.18, 5.57, 6.45, 0.38,
             "保语义：所有重写都是 SWAP 共轭恒等式或自逆门对消；态矢量等价测试兜底。",
             size=8.4, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    add_rect(s, 6.05, 6.12, 6.70, 0.40, fill="FFF7E6", line=YELLOW,
             line_width=0.8)
    add_text(s, 6.18, 6.14, 6.45, 0.36,
             "QFT 族 logf 改善 −21.7 / −39.6 / −62.1；graphstate 2Q 门 681 → 120。",
             size=8.0, bold=True, color=GOLD, anchor=MSO_ANCHOR.MIDDLE)
    takeaway(s, "路由少插一个交换，后处理再消掉一个；两级消化，收益不重复计。",
             y=6.58, h=0.28, size=9.4)
    notes(s, "后处理的核心是 SWAP 的共轭恒等式。SWAP 穿过门时只改标签，推到末端后整体吸收。"
             "硬件约束要求重写后的双比特门仍在耦合图上，否则不允许穿。"
             "最后分解到平台基 rz/sx/x/cz，并用非劣化开关兜底。")


def slide_system(s):
    content_slide(
        s, "核心方法", "2.9",
        "系统实现与工程边界",
        "交付管线零 qiskit：评委环境只装四个包就能跑",
        "自研解析器、IR、SABRE、Weyl/Vatan-Williams 合成、VF2++；qiskit 只作为对照实验工具。",
        14,
        title_size=24,
    )
    stages = [
        ("OpenQASM\n2.0", "输入"),
        ("自研解析器\n+ Circuit IR", "前端"),
        ("程序图 / 设备图\n6D / 7D", "建模"),
        ("RL 布局\nGAT + 指针", "决策"),
        ("SWIFT 路由\n前沿 + 前瞻", "路由"),
        ("置换重写\n+ VF2++", "后处理"),
        ("平台基\nrz/sx/x/cz", "分解"),
        ("QASM / QCIS\nmetrics", "输出"),
    ]
    x, w, gap = 0.55, 1.32, 0.20
    for i, (label, tag) in enumerate(stages):
        color = NAVY if i not in (3, 4, 5) else (BLUE if i == 3 else (RED if i == 4 else GREEN))
        add_rect(s, x, 2.52, w, 0.62, fill=WHITE, line=color, line_width=1.0)
        add_text(s, x + 0.04, 2.54, w - 0.08, 0.42, label, size=8.4,
                 bold=True, color=color, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE, line_spacing=0.92)
        add_text(s, x + 0.04, 2.98, w - 0.08, 0.14, tag, size=6.8,
                 color=GRAY, align=PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            add_line(s, x + w + 0.02, 2.83, x + w + gap - 0.02, 2.83,
                     color=MID, width=0.9, arrow=True)
        x += w + gap
    add_bullets(s, 0.55, 3.32, 6.05, 2.55, [
        ("零 qiskit 边界：", "交付内核与 CLI 从入口断言 qiskit 未加载；自研解析器、IR、SABRE、Weyl/VW 2Q 合成、"
                       "VF2++ 全部不依赖 qiskit。训练、对比、Web 才允许用 qiskit。"),
        ("设备抽象：", "读取 live 校准数据，支持天衍-287/176 等机型；禁用比特、真实标签 Q1..Q105、"
                   "QCIS 指令 RZ/X2P/CZ/M 由 adapter 统一处理。"),
        ("验证：", "≤20 比特全空间态矢量等价；21–28 比特随机直积态采样 32 个；"
                 "584 条产物 582 合法、2 条超时、0 条非法；另有 232 项 pytest。"),
    ], size=10.1, space_after=5)
    add_rect(s, 0.55, 5.55, 6.05, 0.62, fill="1E1E1E", line=None,
             shape=MSO_SHAPE.RECTANGLE)
    add_text(s, 0.72, 5.58, 5.75, 0.56,
             "$ python -m qtrail.pure_cli examples/ghz10.qasm\n"
             "1 SWAP · depth 48 · 2Q depth 12 · fid 0.8256",
             size=8.0, color="7EE787", name="Consolas",
             line_spacing=1.15, anchor=MSO_ANCHOR.MIDDLE)
    add_rect(s, 6.85, 3.32, 5.95, 0.52, fill="1E1E1E", line=None,
             shape=MSO_SHAPE.RECTANGLE)
    add_text(s, 7.02, 3.35, 5.65, 0.46,
             'assert "qiskit" not in sys.modules',
             size=11, color="7EE787", name="Consolas",
             anchor=MSO_ANCHOR.MIDDLE)
    add_bullets(s, 6.85, 4.02, 5.95, 2.20, [
        ("四个核心依赖：", "torch、numpy、networkx、pyyaml；numba 可选，没有就自动回退纯 Python，结果不变。"),
        ("产物：", "平台基 QASM、天衍 QCIS、metrics.json；QCIS 可直接提交平台。"),
        ("可复现：", "一键安装脚本 + CLI + Web + 真机验证接口；评委环境不需要 qiskit 也能跑完整映射。"),
    ], size=8.8, space_after=4)
    takeaway(s, "零 qiskit 不是口号：入口有一行断言，评委环境只装四个包就能跑。",
             y=6.38, h=0.38, size=9.8)
    notes(s, "交付管线是纯自研的：解析器、IR、路由、合成、VF2++ 都不依赖 qiskit。"
             "入口有一行断言，qiskit 没有加载才继续。"
             "设备抽象可以读真实校准数据，输出 QCIS 可以直接提交到天衍平台。")


def slide_setup(s):
    content_slide(
        s, "实验与价值", "3.1",
        "实验设置",
        "584 条线路，六种方法，一套统一口径",
        "Benchpress + MQTBench + QUEKO；按规模分层；所有方法过同一平台基分解。",
        15,
        title_size=25,
    )
    data = [
        ("项目", "设置"),
        ("数据集", "Benchpress + MQTBench + QUEKO；584 条完成（≤10:127 / 11–50:258 / 51–105:199）"),
        ("目标设备", "天衍-287 规格：105 比特、15×7 网格、188 条边、合成校准"),
        ("基线", "盲目 O1/O3、噪声感知 O1/O3、pytket；统一平台基 [rz,sx,x,cz]"),
        ("指标", "SWAP、深度、2Q 深度、估计保真度（乘积模型，含读出）"),
        ("规则", "默认 fidelity；候选池实测决胜；最终路由不截断"),
        ("合法性", "584 条产物 582 合法；2 条超时；0 条非法"),
    ]
    add_table(s, 0.55, 2.52, 7.95, 3.55, data, [1.20, 6.75], font_size=9.4)
    add_bullets(s, 8.75, 2.52, 4.05, 3.65, [
        ("数据集覆盖：", "Benchpress 覆盖 bigint/clifford/dtc/feynman/qaoa/qasmbench/qft/qv/square-heisenberg 等 9 类；"
                    "MQTBench 36 类分层生成；QUEKO 提供已知最优嵌入的稠密基准。"),
        ("基线设置：", "盲目 O1/O3 只看拓扑；感知 O1/O3 把校准误差注入 qiskit Target；pytket 作第三方编译器；"
                   "所有方法统一过平台基分解。"),
        ("为什么统一口径：", "不同编译器的 SWAP 计数和门集不同，不先过同一平台基分解，比较没有意义。"),
        ("三条指标：", "SWAP 看交换次数，深度看关键路径，保真度看门误差 + 读出误差的乘积模型。"),
        ("分层与统计：", "≤10 / 11–50 / 51–105 三层；均值 / 截尾（去 10%）/ 中位数三口径；逐线路数据落盘，可复核。"),
    ], size=9.2, space_after=4)
    takeaway(s, "先把尺子统一，再谈输赢；584 条里 582 条合法，剩下 2 条是超时，不是非法产物。",
             y=6.32, h=0.38, size=9.6)
    notes(s, "实验集是 Benchpress、MQTBench、QUEKO 三条线，584 条完成。"
             "目标设备按天衍-287 规格：105 比特、15×7 网格。"
             "五种基线和本方案全部过同一平台基分解，避免门集口径不同带来的假差距。")


def slide_overall(s):
    content_slide(
        s, "实验与价值", "3.2",
        "总体结果",
        "总体估计保真度与噪声感知 O3 持平，超 pytket 约 26%",
        "584 条线路统一口径；逐线路胜场 332 / 515 / 561（对 O3 / pytket / 盲目 O1）。",
        16,
        title_size=24,
    )
    add_picture_fit(s, FIG / "fig_overall.png", 0.35, 2.52, 7.55, 3.90)
    add_bullets(s, 8.20, 2.52, 4.60, 3.55, [
        ("总体均值：", "QTrial 0.226，噪声感知 O3 0.226，基本持平；pytket 0.180，我们高约 26%；"
                  "盲目 O1 0.107、盲目 O3 0.123，均明显低于我们。"),
        ("逐线路胜场：", "对感知 O3 胜 332 条，对 pytket 胜 515 条，对盲目 O1 胜 561 条；"
                   "不是靠某几条极端线路拉高均值。"),
        ("跨拓扑：", "同一套权重、零微调，在 6×6 网格 / heavy-hex / Sycamore 上分别比盲目 O1 好 4.34× / 2.60× / 1.37×，"
                 "说明学的是结构，不是某台机器。"),
        ("均值口径：", "按各层线路数加权，含门误差和实际测量位读出误差；大层乘积下溢时改看 2Q 深度和 log 口径。"),
        ("胜场口径：", "逐线路比较估计保真度，相等或在噪声内不计胜；332 / 515 / 561 是三条基线的胜场。"),
        ("解读：", "我们不是全面第一；在统一口径下追平强基线，并明显超过 pytket，这是更诚实的定位。"),
    ], size=9.1, space_after=4)
    takeaway(s, "不是全面第一，而是在统一口径下追平强基线，并明显超过 pytket。",
             y=6.42, h=0.36, size=9.6)
    notes(s, "总体均值 0.226，和噪声感知 O3 的 0.226 持平；比 pytket 的 0.180 高约 26%。"
             "逐线路胜场 332 条。同一套权重在三种拓扑上零微调，仍明显优于盲目 O1。")


def slide_stratified(s):
    content_slide(
        s, "实验与价值", "3.3",
        "分层结果",
        "小层持平、中层反超、大层仍差 17.6%",
        "584 条线路按规模分层；差距集中在一个结构族，不是全面落后。",
        17,
        title_size=25,
    )
    add_picture_fit(s, EX / "slide11_1.png", 0.35, 2.48, 12.55, 3.30)
    cards = [
        ("≤10 比特", "0.7325 vs 0.7384（−0.8%）", "胜 68/127",
         "小规模交互图简单，精确嵌入/局部搜索就能接近最优，差距在统计噪声内。", NAVY),
        ("11–50 比特", "0.1460 vs 0.1404（+4.0%）", "胜 141/258",
         "中等规模开始体现布局质量：噪声感知 + 候选决胜把 2Q 门放到更安静的边上。", GREEN),
        ("51–105 比特", "0.0075 vs 0.0091（−17.6%）", "胜 123/199",
         "大层差距主要来自 1Q 门体积与块模板效率；只比 2Q 深度时 2788 vs pytket 3004。", RED),
    ]
    x = 0.55
    for title, body, win, analysis, color in cards:
        add_rect(s, x, 5.90, 3.95, 0.94, fill=LIGHT, line=GRID, line_width=0.7)
        add_text(s, x + 0.12, 5.93, 1.10, 0.22, title, size=8.8, bold=True,
                 color=color)
        add_text(s, x + 1.22, 5.93, 2.60, 0.22, body, size=7.8, color=DARK)
        add_text(s, x + 0.12, 6.16, 3.65, 0.44, analysis, size=7.6, color=GRAY,
                 line_spacing=1.05)
        add_text(s, x + 2.90, 5.93, 0.95, 0.22, win, size=7.2, color=GRAY,
                 align=PP_ALIGN.RIGHT)
        x += 4.14
    notes(s, "分层看：小层基本持平，中层反超 4%，大层仍低 17.6%，但胜率 62%。"
             "差距主要来自大层的 1Q 门体积和块模板效率；如果只比 2Q 深度，"
             "我们 2788，比 pytket 的 3004 低，但仍高于感知 O3 的 2263。")


def slide_real(s):
    content_slide(
        s, "实验与价值", "3.4",
        "真机验证",
        "真机闭环：QFT5 经典保真度 0.90，QFT6 0.84",
        "天衍-176，66 比特备选机，2000 shots；与主表 287 模拟口径不同。",
        18,
        title_size=24,
    )
    add_picture_fit(s, EX / "slide13_1.png", 0.35, 2.50, 8.05, 2.85)
    add_bullets(s, 0.55, 5.50, 8.0, 0.80, [
        ("三层验证：", "① 态矢量等价（映射前后量子态一致）→ ② 平台规则校验（门集/邻接/禁用比特）→ "
                  "③ 真机分布对照（天衍-176，2000 shots）。"),
    ], size=8.2, space_after=2)
    add_bullets(s, 8.65, 2.52, 4.15, 3.50, [
        ("QFT5：", "实测经典保真度 0.90，Qiskit O3 为 0.66；QFT6 为 0.84 对 0.60。小规模 QFT 在真机上明显优于 O3。"),
        ("QAOA12：", "0.30 vs 0.33，互有胜负；GHZ5/10/15 为 0.358 / 0.143 / 0.041，随规模衰减。"),
        ("口径：", "真机是 66 比特备选机，不是 287 主表；shots=2000，经典保真度 = (Σ√pq)²，"
                 "与模拟口径分开看。"),
        ("真机设置：", "7 条 QTrial 线路 + 3 条 Qiskit O3 对照；shots=2000；理想分布用 qiskit Statevector；"
                   "经典保真度 = (Σ√pq)²。"),
        ("合法性：", "584 条产物 582 合法，2 条超时；真机小规模结果说明管线可执行、语义保真。"),
        ("结论边界：", "真机只验证小规模可执行性与语义保真，不代表 287 主表；大规模仍受噪声和连通性限制。"),
    ], size=8.9, space_after=3)
    takeaway(s, "小规模 QFT 在真机上明显优于 Qiskit O3；大规模仍受噪声限制，这也是下一步要啃的硬骨头。",
             y=6.38, h=0.38, size=9.4)
    notes(s, "真机用的是天衍-176，66 比特备选机，不是 287 主表。"
             "QFT5 经典保真度 0.90，Qiskit O3 是 0.66；QFT6 是 0.84 对 0.60。"
             "QAOA12 互有胜负，GHZ 随规模衰减。这个口径和主表分开，不能混着说。")


def slide_value(s):
    content_slide(
        s, "实验与价值", "3.5",
        "落地价值",
        "不是把 AI 贴在量子外面，而是让两边互相解决痛点",
        "校准噪声进入奖励，硬件约束变成训练信号；量子编译任务反哺 AI 的组合优化研究。",
        19,
        title_size=25,
    )
    add_rect(s, 0.55, 2.52, 5.65, 3.45, fill=LIGHT, line=GRID, line_width=0.9)
    add_text(s, 0.78, 2.62, 5.1, 0.32, "AI 赋能量子", size=17, bold=True,
             color=NAVY)
    add_bullets(s, 0.78, 3.02, 5.15, 2.75, [
        ("布局：", "GAT + 指针学“程序图 → 好置换”，稀疏终局奖励避免中间 shaping。"),
        ("噪声：", "7 维校准特征 + 有效距离，校准噪声直接进奖励；跨设备零重训。"),
        ("证据：", "0.226 vs 0.226；332 条胜感知 O3；同一权重三种拓扑 1.37–4.34×。"),
        ("交付：", "从 QASM 到 QCIS 的零 qiskit 管线，评委环境四个包就能跑。"),
    ], size=9.2, space_after=4)
    add_rect(s, 7.15, 2.52, 5.65, 3.45, fill="FFF7E6", line=YELLOW,
             line_width=0.9)
    add_text(s, 7.38, 2.62, 5.1, 0.32, "量子反哺 AI", size=17, bold=True,
             color=GOLD)
    add_bullets(s, 7.38, 3.02, 5.15, 2.75, [
        ("问题：", "NP 难的 QAP 给 AI 一个可学习、可验证、可复现的组合优化任务。"),
        ("配方：", "稀疏奖励 + 自批评基线 + 多起点搜索，89 轮收敛，消费级显卡几分钟。"),
        ("证据：", "584 条 / 582 合法；真机构建出硬件可执行的验证闭环。"),
        ("方法：", "学习给起点、搜索做精化、路由实测决胜——一套可迁移的“学习 + 组合搜索”范式。"),
    ], size=9.2, space_after=4)
    add_line(s, 6.28, 3.85, 7.08, 3.85, color=NAVY, width=1.6, arrow=True)
    add_line(s, 7.08, 4.35, 6.28, 4.35, color=GOLD, width=1.6, arrow=True)
    add_text(s, 6.20, 4.50, 1.05, 0.26, "双向", size=8.5, bold=True,
             color=GRAY, align=PP_ALIGN.CENTER)
    add_text(s, 6.20, 4.74, 1.05, 0.26, "赋能", size=8.5, bold=True,
             color=GRAY, align=PP_ALIGN.CENTER)
    cards = [
        ("应用潜力", "量子化学 / 组合优化 / ML 编译"),
        ("平台交付", "设备抽象 + QCIS 导出"),
        ("可复现价值", "一键脚本 + 232 项测试"),
    ]
    x = 0.55
    for title, body in cards:
        add_rect(s, x, 6.12, 3.95, 0.62, fill=WHITE, line=GRID, line_width=0.7)
        add_text(s, x + 0.12, 6.14, 1.05, 0.58, title, size=8.6, bold=True,
                 color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, x + 1.22, 6.14, 2.65, 0.58, body, size=8.4, color=GRAY,
                 anchor=MSO_ANCHOR.MIDDLE)
        x += 4.14
    notes(s, "双向赋能不是口号。AI 这边，校准噪声直接进奖励，布局策略从数据里学结构；"
             "量子这边，NP 难的 QAP 给 AI 提供了一个可学习、可验证的组合优化任务。"
             "两边各自解决对方的痛点。")


def slide_summary(s):
    content_slide(
        s, "实验与价值", "3.6",
        "总结与展望",
        "三条贡献，一个可复现的闭环",
        "核心创新是稀疏奖励 RL 布局；路由和后处理把布局收益兑现；零 qiskit 让整条链路可交付。",
        20,
        title_size=25,
    )
    cards = [
        ("01", "稀疏奖励 RL 布局", "把 QAP 写成 MDP；GAT + 指针；稀疏终局奖励 + 自批评基线 + 多起点搜索。",
         "0.226 vs 0.226", NAVY,
         "· 只给终局奖励，不做中间 shaping\n· 自批评基线压方差\n· 跨拓扑零重训 1.37–4.34×"),
        ("02", "SWIFT + 置换重写", "路由少插交换；后处理用共轭恒等式消掉；平台基 [rz,sx,x,cz]。",
         "584 条 / 582 合法", RED,
         "· 前沿层 + 扩展集 + 前瞻\n· SWAP 共轭 / 恒等对消\n· VF2++ 精确嵌入"),
        ("03", "零 qiskit 工程闭环", "自研解析 / IR / 路由 / 合成；CLI / Web / QCIS；评委环境四个包。",
         "232 项测试", GREEN,
         "· 自研解析器 / IR / 路由\n· Weyl / VW 3-CX 合成\n· CLI / Web / QCIS"),
    ]
    x = 0.55
    for num, title, body, evidence, color, bullets in cards:
        add_rect(s, x, 2.55, 3.85, 3.15, fill=WHITE, line=GRID, line_width=0.9)
        add_rect(s, x, 2.55, 3.85, 0.07, fill=color, line=None,
                 shape=MSO_SHAPE.RECTANGLE)
        add_text(s, x + 0.20, 2.70, 0.62, 0.42, num, size=18, bold=True,
                 color=color)
        add_text(s, x + 0.82, 2.74, 2.75, 0.34, title, size=13, bold=True,
                 color=color)
        add_text(s, x + 0.20, 3.22, 3.40, 0.72, body, size=9.3, color=DARK,
                 line_spacing=1.12)
        add_text(s, x + 0.20, 4.02, 3.40, 0.72, bullets, size=8.4, color=GRAY,
                 line_spacing=1.10)
        add_rect(s, x + 0.20, 4.92, 3.40, 0.42, fill=LIGHT, line=GRID,
                 line_width=0.6)
        add_text(s, x + 0.20, 4.94, 3.40, 0.38, evidence, size=9.5,
                 bold=True, color=color, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)
        x += 4.14
    add_rect(s, 0.55, 5.82, 12.23, 0.80, fill=LIGHT, line=GRID, line_width=0.8)
    add_text(s, 0.72, 5.84, 11.9, 0.76,
             "下一步：大层 1Q 门体积与块模板效率；更多真机机型；在线校准反馈；"
             "把“学习给起点 + 搜索做精化 + 路由实测决胜”推广到更多编译阶段。\n"
             "局限：大层 2Q 深度仍高于感知 O3；真机只覆盖 66 比特备选机。",
             size=9.2, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE,
             line_spacing=1.15)
    notes(s, "三条贡献：稀疏奖励的布局、SWIFT 加置换重写、零 qiskit 的工程闭环。"
             "核心创新是第一条。下一步重点是大层的 1Q 门体积和块模板效率，以及更多真机机型。")


def slide_thanks(s):
    for sh in s.shapes:
        if sh.has_text_frame and "批评指正" in sh.text_frame.text:
            tf = sh.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = "感谢各位评委老师批评指正！"
            set_font(r, 52, True, "FFFF00")
    add_text(s, 1.34, 5.42, 11.35, 0.40,
             "QTrial · 让量子程序在真实芯片上跑得更深、更稳",
             size=19, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, 1.34, 5.92, 11.35, 0.34,
             "何欣航  夏聆心  袁一茗 · 指导教师：尚涛",
             size=13, color=WHITE, align=PP_ALIGN.CENTER)
    notes(s, "感谢各位评委老师，欢迎批评指正。")


# ---------------------------------------------------------------- build
def build():
    base = prepare_base()
    prs = Presentation(base)
    assert len(prs.slides) == TOTAL, len(prs.slides)

    slide_cover(prs.slides[0])
    slide_toc(prs.slides[1])
    slide_problem(prs.slides[2])
    slide_qap(prs.slides[3])
    slide_opportunity(prs.slides[4])
    slide_pipeline(prs.slides[5])
    slide_core_innovation(prs.slides[6])
    slide_mdp(prs.slides[7])
    slide_network(prs.slides[8])
    slide_training(prs.slides[9])
    slide_multistart(prs.slides[10])
    slide_swift(prs.slides[11])
    slide_post(prs.slides[12])
    slide_system(prs.slides[13])
    slide_setup(prs.slides[14])
    slide_overall(prs.slides[15])
    slide_stratified(prs.slides[16])
    slide_real(prs.slides[17])
    slide_value(prs.slides[18])
    slide_summary(prs.slides[19])
    slide_thanks(prs.slides[20])

    cp = prs.core_properties
    cp.title = "QTrial：自适应量子线路映射优化系统"
    cp.author = "何欣航 夏聆心 袁一茗"
    cp.subject = "2026 年揭榜挂帅擂台赛 · 量子 + AI 双向赋能"
    cp.comments = "基于案例模板重制的 QTrial 竞赛展示 PPT v2（内容加厚版）"

    prs.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    build()
