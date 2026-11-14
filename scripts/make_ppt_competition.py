# -*- coding: utf-8 -*-
"""Build the QTrial competition deck on top of the supplied 20-page template.

Design contract:
- Keep the template's top banner, section title group, subheader bar,
  cover furniture and closing background.
- Replace the content of 15 native content slides with QTrial material.
- Delete the two extra TOC pages, yielding 18 slides.

Run:
    python scripts/make_ppt_competition.py
Output:
    delivery/QTrail_竞赛展示_模板版.pptx
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
OUT = ROOT / "delivery" / "QTrail_竞赛展示_模板版.pptx"
FIG = ROOT / "assets" / "ppt_figs"
EX = ROOT / "assets" / "existing_ppt"

# template palette
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

TOTAL = 18


# ---------------------------------------------------------------- helpers
def rgb(hexstr: str) -> RGBColor:
    return RGBColor.from_string(hexstr.replace("#", "").upper())


def set_font(run, size=14, bold=False, color=NAVY, name="微软雅黑", italic=False):
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


def add_text(slide, x, y, w, h, text, size=14, bold=False, color=NAVY,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             line_spacing=1.0, space_after=0, name="微软雅黑",
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
    lines = str(text).split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        r = p.add_run()
        r.text = line
        set_font(r, size, bold, color, name, italic)
    return box


def add_runs(slide, x, y, w, h, paragraphs, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, line_spacing=1.0, space_after=0,
             margins=(0.02, 0.02, 0.02, 0.02)):
    """paragraphs: list of list of dicts {t,size,bold,color,name}"""
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
            set_font(r, rd.get("size", 13), rd.get("bold", False),
                     rd.get("color", DARK), rd.get("name", "微软雅黑"),
                     rd.get("italic", False))
    return box


def add_rect(slide, x, y, w, h, fill=None, line=None, line_width=1.0,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
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
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        conn.line.dash_style = dash
    if arrow:
        ln = conn.line._get_or_add_ln()
        tail = OxmlElement("a:tailEnd")
        tail.set("type", "triangle")
        tail.set("w", "med")
        tail.set("len", "med")
        ln.append(tail)
    return conn


def add_circle_num(slide, x, y, d, num, fill=NAVY, color=WHITE, size=14):
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


def add_table(slide, x, y, w, h, data, col_widths, font_size=11,
              header_fill=NAVY, first_col_bold=True):
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
            cell.margin_left = Inches(0.07)
            cell.margin_right = Inches(0.07)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(header_fill if r == 0 else
                                           (LIGHT if r % 2 == 0 else WHITE))
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 1 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(val)
            set_font(run, font_size,
                     bold=(r == 0 or (c == 0 and first_col_bold)),
                     color=(WHITE if r == 0 else DARK))
    return table


def add_footer(slide, page):
    add_line(slide, 0.55, 6.98, 12.78, 6.98, color="D9D9D9", width=0.75)
    add_text(slide, 0.55, 7.04, 8.0, 0.24,
             "QTrial · 量子线路映射优化方案", size=8.5, color="808080")
    add_text(slide, 11.55, 7.04, 1.25, 0.24, f"{page:02d} / {TOTAL}",
             size=8.5, color="808080", align=PP_ALIGN.RIGHT)


def clear_content(slide):
    """Keep only the top section-title group and the subheader bar."""
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
    # PowerPoint's mc:AlternateContent blocks are invisible to python-pptx
    # but still rendered. They hold the old template body on several slides.
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
        if sh.has_text_frame and 1.1 <= sh.top / 914400 <= 1.7 \
                and sh.width / 914400 > 12:
            runs = [r for p in sh.text_frame.paragraphs for r in p.runs]
            if len(runs) >= 2:
                runs[0].text = f" {number} "
                runs[1].text = title
                return True
    # template slides 3/4 have no subheader; create one
    add_rect(slide, 0.14, 1.21, 13.06, 0.44, fill=NAVY, line=None,
             shape=MSO_SHAPE.RECTANGLE)
    add_rect(slide, 0.14, 1.21, 0.10, 0.44, fill=YELLOW, line=None,
             shape=MSO_SHAPE.RECTANGLE)
    add_text(slide, 0.24, 1.21, 12.8, 0.44, f" {number}  {title}",
             size=19, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    return True


def content_slide(slide, section, number, subheader, title, subtitle,
                  page, title_size=29):
    clear_content(slide)
    set_header(slide, section, number)
    set_subheader(slide, number, subheader)
    add_text(slide, 0.55, 1.72, 12.2, 0.62, title, size=title_size,
             bold=True, color=NAVY)
    if subtitle:
        add_text(slide, 0.55, 2.30, 12.2, 0.42, subtitle, size=15,
                 color=GRAY)
    add_footer(slide, page)


def card(slide, x, y, w, h, title, body, accent=NAVY, fill=LIGHT,
         title_size=13, body_size=10.5, tag=None):
    add_rect(slide, x, y, w, h, fill=fill, line=GRID, line_width=0.75)
    add_rect(slide, x, y, 0.055, h, fill=accent, line=None,
             shape=MSO_SHAPE.RECTANGLE)
    add_text(slide, x + 0.16, y + 0.09, w - 0.25, 0.30, title,
             size=title_size, bold=True, color=accent)
    add_text(slide, x + 0.16, y + 0.41, w - 0.25, h - 0.48, body,
             size=body_size, color=DARK, line_spacing=1.08)
    if tag:
        add_text(slide, x + 0.16, y + h - 0.30, w - 0.25, 0.24, tag,
                 size=8.5, color=MID, italic=True)


def takeaway(slide, text, y=6.38):
    add_rect(slide, 0.55, y, 12.23, 0.46, fill="FFF7E6", line=YELLOW,
             line_width=0.9)
    add_text(slide, 0.72, y + 0.05, 11.9, 0.36, text, size=12.5,
             bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)


def stat(slide, x, y, w, h, value, label, value_color=NAVY,
         value_size=24, label_size=9):
    add_text(slide, x, y, w, h * 0.62, value, size=value_size, bold=True,
             color=value_color, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.BOTTOM)
    add_text(slide, x, y + h * 0.62, w, h * 0.38, label, size=label_size,
             color=GRAY, align=PP_ALIGN.CENTER)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------- slides
def slide_cover(s):
    # title box: replace all text but keep the box
    title_box = None
    author_box = None
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
        set_font(r, 15, False, BLUE)
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = "QTrial"
        set_font(r2, 54, True, RED)
        p3 = tf.add_paragraph()
        p3.alignment = PP_ALIGN.CENTER
        r3 = p3.add_run()
        r3.text = "自适应量子线路映射优化系统"
        set_font(r3, 30, True, NAVY)
    if author_box is not None:
        tf = author_box.text_frame
        tf.clear()
        lines = [
            "团队成员：何欣航  夏聆心  袁一茗",
            "指导教师：尚涛",
            "2026 年 11 月",
        ]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = line
            set_font(r, 20, False, NAVY)

    # replace the six bottom photos with project evidence cards
    photo_names = {"图片 1", "图片 12", "图片 13", "图片 18", "图片 20", "图片 21"}
    positions = [
        (-0.01, 5.87, 2.25, 1.63),
        (2.27, 5.87, 2.20, 1.63),
        (4.50, 5.85, 2.46, 1.65),
        (6.99, 5.85, 1.35, 1.65),
        (8.37, 5.86, 2.44, 1.64),
        (10.84, 5.86, 2.47, 1.64),
    ]
    for sh in list(s.shapes):
        if sh.name in photo_names:
            sh._element.getparent().remove(sh._element)
    for i, (x, y, w, h) in enumerate(positions, 1):
        add_picture_fit(s, FIG / f"cover_{i}.png", x, y, w, h)
    notes(s, "各位评委老师好，我们是 QTrial 团队。我们的作品是一套面向 NISQ 真机的"
             "自适应量子线路映射系统：用强化学习学布局，用 SWIFT 做路由，用置换重写做后处理，"
             "整条交付管线零 qiskit。")


def slide_toc(s):
    # change the three TOC labels; keep numbers and active-state colors
    labels = ["研究背景", "方案设计", "实验与价值"]
    # find text boxes with the original labels
    originals = {"个人介绍": labels[0], "研究内容": labels[1], "研究进展": labels[2]}
    for sh in s.shapes:
        if sh.shape_type == 6:
            for ch in sh.shapes:
                if ch.has_text_frame:
                    for p in ch.text_frame.paragraphs:
                        for r in p.runs:
                            if r.text in originals:
                                r.text = originals[r.text]
    add_text(s, 1.9, 6.05, 9.55, 0.45,
             "从一条 QASM 到一台真机：布局、路由、后处理，全链路可复现。",
             size=15, color=GRAY, align=PP_ALIGN.CENTER)
    notes(s, "汇报分三部分：为什么映射难、我们怎么做、做得怎么样。")


def slide_problem(s):
    content_slide(
        s, "研究背景", "1.1",
        "量子线路映射：NISQ 时代的刚需",
        "一条逻辑线路，不能直接跑在真实芯片上",
        "拓扑只允许相邻比特做双比特门；远距离交互必须靠 SWAP 搬家，交换越多，深度和误差越大。",
        3,
    )
    add_picture_fit(s, FIG / "fig_problem.png", 0.40, 2.78, 8.05, 4.05)
    card(s, 8.62, 2.80, 4.15, 1.18, "拓扑约束",
         "CX 只能落在相邻物理比特；一次 SWAP = 3 个 CX。", NAVY)
    card(s, 8.62, 4.10, 4.15, 1.18, "噪声约束",
         "同一芯片耦合器 ε2q 可差 40×；T1 只有 25–80 μs。", RED)
    card(s, 8.62, 5.40, 4.15, 1.18, "映射的目标",
         "合法、保语义；SWAP 更少、深度更浅、保真度更高。", BLUE)
    notes(s, "先看左边这条三比特线路：q0 和 q2 之间有一个 CX。放到右边 3×3 网格上，"
             "两个比特隔了四跳。芯片只允许相邻比特做双比特门，所以必须插入 SWAP，"
             "把其中一个比特搬到另一个旁边。一个 SWAP 是三个 CX，跳数越多，代价越大。")


def slide_qap(s):
    content_slide(
        s, "研究背景", "1.2",
        "问题建模：从线路到 QAP",
        "把布局写成加权图匹配：交互越多，物理距离越要小",
        "逻辑交互图 G_C 与设备耦合图 G_D 之间找一个双射 π，最小化静态代价 C(π)。",
        4,
    )
    add_picture_fit(s, FIG / "fig_qap.png", 0.35, 2.72, 8.10, 4.05)
    card(s, 8.62, 2.78, 4.15, 1.10, "QAP 是 NP-难的",
         "105 比特无法精确求解；精确算法只能处理小规模。", RED)
    # three routes
    rows = [
        ("精确解", "ILP / SMT", "小规模可证最优；105 比特不可解。", MID),
        ("启发式", "SABRE / O3 / tket", "毫秒级；规则静态；换芯片要重调。", BLUE),
        ("学习式", "本方案", "从数据学结构；跨设备零重训。", NAVY),
    ]
    y = 4.00
    for name, tool, desc, color in rows:
        add_rect(s, 8.62, y, 4.15, 0.72, fill=WHITE, line=GRID, line_width=0.75)
        add_rect(s, 8.62, y, 0.055, 0.72, fill=color, line=None,
                 shape=MSO_SHAPE.RECTANGLE)
        add_text(s, 8.76, y + 0.06, 0.85, 0.26, name, size=11, bold=True,
                 color=color)
        add_text(s, 9.55, y + 0.06, 3.1, 0.26, tool, size=10, bold=True,
                 color=DARK)
        add_text(s, 8.76, y + 0.35, 3.9, 0.30, desc, size=9, color=GRAY)
        y += 0.78
    add_text(s, 8.62, 6.38, 4.15, 0.40,
             "静态代价只是路由成本的代理；胜负要在路由后看产物。",
             size=10, color=RED, bold=True)
    notes(s, "把线路写成加权交互图，把芯片写成耦合图，布局就是一个双射。"
             "目标函数是交互次数乘物理距离。这个模型是 QAP，NP 难。"
             "精确解只能做小规模，启发式快但规则静态，所以我们选择学习式布局。")


def slide_opportunity(s):
    content_slide(
        s, "研究背景", "1.3",
        "研究机会与本文定位",
        "现有编译器的规则是静态的，芯片的噪声是动态的",
        "同一套启发式参数，换一台芯片、换一组校准数据，效果可能完全不同。",
        5,
    )
    cols = [
        ("精确解", "ILP / SMT", "可证最优", "105 比特不可解；不适合在线编译。", MID),
        ("启发式", "SABRE / O3 / tket", "毫秒级", "规则静态；换芯片要重调参数。", BLUE),
        ("学习式", "GAT + 指针 + 稀疏奖励", "自适应", "从校准数据学结构；跨设备零重训。", NAVY),
    ]
    x = 0.55
    for name, tool, tag, desc, color in cols:
        add_rect(s, x, 2.78, 3.85, 3.15, fill=WHITE, line=GRID, line_width=0.9)
        add_rect(s, x, 2.78, 3.85, 0.08, fill=color, line=None,
                 shape=MSO_SHAPE.RECTANGLE)
        add_text(s, x + 0.22, 2.98, 3.4, 0.35, name, size=17, bold=True,
                 color=color)
        add_text(s, x + 0.22, 3.38, 3.4, 0.28, tool, size=10.5, bold=True,
                 color=DARK)
        add_text(s, x + 0.22, 3.80, 3.4, 0.25, tag, size=12, bold=True,
                 color=color)
        add_text(s, x + 0.22, 4.15, 3.35, 1.35, desc, size=11.5, color=GRAY,
                 line_spacing=1.15)
        x += 4.10
    conclusions = [("结论：不可扩展", MID), ("结论：快但静态", BLUE), ("结论：可学习、可迁移", NAVY)]
    x = 0.55
    for text, color in conclusions:
        add_text(s, x + 0.22, 5.35, 3.45, 0.32, text, size=12, bold=True,
                 color=color)
        x += 4.10
    add_rect(s, 0.55, 6.08, 12.23, 0.72, fill=LIGHT, line=GRID, line_width=0.8)
    add_text(s, 0.75, 6.14, 11.85, 0.60,
             "我们的定位：不替代 SABRE，而是学习一个更好的初始布局；"
             "路由用 SWIFT，后处理用代数重写保证非劣化。",
             size=13, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "三类方法各有取舍。我们不打算做一个端到端黑箱，"
             "而是把学习放在最需要数据、也最能迁移的地方——布局。"
             "路由仍然是可解释的 SWIFT，后处理用代数恒等式保证不劣化。")


def slide_pipeline(s):
    content_slide(
        s, "方案设计", "2.1",
        "总体方案",
        "三处自研，一条从 QASM 到 QCIS 的零 qiskit 管线",
        "布局学、路由快、后处理赚；其余环节复用公开算法，边界写清楚。",
        6,
    )
    # four columns
    cols = [
        ("输入与建模", NAVY, [
            ("OpenQASM 2.0", "解析门序列 / 依赖 DAG"),
            ("程序交互图", "6 维节点特征"),
            ("设备耦合图", "7 维校准特征"),
        ]),
        ("强化学习布局", BLUE, [
            ("GAT 双图编码", "4 层 × 8 头 × 128 维"),
            ("指针逐比特解码", "16 头注意力 + 掩码"),
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
            ("QASM / QCIS", "metrics.json + 校验"),
        ]),
    ]
    x0 = 0.55
    col_w = 2.78
    gap = 0.37
    for ci, (header, color, boxes) in enumerate(cols):
        x = x0 + ci * (col_w + gap)
        add_rect(s, x, 2.76, col_w, 0.36, fill=color, line=None,
                 shape=MSO_SHAPE.RECTANGLE)
        add_text(s, x, 2.76, col_w, 0.36, header, size=11.5, bold=True,
                 color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        for bi, (label, sub) in enumerate(boxes):
            y = 3.26 + bi * 0.82
            add_rect(s, x, y, col_w, 0.66, fill=WHITE, line=GRID, line_width=0.8)
            add_text(s, x + 0.08, y + 0.05, col_w - 0.16, 0.28, label,
                     size=10.5, bold=True, color=color, align=PP_ALIGN.CENTER)
            add_text(s, x + 0.08, y + 0.34, col_w - 0.16, 0.26, sub,
                     size=8.2, color=GRAY, align=PP_ALIGN.CENTER)
        # arrows between columns
        if ci < len(cols) - 1:
            add_line(s, x + col_w + 0.04, 4.45, x + col_w + gap - 0.04, 4.45,
                     color=MID, width=1.4, arrow=True)
    # training loop
    add_line(s, 11.45, 5.45, 11.45, 5.80, color=YELLOW, width=1.4,
             dash=None)
    add_line(s, 11.45, 5.80, 4.80, 5.80, color=YELLOW, width=1.4)
    add_line(s, 4.80, 5.80, 4.80, 5.45, color=YELLOW, width=1.4, arrow=True)
    add_text(s, 5.05, 5.86, 6.20, 0.28,
             "训练链：REINFORCE 自批评更新同一套权重（89 轮收敛）",
             size=9, color="B8860B", italic=True)
    add_text(s, 0.55, 6.28, 12.25, 0.38,
             "解析、图构建复用公开算法；三处自研：布局 / 路由 / 后处理。"
             "训练与推理共享同一套布局策略，用候选决胜对齐口径。",
             size=10.5, color=GRAY)
    notes(s, "整条管线四段：输入建模、RL 布局、SWIFT 路由、后处理交付。"
             "三处自研分别是布局、路由、后处理；解析和图构建复用公开算法。"
             "训练链和推理链共享同一套布局策略。")


def slide_mdp(s):
    content_slide(
        s, "方案设计", "2.2",
        "强化学习布局：把 QAP 写成 MDP",
        "状态是部分布局，动作是选一个位置，奖励只在终点给一次",
        "把排列生成写成序列决策；稀疏终局奖励避免中间塑形带来的偏差。",
        7,
    )
    # left text
    add_text(s, 0.55, 2.80, 0.22, 0.32, "S", size=16, bold=True, color=BLUE)
    add_text(s, 0.86, 2.80, 2.97, 0.28, "状态 S", size=13, bold=True, color=BLUE)
    add_text(s, 0.86, 3.08, 2.97, 0.72,
             "已放置集合 + 程序图嵌入 + 设备图嵌入；每步重新编码，节点带 7 维校准特征。",
             size=9.8, color=DARK, line_spacing=1.08)
    add_text(s, 0.55, 3.92, 0.22, 0.32, "A", size=16, bold=True, color=RED)
    add_text(s, 0.86, 3.92, 2.97, 0.28, "动作 A", size=13, bold=True, color=RED)
    add_text(s, 0.86, 4.20, 2.97, 0.62,
             "为下一个逻辑比特选一个未占用物理位置；掩码保证布局是双射。",
             size=9.8, color=DARK, line_spacing=1.08)
    add_text(s, 0.55, 4.94, 0.22, 0.32, "R", size=16, bold=True, color=YELLOW)
    add_text(s, 0.86, 4.94, 2.97, 0.28, "奖励 R", size=13, bold=True, color="B8860B")
    add_text(s, 0.86, 5.22, 2.97, 0.72,
             "t<T 恒为 0；t=T 给 −C(π)/Σw 一次；另有深度项与紧凑性项。",
             size=9.8, color=DARK, line_spacing=1.08)
    add_rect(s, 0.55, 6.05, 3.25, 0.72, fill=LIGHT, line=GRID, line_width=0.8)
    add_text(s, 0.68, 6.10, 3.0, 0.62,
             "刻意不做中间 shaping：中间布局没有可信的好坏标签，加了会把策略带偏。",
             size=9.2, color=NAVY, line_spacing=1.05)
    # image
    add_picture_fit(s, EX / "slide07_1.png", 4.05, 2.72, 8.70, 4.10)
    notes(s, "布局被写成一个序列决策：每步给下一个逻辑比特选位置，选完为止。"
             "奖励只在完整布局时给一次，用 QAP 代价除以总交互权重。"
             "中间不做 shaping，因为中间布局没有可信的好坏标签。")


def slide_network(s):
    content_slide(
        s, "方案设计", "2.3",
        "策略网络与训练",
        "GAT 编码两张图，指针解码逐比特放置",
        "同一套权重处理任意节点数；校准特征按设备自身统计量归一化，换芯片不换权重。",
        8,
    )
    add_picture_fit(s, EX / "slide08_1.png", 0.40, 2.70, 8.15, 3.68)
    cards = [
        ("为什么 GAT", "稀疏图；邻居重要性可学习；4 层 = 4 跳感受野。", NAVY),
        ("为什么指针", "输出是排列，维度随输入变化；掩码保证双射。", BLUE),
        ("为什么自批评", "基线复用策略本身，不额外引入价值网络。", RED),
    ]
    y = 2.76
    for title, body, color in cards:
        card(s, 8.78, y, 4.0, 1.10, title, body, color, title_size=12,
             body_size=9.6)
        y += 1.20
    # stats strip
    stats = [
        ("4096", "实例 / 轮"),
        ("512", "batch"),
        ("89", "轮收敛"),
        ("3.2 s", "单轮耗时"),
        ("~60 万", "模型参数"),
    ]
    x = 0.55
    for value, label in stats:
        add_rect(s, x, 6.45, 2.30, 0.42, fill=LIGHT, line=None,
                 shape=MSO_SHAPE.RECTANGLE)
        add_text(s, x + 0.10, 6.45, 1.05, 0.42, value, size=15, bold=True,
                 color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, x + 1.12, 6.45, 1.12, 0.42, label, size=8.5, color=GRAY,
                 anchor=MSO_ANCHOR.MIDDLE)
        x += 2.42
    notes(s, "编码器是 GAT，程序图和设备图各一个；解码器是指针网络，每步指向一个物理比特。"
             "训练用 REINFORCE 加自批评基线。模型只有 60 万参数，89 轮收敛，"
             "单轮 3.2 秒，消费级显卡几分钟就能训完。")


def slide_swift(s):
    content_slide(
        s, "方案设计", "2.4",
        "SWIFT 路由内核",
        "不只看当前能执行的门，还要看下一步会被解锁的门",
        "SWIFT = Swap-based Incremental Front-layered rouTer；用前沿层、扩展集和前瞻评分决定下一个 SWAP。",
        9,
    )
    # circuit diagram
    add_rect(s, 0.45, 2.72, 6.15, 3.55, fill=WHITE, line=GRID, line_width=0.8)
    ys = [3.18, 3.72, 4.26, 4.80, 5.34]
    for i, y in enumerate(ys):
        add_line(s, 0.75, y, 6.30, y, color=NAVY, width=1.1)
        add_text(s, 0.52, y - 0.14, 0.22, 0.28, f"q{i}", size=9.5, color=NAVY,
                 align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    # front layer: CX(q0,q1), CX(q2,q3) at x=1.55
    for a, b, x in [(0, 1, 1.55), (2, 3, 1.55)]:
        add_line(s, x, ys[a], x, ys[b], color=BLUE, width=1.6)
        add_circle_num(s, x - 0.07, ys[a] - 0.07, 0.14, "", fill=BLUE, size=1)
        add_circle_num(s, x - 0.10, ys[b] - 0.10, 0.20, "", fill=WHITE,
                       color=BLUE, size=1)
        # ring target
        ring = s.shapes[-1]
        ring.line.color.rgb = rgb(BLUE)
        ring.line.width = Pt(1.6)
    add_text(s, 1.15, 2.86, 1.6, 0.26, "F 前沿层", size=9.5, bold=True,
             color=BLUE)
    # extended set: CX(q1,q2) at x=3.15, dashed orange
    add_line(s, 3.15, ys[1], 3.15, ys[2], color="BF9000", width=1.6, dash=None)
    add_circle_num(s, 3.08, ys[1] - 0.07, 0.14, "", fill="BF9000", size=1)
    add_circle_num(s, 3.05, ys[2] - 0.10, 0.20, "", fill=WHITE, color="BF9000",
                   size=1)
    ring = s.shapes[-1]
    ring.line.color.rgb = rgb("BF9000")
    ring.line.width = Pt(1.6)
    add_text(s, 2.75, 3.92, 1.7, 0.26, "E 扩展集", size=9.5, bold=True,
             color="B8860B")
    # candidate SWAPs at x=2.25
    add_line(s, 2.25, ys[1] + 0.05, 2.25, ys[2] - 0.05, color=MID,
             width=1.2, arrow=True)
    add_line(s, 2.32, ys[2] - 0.05, 2.32, ys[1] + 0.05, color=MID,
             width=1.2, arrow=True)
    add_text(s, 2.02, 3.55, 0.75, 0.22, "SWAP", size=8, color=MID)
    # chosen swap in red between q3/q4
    add_line(s, 2.25, ys[3] + 0.05, 2.25, ys[4] - 0.05, color=RED,
             width=2.0, arrow=True)
    add_text(s, 2.02, 5.0, 0.75, 0.22, "chosen", size=8, color=RED, bold=True)
    # annotation
    add_text(s, 0.75, 5.75, 5.65, 0.35,
             "蓝色 = 当前可并行执行；黄色 = 即将解锁；灰色/红色 = 候选 SWAP。",
             size=8.8, color=GRAY)

    # right side formula
    add_rect(s, 6.85, 2.76, 5.95, 0.95, fill="FFF7E6", line=YELLOW,
             line_width=1.0)
    add_text(s, 7.00, 2.82, 5.65, 0.30,
             "评分：H = Σ_F w·d + (0.5/N) Σ_E w·d + δ·history + λ_ms·makespan",
             size=9.2, bold=True, color=NAVY)
    add_text(s, 7.00, 3.12, 5.65, 0.48,
             "前沿层看当前代价，扩展集看下一步；历史项防来回交换，makespan 项可选。",
             size=8.8, color=GRAY)
    bullets = [
        ("前沿层 F", "依赖已满足、端点空闲的 2Q 门，可并行执行。"),
        ("扩展集 E", "紧跟其后的门，前瞻避免“只顾眼前”的短视。"),
        ("历史惩罚 δ", "记录每条边被交换的次数，压住来回换位。"),
        ("16 个种子", "取 SWAP 最少者；最终路由永不截断。"),
    ]
    y = 3.88
    for title, body in bullets:
        add_text(s, 6.90, y, 1.35, 0.26, title, size=10.2, bold=True,
                 color=BLUE)
        add_text(s, 8.25, y, 4.45, 0.42, body, size=9.4, color=DARK)
        y += 0.52
    add_text(s, 6.90, 6.10, 5.85, 0.30,
             "前瞻启发式单独贡献 15–20% 的交换减少。",
             size=9, color=RED, bold=True)
    notes(s, "SWIFT 的核心是三层：前沿层、扩展集、前瞻评分。"
             "前沿层是现在就能执行的门，扩展集是下一批会被解锁的门。"
             "评分同时看两层，再加历史惩罚和可选的 makespan 项。"
             "最后用 16 个种子取最优，最终路由不做截断。")


def slide_post(s):
    content_slide(
        s, "方案设计", "2.5",
        "置换重写与平台基",
        "SWAP 可以穿过门：把交换推到末端，再整体吸收",
        "路由少插一个交换，后处理再消掉一个；两级消化，收益不重复计。",
        10,
    )
    add_picture_fit(s, EX / "slide09_2.png", 0.40, 2.72, 5.75, 2.18)
    # identities
    add_text(s, 0.55, 5.05, 5.55, 1.35,
             "SWAP(a,b); CX(a,c) → CX(b,c); SWAP(a,b)\n"
             "SWAP(a,b); CX(a,b) → CX(b,a); SWAP(a,b)\n"
             "SWAP(a,b); R(a)   → R(b);   SWAP(a,b)",
             size=8.8, color=GRAY, line_spacing=1.35,
             name="Consolas")
    # four chips
    chips = [
        ("A1", "共轭推挤", "SWAP 越过对易门，与后续 SWAP 对消。", BLUE),
        ("A2", "恒等对消", "CX·CX = I；尾部 SWAP 吸收进测量基。", NAVY),
        ("A3", "平台基分解", "按 [rz, sx, x, cz] 重写，压等效 2Q 门。", RED),
        ("A4", "VF2++ 精确嵌入", "可嵌入线路直接 0-SWAP，不插交换。", GREEN),
    ]
    x = 6.35
    y = 2.76
    for i, (tag, title, body, color) in enumerate(chips):
        xx = x + (i % 2) * 3.25
        yy = y + (i // 2) * 1.55
        add_rect(s, xx, yy, 3.10, 1.38, fill=LIGHT, line=GRID, line_width=0.8)
        add_circle_num(s, xx + 0.12, yy + 0.12, 0.34, tag, fill=color, size=8)
        add_text(s, xx + 0.54, yy + 0.10, 2.45, 0.30, title, size=11.5,
                 bold=True, color=color)
        add_text(s, xx + 0.14, yy + 0.50, 2.82, 0.78, body, size=9.2,
                 color=DARK, line_spacing=1.08)
    add_text(s, 6.35, 5.98, 6.35, 0.30,
             "非劣化开关：后处理产物按决胜规则不劣于路由产物才采用。",
             size=9.2, bold=True, color=RED)
    add_text(s, 6.35, 6.28, 6.35, 0.30,
             "QFT 族 logf 改善 −21.7 / −39.6 / −62.1；graphstate 2Q 门 681 → 120。",
             size=8.8, color=GRAY)
    notes(s, "后处理分四层：共轭推挤、恒等对消、平台基分解、VF2++ 精确嵌入。"
             "核心恒等式是 SWAP 穿过门时只改标签；推到末端后整体吸收。"
             "最后有非劣化开关，后处理不优于路由产物就回退。")


def slide_system(s):
    content_slide(
        s, "方案设计", "2.6",
        "系统实现与工程边界",
        "交付管线零 qiskit：评委环境只装四个包就能跑",
        "自研解析器、IR、SABRE、Weyl/Vatan-Williams 合成、VF2++；qiskit 只作为对照实验工具。",
        11,
    )
    # top architecture chain
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
    x = 0.55
    w = 1.32
    gap = 0.20
    for i, (label, tag) in enumerate(stages):
        color = NAVY if i not in (3, 4, 5) else (BLUE if i == 3 else (RED if i == 4 else GREEN))
        add_rect(s, x, 2.78, w, 0.72, fill=WHITE, line=color, line_width=1.1)
        add_text(s, x + 0.04, 2.82, w - 0.08, 0.46, label, size=9.2,
                 bold=True, color=color, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE, line_spacing=0.95)
        add_text(s, x + 0.04, 3.30, w - 0.08, 0.18, tag, size=7.2,
                 color=GRAY, align=PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            add_line(s, x + w + 0.02, 3.14, x + w + gap - 0.02, 3.14,
                     color=MID, width=1.0, arrow=True)
        x += w + gap
    # support chains
    add_rect(s, 0.55, 3.78, 5.55, 0.55, fill=LIGHT, line=GRID, line_width=0.8)
    add_text(s, 0.70, 3.83, 5.30, 0.45,
             "训练支撑：数据池 → REINFORCE → checkpoint（89 轮）",
             size=9.2, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    add_rect(s, 6.35, 3.78, 6.43, 0.55, fill=LIGHT, line=GRID, line_width=0.8)
    add_text(s, 6.50, 3.83, 6.15, 0.45,
             "验证链：态矢量等价 → 平台规则校验 → 真机分布对照",
             size=9.2, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    # code box
    add_rect(s, 0.55, 4.60, 6.55, 0.62, fill="1E1E1E", line=None,
             shape=MSO_SHAPE.RECTANGLE)
    add_text(s, 0.75, 4.66, 6.2, 0.50,
             'assert "qiskit" not in sys.modules',
             size=12, color="7EE787", name="Consolas",
             anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, 7.30, 4.62, 5.45, 0.60,
             "入口处一行断言：交付 CLI 不加载 qiskit；\n"
             "训练、对比、Web 才允许使用。",
             size=9.4, color=GRAY, line_spacing=1.1)
    # three cards
    cards = [
        ("零 qiskit 边界", "映射内核与 CLI 零 qiskit；\n训练/对比/Web 可用 qiskit。", NAVY),
        ("设备抽象", "读取校准数据；支持天衍机型；\n禁用比特 / 真实标签。", BLUE),
        ("可复现", "一键脚本；metrics.json；\nQCIS 可直接提交。", RED),
    ]
    x = 0.55
    for title, body, color in cards:
        card(s, x, 5.45, 3.95, 1.05, title, body, color, title_size=11.5,
             body_size=8.8)
        x += 4.14
    notes(s, "交付管线是纯自研的：解析器、IR、路由、合成、VF2++ 都不依赖 qiskit。"
             "入口有一行断言，qiskit 没有加载才继续。"
             "设备抽象可以读真实校准数据，输出 QCIS 可以直接提交到天衍平台。")


def slide_setup(s):
    content_slide(
        s, "实验与价值", "3.1",
        "实验设置",
        "584 条线路，六种方法，一套统一口径",
        "Benchpress + MQTBench + QUEKO；按规模分层；所有方法过同一平台基分解。",
        12,
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
    add_table(s, 0.55, 2.76, 8.55, 3.45, data, [1.35, 7.2], font_size=10.2)
    stats = [
        ("584", "条线路完成"),
        ("582", "条合法，0 条非法"),
        ("6", "种方法同口径"),
    ]
    y = 2.80
    for value, label in stats:
        add_text(s, 9.55, y, 3.3, 0.55, value, size=28, bold=True,
                 color=NAVY, align=PP_ALIGN.LEFT)
        add_text(s, 9.58, y + 0.58, 3.25, 0.28, label, size=10, color=GRAY)
        y += 1.02
    add_text(s, 9.55, 5.95, 3.25, 0.75,
             "为什么强调统一口径：不同编译器的 SWAP 计数和门集不同，"
             "不过同一后端分解，比较没有意义。",
             size=8.8, color=RED, line_spacing=1.1)
    takeaway(s, "先把尺子统一，再谈输赢。")
    notes(s, "实验集是 Benchpress、MQTBench、QUEKO 三条线，584 条完成。"
             "目标设备按天衍-287 规格：105 比特、15×7 网格。"
             "五种基线和本方案全部过同一平台基分解，避免门集口径不同带来的假差距。")


def slide_overall(s):
    content_slide(
        s, "实验与价值", "3.2",
        "总体结果",
        "总体估计保真度与噪声感知 O3 持平，超 pytket 约 26%",
        "584 条线路统一口径；逐线路胜场 332 / 515 / 561（对 O3 / pytket / 盲目 O1）。",
        13,
    )
    add_picture_fit(s, FIG / "fig_overall.png", 0.35, 2.68, 7.75, 4.10)
    stats = [
        ("0.226 vs 0.226", "与噪声感知 O3 持平", NAVY),
        ("+26%", "总体均值高于 pytket", GREEN),
        ("332 条", "逐线路优于感知 O3", RED),
    ]
    y = 2.78
    for value, label, color in stats:
        add_text(s, 8.35, y, 4.45, 0.50, value, size=24, bold=True,
                 color=color)
        add_text(s, 8.38, y + 0.50, 4.35, 0.28, label, size=9.5, color=GRAY)
        y += 0.95
    add_rect(s, 8.35, 5.66, 4.40, 0.62, fill=LIGHT, line=GRID, line_width=0.8)
    add_text(s, 8.48, 5.69, 4.15, 0.56,
             "同一套权重，三种拓扑（vs 盲目 O1）：\n"
             "6×6 4.34× · heavy-hex 2.60× · Sycamore 1.37×",
             size=8.3, color=NAVY, line_spacing=1.1)
    takeaway(s, "不是全面第一，而是在统一口径下追平强基线，并明显超过 pytket。")
    notes(s, "总体均值 0.226，和噪声感知 O3 的 0.226 持平；比 pytket 的 0.180 高约 26%。"
             "逐线路胜场 332 条。同一套权重在三种拓扑上零微调，仍明显优于盲目 O1。")


def slide_stratified(s):
    content_slide(
        s, "实验与价值", "3.3",
        "分层结果",
        "小层持平、中层反超、大层仍差 17.6%",
        "584 条线路按规模分层；差距集中在一个结构族，不是全面落后。",
        14,
    )
    add_picture_fit(s, EX / "slide11_1.png", 0.35, 2.62, 12.55, 3.72)
    cards = [
        ("≤10 比特", "0.7325 vs 0.7384（−0.8%）\n胜 68/127", NAVY),
        ("11–50 比特", "0.1460 vs 0.1404（+4.0%）\n胜 141/258", GREEN),
        ("51–105 比特", "0.0075 vs 0.0091（−17.6%）  胜 123/199", RED),
    ]
    x = 0.55
    for title, body, color in cards:
        add_rect(s, x, 6.38, 3.95, 0.48, fill=LIGHT, line=GRID, line_width=0.7)
        add_text(s, x + 0.12, 6.40, 1.15, 0.44, title, size=9.5, bold=True,
                 color=color, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, x + 1.25, 6.40, 2.60, 0.44, body.replace("\n", "  "),
                 size=8.5, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
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
        15,
    )
    add_picture_fit(s, EX / "slide13_1.png", 0.35, 2.68, 8.35, 2.95)
    layers = [
        ("① 态矢量等价", "映射前后量子态一致"),
        ("② 平台规则校验", "门集 / 邻接 / 禁用比特"),
        ("③ 真机分布对照", "天衍-176，2000 shots"),
    ]
    x = 0.55
    for title, body in layers:
        add_rect(s, x, 5.66, 2.65, 0.62, fill=LIGHT, line=GRID, line_width=0.7)
        add_text(s, x + 0.10, 5.68, 2.45, 0.24, title, size=9.0, bold=True,
                 color=NAVY)
        add_text(s, x + 0.10, 5.94, 2.45, 0.30, body, size=7.8, color=GRAY)
        x += 2.80
    add_text(s, 8.85, 2.78, 3.9, 0.55, "0.90", size=32, bold=True, color=NAVY)
    add_text(s, 8.88, 3.32, 3.85, 0.26, "QFT5 实测经典保真度", size=9.5,
             color=GRAY)
    add_text(s, 8.88, 3.58, 3.85, 0.24, "Qiskit O3：0.66", size=8.5,
             color=RED)
    add_text(s, 8.85, 3.98, 3.9, 0.55, "0.84", size=32, bold=True, color=NAVY)
    add_text(s, 8.88, 4.52, 3.85, 0.26, "QFT6 实测经典保真度", size=9.5,
             color=GRAY)
    add_text(s, 8.88, 4.78, 3.85, 0.24, "Qiskit O3：0.60", size=8.5,
             color=RED)
    add_text(s, 8.88, 5.20, 3.85, 0.55,
             "QAOA12 0.30 vs 0.33；\nGHZ5/10/15 0.358 / 0.143 / 0.041。",
             size=8.6, color=DARK, line_spacing=1.15)
    takeaway(s, "小规模 QFT 在真机上明显优于 Qiskit O3；大规模仍受噪声限制。")
    notes(s, "真机用的是天衍-176，66 比特备选机，不是 287 主表。"
             "QFT5 经典保真度 0.90，Qiskit O3 是 0.66；QFT6 是 0.84 对 0.60。"
             "QAOA12 互有胜负，GHZ 随规模衰减。这个口径和主表分开，不能混着说。")


def slide_value(s):
    content_slide(
        s, "实验与价值", "3.5",
        "落地价值",
        "不是把 AI 贴在量子外面，而是让两边互相解决痛点",
        "校准噪声进入奖励，硬件约束变成训练信号；量子编译任务反哺 AI 的组合优化研究。",
        16,
    )
    add_rect(s, 0.55, 2.78, 5.55, 3.35, fill=LIGHT, line=GRID, line_width=0.9)
    add_text(s, 0.80, 2.90, 5.0, 0.38, "AI 赋能量子", size=18, bold=True,
             color=NAVY)
    add_text(s, 0.80, 3.38, 5.0, 2.55,
             "· GAT + 指针学布局，稀疏终局奖励\n"
             "· 校准噪声进奖励，跨设备零重训\n"
             "· 证据：0.226 vs 0.226；332 条胜 O3\n"
             "· 同一权重三种拓扑：1.37–4.34×",
             size=12, color=DARK, line_spacing=1.45)
    add_rect(s, 7.20, 2.78, 5.55, 3.35, fill="FFF7E6", line=YELLOW,
             line_width=0.9)
    add_text(s, 7.45, 2.90, 5.0, 0.38, "量子反哺 AI", size=18, bold=True,
             color="B8860B")
    add_text(s, 7.45, 3.38, 5.0, 2.55,
             "· NP 难 QAP 作为可学习策略 benchmark\n"
             "· 稀疏奖励下可复现的训练配方\n"
             "· 证据：89 轮收敛；584 条 / 582 合法\n"
             "· 真机构建出硬件可执行的验证闭环",
             size=12, color=DARK, line_spacing=1.45)
    add_text(s, 0.80, 5.55, 5.0, 0.40,
             "落到交付：QASM → QCIS 的零 qiskit 管线",
             size=9.5, color=NAVY, italic=True)
    add_text(s, 7.45, 5.55, 5.0, 0.40,
             "落到方法：把硬件约束写进学习目标",
             size=9.5, color="B8860B", italic=True)
    # center arrows
    add_line(s, 6.22, 4.05, 7.10, 4.05, color=NAVY, width=1.8, arrow=True)
    add_line(s, 7.10, 4.55, 6.22, 4.55, color="B8860B", width=1.8, arrow=True)
    add_text(s, 6.18, 4.70, 1.05, 0.30, "双向", size=9, bold=True,
             color=GRAY, align=PP_ALIGN.CENTER)
    add_text(s, 6.18, 4.95, 1.05, 0.30, "赋能", size=9, bold=True,
             color=GRAY, align=PP_ALIGN.CENTER)
    # bottom three cards
    cards = [
        ("应用潜力", "量子化学 / 组合优化 / ML 编译"),
        ("平台交付", "设备抽象 + QCIS 导出"),
        ("可复现价值", "一键脚本 + 232 项测试"),
    ]
    x = 0.55
    for title, body in cards:
        add_rect(s, x, 6.30, 3.95, 0.48, fill=WHITE, line=GRID, line_width=0.7)
        add_text(s, x + 0.12, 6.32, 1.15, 0.44, title, size=8.8, bold=True,
                 color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, x + 1.25, 6.32, 2.60, 0.44, body, size=8.3, color=GRAY,
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
        "从一条 QASM 到一台真机：布局、路由、后处理，全链路可复现。",
        17,
    )
    cards = [
        ("01", "布局", "把 QAP 写成 MDP；GAT + 指针；稀疏终局奖励",
         "0.226 vs 0.226", NAVY,
         "· 稀疏终局奖励 + 自批评\n· 多起点 + 局部搜索\n· 同一权重跨拓扑"),
        ("02", "路由 + 后处理", "SWIFT 少插；置换重写消掉；平台基 [rz,sx,x,cz]",
         "584 条 / 582 合法", RED,
         "· 前沿层 + 扩展集 + 前瞻\n· SWAP 共轭 / 恒等对消\n· VF2++ 精确嵌入"),
        ("03", "工程闭环", "自研解析 / IR / 路由 / 合成；CLI / Web / QCIS",
         "4 个核心依赖", GREEN,
         "· 自研解析器 / IR / 路由\n· Weyl / VW 3-CX 合成\n· CLI / Web / QCIS"),
    ]
    x = 0.55
    for num, title, body, evidence, color, bullets in cards:
        add_rect(s, x, 2.78, 3.85, 3.20, fill=WHITE, line=GRID, line_width=0.9)
        add_rect(s, x, 2.78, 3.85, 0.08, fill=color, line=None,
                 shape=MSO_SHAPE.RECTANGLE)
        add_text(s, x + 0.22, 2.98, 0.70, 0.50, num, size=20, bold=True,
                 color=color)
        add_text(s, x + 0.92, 3.02, 2.65, 0.38, title, size=15, bold=True,
                 color=color)
        add_text(s, x + 0.22, 3.62, 3.35, 0.90, body, size=10.5, color=DARK,
                 line_spacing=1.15)
        add_text(s, x + 0.22, 4.52, 3.35, 0.52, bullets, size=9.0, color=GRAY,
                 line_spacing=1.12)
        add_rect(s, x + 0.22, 5.12, 3.35, 0.55, fill=LIGHT, line=GRID,
                 line_width=0.6)
        add_text(s, x + 0.22, 5.14, 3.35, 0.50, evidence, size=11,
                 bold=True, color=color, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)
        x += 4.14
    add_rect(s, 0.55, 6.18, 12.23, 0.62, fill=LIGHT, line=GRID, line_width=0.8)
    add_text(s, 0.72, 6.22, 11.9, 0.54,
             "下一步：大层 1Q 门体积与块模板效率；更多真机机型；在线校准反馈。",
             size=11.5, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "三条贡献：稀疏奖励的布局、SWIFT 加置换重写、零 qiskit 的工程闭环。"
             "下一步重点是大层的 1Q 门体积和块模板效率，以及更多真机机型。")


def slide_thanks(s):
    # keep the background photo; replace main text and add subtitle
    for sh in s.shapes:
        if sh.has_text_frame and "批评指正" in sh.text_frame.text:
            tf = sh.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = "感谢各位评委老师批评指正！"
            set_font(r, 54, True, "FFFF00")
    add_text(s, 1.34, 5.45, 11.35, 0.45,
             "QTrial · 让量子程序在真实芯片上跑得更深、更稳",
             size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, 1.34, 5.98, 11.35, 0.36,
             "何欣航  夏聆心  袁一茗 · 指导教师：尚涛",
             size=14, color=WHITE, align=PP_ALIGN.CENTER)
    notes(s, "感谢各位评委老师，欢迎批评指正。")


# ---------------------------------------------------------------- build
def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, OUT)
    prs = Presentation(OUT)

    # remove the two extra TOC pages (original slides 16 and 5, 1-based)
    def delete_slide(index):
        xml_slides = prs.slides._sldIdLst
        slides = list(xml_slides)
        rId = slides[index].get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        prs.part.drop_rel(rId)
        xml_slides.remove(slides[index])

    delete_slide(15)
    delete_slide(4)
    assert len(prs.slides) == TOTAL

    slide_cover(prs.slides[0])
    slide_toc(prs.slides[1])
    slide_problem(prs.slides[2])
    slide_qap(prs.slides[3])
    slide_opportunity(prs.slides[4])
    slide_pipeline(prs.slides[5])
    slide_mdp(prs.slides[6])
    slide_network(prs.slides[7])
    slide_swift(prs.slides[8])
    slide_post(prs.slides[9])
    slide_system(prs.slides[10])
    slide_setup(prs.slides[11])
    slide_overall(prs.slides[12])
    slide_stratified(prs.slides[13])
    slide_real(prs.slides[14])
    slide_value(prs.slides[15])
    slide_summary(prs.slides[16])
    slide_thanks(prs.slides[17])

    cp = prs.core_properties
    cp.title = "QTrial：自适应量子线路映射优化系统"
    cp.author = "何欣航 夏聆心 袁一茗"
    cp.subject = "2026 年揭榜挂帅擂台赛 · 量子 + AI 双向赋能"
    cp.comments = "基于案例模板重制的 QTrial 竞赛展示 PPT"

    prs.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    build()
