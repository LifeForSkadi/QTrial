"""作品报告书 docx 格式模块（参照北航本科毕业论文撰写规范）。

页面：A4；页边距 上2.8/下2.2/左3.0/右2.0 cm；页眉1.8/页脚1.4 cm。
正文：中文宋体小四、西文数字 Times New Roman、1.5 倍行距、首行缩进
2 字符、段前后 0、两端对齐。
标题：一级黑体三号居中段前后 1 行；二级黑体四号左顶格段前后 0.5 行；
三级黑体小四左顶格段前后 0.5 行。
"""
from __future__ import annotations

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SONG = "宋体"
HEI = "黑体"
TIMES = "Times New Roman"


def set_font(run, chinese=SONG, western=TIMES, size=12, bold=False):
    run.font.name = western
    run.font.size = Pt(size)
    run.font.bold = bold
    r = run._element.rPr
    rFonts = r.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        r.append(rFonts)
    rFonts.set(qn("w:ascii"), western)
    rFonts.set(qn("w:hAnsi"), western)
    rFonts.set(qn("w:eastAsia"), chinese)


def para(doc, text="", *, style=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         indent_chars=2, space_before=0, space_after=0, line15=True,
         bold=False, size=12, chinese=SONG):
    p = doc.add_paragraph(style=style)
    pf = p.paragraph_format
    pf.alignment = align
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    if line15:
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if indent_chars:
        pf.first_line_indent = Pt(size * indent_chars)
    if text:
        run = p.add_run(text)
        set_font(run, chinese=chinese, size=size, bold=bold)
    return p


def heading1(doc, text, page_break=False):
    """一级标题：黑体三号居中，段前后 1 行。"""
    if page_break:
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(16)
    pf.space_after = Pt(16)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    run = p.add_run(text)
    set_font(run, chinese=HEI, size=16, bold=True)
    p.style = doc.styles["Heading 1"]
    # 标题样式字体同步（目录域提取用）
    st = doc.styles["Heading 1"]
    st.font.name = TIMES
    st.font.size = Pt(16)
    st.font.bold = True
    st._element.rPr.rFonts.set(qn("w:eastAsia"), HEI)
    return p


def heading2(doc, text):
    """二级标题：黑体四号左顶格，段前后 0.5 行。"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    run = p.add_run(text)
    set_font(run, chinese=HEI, size=14, bold=True)
    p.style = doc.styles["Heading 2"]
    st = doc.styles["Heading 2"]
    st.font.name = TIMES
    st.font.size = Pt(14)
    st.font.bold = True
    st._element.rPr.rFonts.set(qn("w:eastAsia"), HEI)
    return p


def heading3(doc, text):
    """三级标题：黑体小四左顶格，段前后 0.5 行。"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    run = p.add_run(text)
    set_font(run, chinese=HEI, size=12, bold=True)
    p.style = doc.styles["Heading 3"]
    st = doc.styles["Heading 3"]
    st.font.name = TIMES
    st.font.size = Pt(12)
    st.font.bold = True
    st._element.rPr.rFonts.set(qn("w:eastAsia"), HEI)
    return p


def figure(doc, img_path, caption, width_cm=14.0):
    """图：图片居中，图题置于下方居中（图序与图题间空 1 字符）。"""
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run()
    run.add_picture(img_path, width=Cm(width_cm))
    cap = doc.add_paragraph()
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(3)
    cap.paragraph_format.space_after = Pt(10)
    cap.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    seq, _, title = caption.partition("  ")
    run = cap.add_run(seq + " ")
    set_font(run, size=10.5, bold=True)
    run2 = cap.add_run(title)
    set_font(run2, size=10.5)
    return cap


def table_title(doc, caption):
    """表题置于上方居中。"""
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(caption)
    set_font(run, size=10.5, bold=True)
    return p


def ref_para(doc, text):
    """参考文献条目：悬挂缩进、五号。"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.left_indent = Pt(21)
    pf.first_line_indent = Pt(-21)
    run = p.add_run(text)
    set_font(run, size=10.5)
    return p


def para_with_math(doc, text, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                   indent_chars=2, size=12):
    """正文段落：行内数学符号按 mathdef.INLINE_PATTERNS 转为
    m:oMath 对象（Word/MathType 可双击编辑），其余为普通文本。"""
    from mathdef import convert_inline, append_inline_math
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = align
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if indent_chars:
        pf.first_line_indent = Pt(size * indent_chars)
    for kind, payload in convert_inline(text):
        if kind == "t":
            run = p.add_run(payload)
            set_font(run, size=size)
        else:
            append_inline_math(p._p, payload)
    return p


def equation_para(doc, key):
    """显示公式段落（居中）：编号标签 + m:oMath 公式对象。"""
    from mathdef import DISPLAY, build_omath
    label, items = DISPLAY[key]
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    run = p.add_run(label)
    set_font(run, size=12, bold=True)
    p._p.append(build_omath(items))
    return p


def setup_page(doc):
    for sec in doc.sections:
        sec.page_width = Cm(21.0)
        sec.page_height = Cm(29.7)
        sec.top_margin = Cm(2.8)
        sec.bottom_margin = Cm(2.2)
        sec.left_margin = Cm(3.0)
        sec.right_margin = Cm(2.0)
        sec.header_distance = Cm(1.8)
        sec.footer_distance = Cm(1.4)
    # 页脚居中页码
    footer = doc.sections[0].footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1)
    run._r.append(instr)
    run._r.append(fld2)
    set_font(run, size=10.5)


def insert_toc(doc):
    """目录：插入 TOC 域（打开后右键更新域/F9 生成页码）。"""
    p = doc.add_paragraph()
    run = p.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t")
    t.text = "（目录：在 Word 中右键此处选择“更新域”自动生成三级目录）"
    fld3 = OxmlElement("w:fldChar")
    fld3.set(qn("w:fldCharType"), "end")
    r = run._r
    r.append(fld1)
    r.append(instr)
    r.append(fld2)
    r.append(t)
    r.append(fld3)


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
