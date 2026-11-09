# -*- coding: utf-8 -*-
"""QTrial 展示 PPT 生成器 v2（2026-11-09，用户自定义字体/配色/布局版）。

设计约定：
  字体/配色/版式全部集中在顶部 CONFIG——用户改一处即可整体换风格。
  产出 .pptx（PowerPoint/WPS 可直接打开编辑）。
用法：python scripts/thesis/make_ppt_v2.py
输出：作品展示PPT_v2.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Cm, Pt

# ============================================================ 风格配置
CONFIG = {
    "fonts": {
        "cn": "微软雅黑",           # 中文字体
        "en": "Segoe UI",          # 西文字体
    },
    "colors": {
        "primary": "0B3D91",       # 主色：量子蓝
        "accent": "00A3E0",        # 强调：科技青
        "highlight": "FF8C42",     # 高亮：数据橙
        "bg_light": "EEF4FB",      # 浅蓝底
        "bg_gray": "F5F7FA",       # 浅灰底
        "text": "222A35",          # 正文
        "text_light": "5A6572",    # 次要文字
        "white": "FFFFFF",
    },
    "layout": {
        "slide_w": 13.33,          # 16:9（英寸）
        "slide_h": 7.5,
        "title_h": 1.1,            # 标题条高度
        "margin": 0.9,             # 页面边距
        "footer_h": 0.55,
    },
}


def _rgb(hex_s):
    return RGBColor.from_string(hex_s)


def _font(run, size=14, bold=False, color=None, cn=True):
    f = run.font
    f.name = CONFIG["fonts"]["cn"] if cn else CONFIG["fonts"]["en"]
    f.size = Pt(size)
    f.bold = bold
    if color:
        f.color.rgb = _rgb(color)
    # 中文字体需设置 east asian 字体
    from pptx.oxml.ns import qn
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn('a:ea'))
    if ea is None:
        ea = rPr.makeelement(qn('a:ea'), {})
        rPr.append(ea)
    ea.set('typeface', CONFIG["fonts"]["cn"])


def _add_text(slide, x, y, w, h, lines, align=PP_ALIGN.LEFT,
              anchor=MSO_ANCHOR.TOP):
    """lines: [(text, size, bold, color, align_override), ...]"""
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, line in enumerate(lines):
        text, size, bold, color = line[0], line[1], line[2], line[3]
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = line[4] if len(line) > 4 else align
        run = p.add_run()
        run.text = text
        _font(run, size=size, bold=bold, color=color)
    return tb


def _add_rect(slide, x, y, w, h, fill, line=None, radius=True):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Cm(x), Cm(y), Cm(w), Cm(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = _rgb(fill)
    if line:
        shp.line.color.rgb = _rgb(line)
        shp.line.width = Pt(1)
    else:
        shp.line.fill.background()
    return shp


def _add_arrow(slide, x1, y1, x2, y2, color="00A3E0"):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                      Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    conn.line.color.rgb = _rgb(color)
    conn.line.width = Pt(1.5)
    return conn


def _page_decor(slide, prs, title=None, subtitle=None, page_no=None):
    """内容页统一版式：顶部标题条 + 页脚。"""
    L = CONFIG["layout"]
    # 顶部色条
    _add_rect(slide, 0, 0, L["slide_w"], 0.18, CONFIG["colors"]["primary"],
              radius=False)
    # 标题
    if title:
        _add_rect(slide, 0, 0.35, 0.55, L["title_h"] - 0.35,
                  CONFIG["colors"]["accent"], radius=False)
        _add_text(slide, L["margin"], 0.42, 10.5, 0.9,
                  [(title, 24, True, CONFIG["colors"]["primary"])])
    if subtitle:
        _add_text(slide, L["margin"], 1.15, 11.5, 0.5,
                  [(subtitle, 12, False, CONFIG["colors"]["text_light"])])
    # 页脚
    _add_text(slide, L["margin"], L["slide_h"] - L["footer_h"] - 0.1,
              9, 0.5, [("QTrail · 天衍-287 量子线路映射优化", 9, False,
                        CONFIG["colors"]["text_light"])])
    if page_no:
        _add_text(slide, L["slide_w"] - 2.2, L["slide_h"] - L["footer_h"] - 0.1,
                  1.5, 0.5, [(str(page_no), 9, False,
                              CONFIG["colors"]["text_light"])],
                  align=PP_ALIGN.RIGHT)


def cover(prs):
    L = CONFIG["layout"]
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白版式
    # 背景
    _add_rect(slide, 0, 0, L["slide_w"], L["slide_h"],
              CONFIG["colors"]["primary"], radius=False)
    _add_rect(slide, 0, L["slide_h"] - 1.2, L["slide_w"], 1.2,
              CONFIG["colors"]["accent"], radius=False)
    # 装饰圆
    for cx, cy, d in ((11.2, 1.2, 2.2), (11.8, 4.5, 1.1)):
        shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Cm(cx), Cm(cy), Cm(d), Cm(d))
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb("1E5BB8")
        shp.line.fill.background()
    _add_text(slide, 1.6, 2.0, 10.5, 1.4,
              [("QTrail", 54, True, CONFIG["colors"]["white"])])
    _add_text(slide, 1.6, 3.35, 10.5, 1.8,
              [("基于稀疏奖励强化学习的", 26, False, CONFIG["colors"]["white"]),
               ("自适应量子线路映射优化方案", 26, False,
                CONFIG["colors"]["white"])])
    _add_text(slide, 1.6, 5.2, 10.5, 0.6,
              [("面向天衍-287 超导量子处理器 · 零依赖自主实现管线",
                14, False, CONFIG["colors"]["white"])])
    _add_text(slide, 1.6, 6.55, 10.5, 0.6,
              [("2026 年「揭榜挂帅」擂台赛 · 量子+AI 双向赋能榜题",
                12, False, CONFIG["colors"]["white"])])


def toc(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _page_decor(slide, prs, title="目录", page_no=2)
    items = [
        ("01", "研究背景与问题定义", "量子线路映射 · 天衍-287 平台约束"),
        ("02", "技术方案", "稀疏奖励强化学习布局 + 自主路由与后处理栈"),
        ("03", "v5 改进管线", "噪声感知重标记 / 2Q 块重合成 / 1Q 游程优化"),
        ("04", "实验设计与结果", "587 条线路 · 六方法对比 · 真机口径验证"),
        ("05", "总结与展望", "诚实边界 · 后续方向"),
    ]
    y = 2.0
    for num, title, sub in items:
        _add_rect(slide, 1.2, y, 1.1, 0.8, CONFIG["colors"]["bg_light"],
                  radius=False)
        _add_text(slide, 1.2, y + 0.12, 1.1, 0.6,
                  [(num, 20, True, CONFIG["colors"]["accent"])],
                  align=PP_ALIGN.CENTER)
        _add_text(slide, 2.6, y + 0.05, 6.5, 0.5,
                  [(title, 18, True, CONFIG["colors"]["text"])])
        _add_text(slide, 2.6, y + 0.5, 8.5, 0.4,
                  [(sub, 11, False, CONFIG["colors"]["text_light"])])
        y += 1.0


def arch_page(prs):
    """技术架构页：8 阶段流程图（形状 + 连接线）。"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _page_decor(slide, prs, title="技术架构：八阶段自主管线", subtitle="零 qiskit 依赖 · 全部组件自主实现",
                page_no=3)
    stages = [
        ("解析器", "OpenQASM 2.0\nqelib1 / ccx 展开"),
        ("电路 IR", "指令列表\n惰性依赖 DAG"),
        ("程序图", "交互计数加权\n6 维节点特征"),
        ("RL 布局", "GAT+指针网络\n多起点+局部搜索"),
        ("路由竞技", "候选池×自研 SABRE\n实测评分决胜"),
        ("后处理 v5", "C1+L3 重标记\nA2 块合成\nB 1Q 优化"),
        ("平台基分解", "cx→H-CZ-H\nU3→ZSXXZ"),
        ("QCIS 输出", "X2P/RZ/CZ/M\n真机可直接提交"),
    ]
    x, y, w, h, gap = 0.55, 3.1, 1.42, 1.9, 0.17
    for i, (name, desc) in enumerate(stages):
        cx = x + i * (w + gap)
        fill = (CONFIG["colors"]["primary"] if i in (3, 5)
                else CONFIG["colors"]["bg_light"])
        _add_rect(slide, cx, y, w, h, fill)
        _add_text(slide, cx + 0.1, y + 0.15, w - 0.2, 0.55,
                  [(name, 13, True, CONFIG["colors"]["white"] if i in (3, 5)
                    else CONFIG["colors"]["text"])], align=PP_ALIGN.CENTER)
        _add_text(slide, cx + 0.1, y + 0.75, w - 0.2, 1.0,
                  [(line, 8.5, False, CONFIG["colors"]["text_light"])
                   for line in desc.split("\n")], align=PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            _add_arrow(slide, cx + w + 0.02, y + h / 2,
                       cx + w + gap - 0.02, y + h / 2)
    _add_text(slide, 0.55, 5.6, 12, 0.6,
              [("输入 QASM → 映射后 QASM + QCIS + 指标 JSON · 全程 qiskit-free",
                11, False, CONFIG["colors"]["text_light"])],
              align=PP_ALIGN.CENTER)


def results_page(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _page_decor(slide, prs, title="v5 改进效果（消融口径）", subtitle="固定种子 · 同布局对比 · 非劣化开关兜底",
                page_no=4)
    # 左侧：改进组件
    _add_text(slide, 0.9, 2.0, 5.5, 0.5, [("改进组件", 15, True,
                                          CONFIG["colors"]["text"])])
    comps = [
        ("B · 1Q 游程优化", "游程矩阵合并 + H² 奇偶消解（v6 移植+收益检查）"),
        ("C1 · 噪声感知重标记", "贪心合法重定位（真机校准下收益显著）"),
        ("L3 · 学习式重标记", "复用 QAPolicy 解码候选（零重训练）"),
        ("A2 · 2Q 块重合成", "KAK + 拟合合成（同对块收益上限 1.3%）"),
    ]
    y = 2.5
    for name, desc in comps:
        _add_rect(slide, 0.9, y, 5.5, 1.0, CONFIG["colors"]["bg_light"])
        _add_text(slide, 1.15, y + 0.1, 5.0, 0.4,
                  [(name, 12.5, True, CONFIG["colors"]["primary"])])
        _add_text(slide, 1.15, y + 0.48, 5.0, 0.45,
                  [(desc, 9.5, False, CONFIG["colors"]["text_light"])])
        y += 1.15
    # 右侧：消融结果
    _add_text(slide, 7.0, 2.0, 5.5, 0.5, [("消融结论（合成口径）", 15, True,
                                          CONFIG["colors"]["text"])])
    stats = [
        ("small", "+2.4%", "B 贡献全部"),
        ("medium", "+12.1%", "B 贡献全部"),
        ("large", "+13069%", "B 贡献全部"),
        ("真机口径", "+1.4~219%", "C1+L3 起效"),
    ]
    y = 2.5
    for layer, gain, note in stats:
        _add_rect(slide, 7.0, y, 5.5, 0.95, CONFIG["colors"]["bg_gray"])
        _add_text(slide, 7.3, y + 0.22, 1.8, 0.5,
                  [(layer, 12, True, CONFIG["colors"]["text"])])
        _add_text(slide, 9.2, y + 0.2, 1.6, 0.55,
                  [(gain, 15, True, CONFIG["colors"]["highlight"])])
        _add_text(slide, 10.8, y + 0.28, 1.6, 0.5,
                  [(note, 9, False, CONFIG["colors"]["text_light"])])
        y += 1.1
    _add_text(slide, 7.0, 6.6, 5.5, 0.5,
              [("真机口径下 C1/L3 噪声感知价值显著（校准方差大）",
                10, False, CONFIG["colors"]["text_light"])])


def main():
    prs = Presentation()
    prs.slide_width = Cm(CONFIG["layout"]["slide_w"] * 2.54)
    prs.slide_height = Cm(CONFIG["layout"]["slide_h"] * 2.54)
    cover(prs)
    toc(prs)
    arch_page(prs)
    results_page(prs)
    out = Path(__file__).resolve().parent.parent.parent / "作品展示PPT_v2.pptx"
    prs.save(out)
    print(f"已生成 -> {out}")


if __name__ == "__main__":
    main()
