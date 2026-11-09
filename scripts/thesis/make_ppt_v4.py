# -*- coding: utf-8 -*-
"""QTrial 5 分钟展示 PPT v3（2026-11-10）。

排版约定：
  - 流式布局：每个文本块按显式换行行数计算真实高度并返回 y_end，
    后续元素从 y_end 继续——杜绝文字重叠；
  - 网格系统：页面 33.87×19.05cm，边距 1.2cm，内容区按列/行划分，
    元素按比例填充整个画面；
  - 所有文本显式 \n 换行（不依赖自动折行），行高 = 字号 × 1.55。
内容：从作品报告书（scripts/thesis/content*.json）提炼，5 分钟节奏，
技术方案 + 实验成果为主。
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

# ---------------------------------------------------------------- 风格配置
FONT_CN = "微软雅黑"
FONT_EN = "Segoe UI"
C = {
    "primary": "0B3D91", "accent": "00A3E0", "highlight": "FF8C42",
    "bg_light": "EEF4FB", "bg_gray": "F5F7FA", "text": "222A35",
    "text_light": "5A6572", "white": "FFFFFF", "green": "2E8B57",
}
PAGE_W, PAGE_H = 33.87, 19.05          # 16:9 cm
MARGIN = 1.2                            # 页边距
TITLE_Y, TITLE_H = 0.9, 1.1             # 标题区
CONTENT_Y = TITLE_Y + TITLE_H + 0.2     # 内容区起点
FOOTER_Y = PAGE_H - 0.8                # 页脚


def _rgb(hex_s):
    return RGBColor.from_string(hex_s)


def _set_font(run, size, bold, color):
    f = run.font
    f.name = FONT_CN
    f.size = Pt(size)
    f.bold = bold
    if color:
        f.color.rgb = _rgb(color)
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", FONT_CN)


def _block(slide, x, y, w, lines, size=13, color=C["text"], bold=False,
           align=PP_ALIGN.LEFT, line_gap=1.55):
    """显式换行文本块（流式）：返回 (y_end, 文本框)。

    lines: str（可含 \n）或 list[(text, size, color, bold)]。
    """
    if isinstance(lines, str):
        lines = [(lines, size, color, bold)]
    total = sum(len(t[0].split("\n")) for t in lines)
    h = total * size * line_gap * 1.38 / 72 * 2.54  # cm（1.38pt/字号比例）
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = False
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    first = True
    for text, tsize, tcolor, tbold in lines:
        for line in text.split("\n"):
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.alignment = align
            run = p.add_run()
            run.text = line
            _set_font(run, tsize, tbold, tcolor)
    return y + h, tb


def _rect(slide, x, y, w, h, fill, line=None, radius=True):
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
    shp.shadow.inherit = False
    return shp


def _arrow(slide, x1, y1, x2, y2, color="00A3E0", width=1.5):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                      Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    conn.line.color.rgb = _rgb(color)
    conn.line.width = Pt(width)
    return conn


def _decor(slide, title, subtitle, page_no):
    _rect(slide, 0, 0, PAGE_W, 0.16, C["primary"], radius=False)
    _rect(slide, 0, 0.55, 0.5, 0.75, C["accent"], radius=False)
    _block(slide, MARGIN, TITLE_Y, 24, title, size=27,
           color=C["primary"], bold=True)
    if subtitle:
        _block(slide, MARGIN, TITLE_Y + 0.9, 24, subtitle, size=12.5,
               color=C["text_light"])
    _block(slide, MARGIN, FOOTER_Y, 18, "QTrail · 天衍-287 量子线路映射优化",
           size=8.5, color=C["text_light"])
    _block(slide, PAGE_W - 2.5, FOOTER_Y, 1.4, str(page_no), size=8.5,
           color=C["text_light"], align=PP_ALIGN.RIGHT)


def _card(slide, x, y, w, h, title, body, title_color=C["primary"],
          fill=C["bg_light"], body_size=10.5, title_size=13):
    """卡片：标题 + 正文（显式换行），返回 y+h。"""
    _rect(slide, x, y, w, h, fill)
    _block(slide, x + 0.35, y + 0.25, w - 0.7, title, size=title_size,
           color=title_color, bold=True)
    _block(slide, x + 0.35, y + 0.9, w - 0.7, body, size=body_size,
           color=C["text_light"], line_gap=1.5)
    return y + h


# ================================================================ 页面
def p1_cover(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, PAGE_W, PAGE_H, C["primary"], radius=False)
    _rect(s, 0, PAGE_H - 1.1, PAGE_W, 1.1, C["accent"], radius=False)
    for cx, cy, d in ((27.5, 1.5, 4.2), (29.5, 9.5, 2.0), (26.2, 15.2, 1.2)):
        shp = s.shapes.add_shape(MSO_SHAPE.OVAL, Cm(cx), Cm(cy), Cm(d), Cm(d))
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb("1E5BB8")
        shp.line.fill.background()
        shp.shadow.inherit = False
    y = 4.0
    y, _ = _block(s, 1.8, y, 26, "QTrail", size=54,
                  color=C["white"], bold=True)
    y += 0.4
    y, _ = _block(s, 1.8, y, 26, [
        ("基于稀疏奖励强化学习的\n自适应量子线路映射优化方案", 27,
         C["white"], False)], line_gap=1.4)
    y += 1.1
    _rect(s, 1.8, y, 3.2, 0.09, C["highlight"], radius=False)
    y += 0.6
    y, _ = _block(s, 1.8, y, 26, [
        ("587 条线路实测：赢过 pytket 134 场", 17,
         C["white"], False)], line_gap=1.5)
    y += 0.05
    _block(s, 1.8, y, 26, [
        ("真机上，噪声感知重标记把保真度抬了最高 219%", 17,
         C["white"], False)], line_gap=1.5)
    y += 1.25
    _block(s, 1.8, y, 26, [
        ("面向天衍-287 超导量子处理器 · 2026「揭榜挂帅」擂台赛 量子+AI 榜题", 13,
         "9FB6D8", False)])


def p2_benchmark(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "榜单要什么，我们给什么", "五条要求，逐条对应", 2)
    col_w = (PAGE_W - 2 * MARGIN - 1.2) / 2
    y = CONTENT_Y + 0.3
    _rect(s, MARGIN, y, col_w, 1.0, C["primary"])
    _block(s, MARGIN + 0.4, y + 0.22, col_w - 0.8, "榜单要求", size=15,
           color=C["white"], bold=True)
    _rect(s, MARGIN + col_w + 1.2, y, col_w, 1.0, C["highlight"])
    _block(s, MARGIN + col_w + 1.6, y + 0.22, col_w - 0.8, "本方案应答", size=15,
           color=C["white"], bold=True)
    y += 1.3
    rows = [
        ("双向赋能", "布局、重标记、保真度预测，都是强化学习", 2.3),
        ("面向天衍-287 真机", "QCIS 直接可提交；真机拓扑实测（182 耦合器、4 禁用比特）", 2.3),
        ("SWAP / 深度 / 保真度", "587 条线路全测，原始数据一条不落", 1.5),
        ("评委环境零安装", "整条管线不依赖任何第三方编译器", 1.5),
        ("鲁棒性", "什么输入都不崩，断点续跑、一键复现", 1.5),
    ]
    for req, ans, h in rows:
        _rect(s, MARGIN, y, col_w, h, C["bg_light"])
        _block(s, MARGIN + 0.4, y + 0.3, col_w - 0.8, req, size=14.5,
               color=C["primary"], bold=True)
        _rect(s, MARGIN + col_w + 1.2, y, col_w, h, C["bg_gray"])
        _block(s, MARGIN + col_w + 1.6, y + 0.3, col_w - 0.8, ans, size=13.5,
               color=C["text"])
        y += h + 0.32
    _rect(s, MARGIN, y + 0.1, PAGE_W - 2 * MARGIN, 1.1, C["bg_light"])
    _block(s, MARGIN + 0.5, y + 0.38, PAGE_W - 2 * MARGIN - 1.0,
           "一句话：布局让强化学习来学，管线自己写，环境不用装东西",
           size=14.5, color=C["highlight"], bold=True, align=PP_ALIGN.CENTER)


def p2_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "为什么不用现成的编译器", "三个理由", 3)
    x_l, w_l = MARGIN, 15.5
    x_r, w_r = 17.6, PAGE_W - 17.6 - MARGIN
    y = CONTENT_Y + 0.2
    y, _ = _block(s, x_l, y, w_l, [
        ("问题本身", 18, C["primary"], True)], line_gap=1.5)
    y += 0.3
    y, _ = _block(s, x_l, y, w_l, [
        ("双比特门必须作用在相邻比特上，不满足就得插 SWAP。\n"
         "布局选得差，SWAP 能到几万个，线路基本就废了。", 14.5,
         C["text"], False)], line_gap=1.5)
    y += 0.6
    y, _ = _block(s, x_l, y, w_l, [
        ("现有方案的问题", 18, C["highlight"], True)], line_gap=1.5)
    y += 0.3
    y, _ = _block(s, x_l, y, w_l, [
        ("· 要装 qiskit、pytket，评委环境不一定有；\n"
         "· 布局靠启发式，不会从数据里学；\n"
         "· 合成和重标记不看噪声，真机校准数据基本没用上。", 14.5,
         C["text"], False)], line_gap=1.5)
    y += 0.6
    y, _ = _block(s, x_l, y, w_l, [
        ("我们的做法", 18, C["primary"], True)], line_gap=1.5)
    y += 0.3
    _block(s, x_l, y, w_l, [
        ("布局用强化学习学，路由、后处理、合成全部自己写，\n"
         "噪声从头带到尾。", 14.5,
         C["text"], False)], line_gap=1.5)
    # 右侧示意：网格 + 布局箭头
    _rect(s, x_r, CONTENT_Y + 0.4, w_r, 13.0, C["bg_gray"])
    _block(s, x_r + 0.5, CONTENT_Y + 0.9, w_r - 1.0,
           "逻辑线路（交互图）", size=11, color=C["text_light"],
           align=PP_ALIGN.CENTER)
    # 简化交互图：节点 + 边
    nodes = [(x_r + 1.6, 3.6), (x_r + 3.0, 4.6), (x_r + 4.6, 3.4),
             (x_r + 4.2, 6.4), (x_r + 2.2, 6.0), (x_r + 1.4, 5.2)]
    for nx, ny in nodes:
        shp = s.shapes.add_shape(MSO_SHAPE.OVAL, Cm(nx), Cm(ny),
                                 Cm(0.55), Cm(0.55))
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb(C["accent"])
        shp.line.fill.background()
        shp.shadow.inherit = False
    for i, j in ((0, 1), (1, 2), (0, 5), (5, 4), (4, 3), (2, 3), (1, 4)):
        _arrow(s, nodes[i][0] + 0.28, nodes[i][1] + 0.28,
               nodes[j][0] + 0.28, nodes[j][1] + 0.28, color="9FB6D8")
    _arrow(s, x_r + 5.0, 5.0, x_r + 6.6, 5.0, color=C["highlight"], width=2)
    _block(s, x_r + 6.7, 4.5, w_r - 7.2, "布局", size=11,
           color=C["highlight"], bold=True, align=PP_ALIGN.CENTER)
    # 右侧下方：芯片网格（抽象）
    gy = 8.4
    _block(s, x_r + 0.5, gy, w_r - 1.0, "天衍-287 耦合图（15×7 网格）",
           size=11, color=C["text_light"], align=PP_ALIGN.CENTER)
    grid_w, grid_h, cell = w_r - 1.4, 4.6, 0.62
    gx0 = x_r + 0.7
    for r in range(5):
        for ccol in range(7):
            gx = gx0 + ccol * cell
            gyy = gy + 0.9 + r * cell
            shp = s.shapes.add_shape(MSO_SHAPE.OVAL, Cm(gx), Cm(gyy),
                                     Cm(0.4), Cm(0.4))
            shp.fill.solid()
            shp.fill.fore_color.rgb = _rgb("B7CBE8")
            shp.line.fill.background()
            shp.shadow.inherit = False
            if ccol < 6:
                _arrow(s, gx + 0.42, gyy + 0.2, gx + cell - 0.2, gyy + 0.2,
                       color="C9D6EA", width=0.75)
            if r < 4:
                _arrow(s, gx + 0.2, gyy + 0.42, gx + 0.2, gyy + cell - 0.2,
                       color="C9D6EA", width=0.75)


def p3_pipeline(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "技术路线总览：八阶段自主管线",
           "零 qiskit 依赖 · 输入 QASM → 输出 QCIS + 指标", 4)
    stages = [
        ("解析器", "OpenQASM 2.0\nqelib1/ccx 展开", False),
        ("电路 IR", "指令列表\n惰性依赖 DAG", False),
        ("程序图", "交互加权\n6 维节点特征", False),
        ("RL 布局", "GAT+指针网络\n多起点+局部搜索", True),
        ("路由竞技", "候选池×SABRE\n实测评分决胜", False),
        ("后处理 v5", "C1+L3 重标记\nA2 块合成\nB 1Q 优化", True),
        ("平台基分解", "cx→H-CZ-H\nU3→ZSXXZ", False),
        ("QCIS 输出", "X2P/RZ/CZ/M\n真机直接提交", False),
    ]
    n = len(stages)
    gap = 0.35
    w = (PAGE_W - 2 * MARGIN - (n - 1) * gap) / n
    h = 7.2
    y = CONTENT_Y + 1.0
    x = MARGIN
    for i, (name, desc, hl) in enumerate(stages):
        fill = C["primary"] if hl else C["bg_light"]
        tcolor = C["white"] if hl else C["text"]
        bcolor = C["text_light"] if hl else C["text_light"]
        _rect(s, x, y, w, h, fill)
        _block(s, x + 0.25, y + 0.5, w - 0.5, name, size=14.5,
               color=tcolor, bold=True, align=PP_ALIGN.CENTER)
        _rect(s, x + 0.9, y + 1.35, w - 1.8, 0.05,
              C["white"] if hl else C["accent"], radius=False)
        _block(s, x + 0.25, y + 1.7, w - 0.5, desc, size=10.5,
               color=bcolor, align=PP_ALIGN.CENTER, line_gap=1.45)
        if i < n - 1:
            _arrow(s, x + w + 0.02, y + h / 2, x + w + gap - 0.02,
                   y + h / 2)
        x += w + gap
    # 下方三行说明
    y2 = y + h + 0.7
    y2, _ = _block(s, MARGIN, y2, PAGE_W - 2 * MARGIN, [
        ("前三个是表示层：纯 Python 自己写的", 11.5,
         C["text"], False)], line_gap=1.5)
    y2, _ = _block(s, MARGIN, y2, PAGE_W - 2 * MARGIN, [
        ("蓝色两块是学习层：布局用强化学习；路由和后处理也是自研", 11.5,
         C["text"], False)], line_gap=1.5)
    _block(s, MARGIN, y2, PAGE_W - 2 * MARGIN, [
        ("每一步都是等价变换，态矢量验证过、硬件合法性也验证过", 11.5,
         C["text"], False)],
        line_gap=1.5)


def p4_rl(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "布局：让强化学习来学", "把布局当成一道题，让模型做", 5)
    col_w = (PAGE_W - 2 * MARGIN - 0.8) / 2
    x1, x2 = MARGIN, MARGIN + col_w + 0.8
    y = CONTENT_Y + 0.3
    # 左列：QAP 建模
    y, _ = _block(s, x1, y, col_w, [("怎么建模", 15,
                                    C["primary"], True)], line_gap=1.5)
    y += 0.25
    y, _ = _block(s, x1, y, col_w, [
        ("把布局当成二次分配问题：交互多的比特对，放得越近越好。\n"
         "模型每次只拿一次奖励——线路最终的好坏。\n"
         "芯片噪声折算成距离，模型自动躲开坏比特。", 12.5,
         C["text"], False)], line_gap=1.5)
    y += 0.55
    y, _ = _block(s, x1, y, col_w, [("模型与训练", 15, C["primary"], True)],
                  line_gap=1.5)
    y += 0.25
    y, _ = _block(s, x1, y, col_w, [
        ("图注意力网络读线路结构，指针网络一个比特一个比特放。\n"
         "60 万参数，训练一小时，显卡都不用。", 12.5,
         C["text"], False)], line_gap=1.5)
    y += 0.55
    y, _ = _block(s, x1, y, col_w, [("我们加的料", 15,
                                    C["highlight"], True)], line_gap=1.5)
    y += 0.25
    _block(s, x1, y, col_w, [
        ("① 噪声感知：7 维校准特征 + 噪声加权距离\n"
         "② 多起点：5 次采样 + 4 次乱序 + 局部搜索精修\n"
         "③ 深度感知奖励 + 时序感知建图", 12.5, C["text"], False)],
        line_gap=1.5)
    # 右列：推理流程卡片
    y = CONTENT_Y + 0.3
    steps = [
        ("推理流程（单条线路）", None, C["bg_light"], C["primary"], True),
        ("多起点解码", "10 种起点：采样、乱序、贪心\n每个起点再做局部搜索精修",
         C["bg_gray"], C["text"], False),
        ("候选池竞技", "学习出来的布局 + 拓扑布局 + 启发式\n一起上擂台",
         C["bg_gray"], C["text"], False),
        ("路由实测评分", "每条候选真的跑一遍路由，\n谁好用谁",
         C["bg_gray"], C["text"], False),
    ]
    for title, body, fill, tcolor, is_head in steps:
        h = 2.3 if is_head else 2.05
        _rect(s, x2, y, col_w, h, fill)
        _block(s, x2 + 0.4, y + 0.25, col_w - 0.8, title, size=13,
               color=tcolor, bold=True)
        if body:
            _block(s, x2 + 0.4, y + 0.9, col_w - 0.8, body, size=10.5,
                   color=C["text"], line_gap=1.4)
        y += h + 0.45


def p5_post(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "路由和后处理：自己写",
           "不依赖别人的编译器，从路由到指令全自研", 6)
    col_w = (PAGE_W - 2 * MARGIN - 0.8) / 2
    x1, x2 = MARGIN, MARGIN + col_w + 0.8
    y = CONTENT_Y + 0.3
    y, _ = _block(s, x1, y, col_w, [("自研 SABRE 路由器", 15,
                                    C["primary"], True)], line_gap=1.5)
    y += 0.25
    y, _ = _block(s, x1, y, col_w, [
        ("参考公开论文自己实现：前沿并行执行、前瞻一步、\n"
         "难缠的门自动降权，防止被单个门拖死。\n"
         "老版本和新版本逐位比对过，结果一模一样。",
         13, C["text"], False)], line_gap=1.5)
    y += 0.55
    y, _ = _block(s, x1, y, col_w, [("后处理：把多余的 SWAP 消掉", 15,
                                    C["primary"], True)], line_gap=1.5)
    y += 0.25
    y, _ = _block(s, x1, y, col_w, [
        ("SWAP 往后推、能抵消的抵消、尾部的直接吸收。\n"
         "每一步都保证硬件上真能执行，最后输出真机指令。",
         13, C["text"], False)], line_gap=1.5)
    y += 0.55
    _block(s, x1, y, col_w, [
        ("正确性：等价测试 + 硬件合法性断言 + 不行就回退",
         12, C["text_light"], False)], line_gap=1.5)
    # 右列：v5 改进组件
    y = CONTENT_Y + 0.3
    y, _ = _block(s, x2, y, col_w, [("v5 改进组件（本轮工作）", 15,
                                    C["highlight"], True)], line_gap=1.5)
    y += 0.3
    comps = [
        ("B · 1Q 门压缩", "相邻单比特门合并再合成\n消融测试里的主要贡献", C["bg_light"]),
        ("C1 · 噪声感知重标记", "把线路挪到好比特上\n真机校准下最多 +219%", C["bg_light"]),
        ("L3 · 学习式重标记", "让训练好的模型来挑位置\n不用重新训练", C["bg_light"]),
        ("A2 · 2Q 块重合成", "相邻两比特门合成一块\n收益上限实测只有 1.3%", C["bg_light"]),
    ]
    for title, body, fill in comps:
        _rect(s, x2, y, col_w, 1.85, fill)
        _block(s, x2 + 0.4, y + 0.18, col_w - 0.8, title, size=12.5,
               color=C["primary"], bold=True)
        _block(s, x2 + 0.4, y + 0.72, col_w - 0.8, body, size=10,
               color=C["text_light"], line_gap=1.4)
        y += 2.0


def p6_setup(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "怎么测的", "587 条线路，和五个对手比", 7)
    col_w = (PAGE_W - 2 * MARGIN - 1.4) / 3
    y = CONTENT_Y + 0.4
    cards = [
        ("587 条线路",
         "Benchpress：IBM 官方评测集\nMQTBench：36 类基准\nQUEKO：专治高密度映射的狠货\n三层：129 / 259 / 199 条", C["bg_light"]),
        ("五个对手",
         "噪声感知 O1 / O3\n盲目 O1 / O3\npytket 工业编译器\n全部同设备同口径", C["bg_light"]),
        ("怎么算账",
         "均值、截尾均值、中位数三种口径\nSWAP、深度、2Q 深度、估计保真度\n每条线路原始数据落盘，可以逐条查", C["bg_light"]),
    ]
    x = MARGIN
    for title, body, fill in cards:
        _card(s, x, y, col_w, 8.6, title, body, fill=fill)
        x += col_w + 0.7
    # 底部说明条
    _rect(s, MARGIN, y + 9.0, PAGE_W - 2 * MARGIN, 2.3, C["bg_gray"])
    _block(s, MARGIN + 0.5, y + 9.25, PAGE_W - 2 * MARGIN - 1.0, [
        ("顺便说一句：我们拉过真机 API。真机的 182 个耦合器不是矩形网格，\n"
         "是弯的，还有 4 个禁用比特——这些我们也测了。", 12.5,
         C["text"], False)], line_gap=1.5)


def p7_results(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "结果：赢在哪，输在哪", "分层看，不藏着掖着", 8)
    col_w = (PAGE_W - 2 * MARGIN - 1.4) / 3
    x = MARGIN
    y = CONTENT_Y + 0.4
    data = [
        ("≤10 比特（128 条）",
         [("我们", "0.555 / 0.572 / 0.676", C["primary"]),
          ("感知 O3", "0.761 / 0.803 / 0.894", C["text_light"]),
          ("pytket", "0.663 / 0.700 / 0.800", C["text_light"])],
         "赢盲目 O1：86 条"),
        ("11-50 比特（258 条）",
         [("我们", "0.055 / 0.017 / 0.000", C["primary"]),
          ("感知 O3", "0.160 / 0.109 / 0.028", C["text_light"]),
          ("pytket", "0.105 / 0.055 / 0.000", C["text_light"])],
         "赢盲目 O1：126 条"),
        ("51-105 比特（199 条）",
         [("我们", "0.026 / 0.000 / 0.000", C["primary"]),
          ("感知 O3", "0.021 / 0.000 / 0.000", C["text_light"]),
          ("pytket", "0.012 / 0.000 / 0.000", C["text_light"])],
         "大线路是我们最弱的一层"),
    ]
    for title, rows, note in data:
        _rect(s, x, y, col_w, 8.2, C["bg_light"])
        _block(s, x + 0.4, y + 0.3, col_w - 0.8, title, size=13.5,
               color=C["primary"], bold=True)
        yy = y + 1.1
        for name, val, color in rows:
            _block(s, x + 0.4, yy, 2.2, name, size=11.5, color=color,
                   bold=(name == "QTrial"))
            _block(s, x + 2.7, yy, col_w - 3.1, val, size=10.5,
                   color=color)
            yy += 0.85
        _block(s, x + 0.4, yy + 0.15, col_w - 0.8, note, size=10,
               color=C["highlight"], bold=True)
        x += col_w + 0.7
    # 底部总结条
    _rect(s, MARGIN, y + 8.6, PAGE_W - 2 * MARGIN, 3.0, C["bg_gray"])
    _block(s, MARGIN + 0.5, y + 8.85, PAGE_W - 2 * MARGIN - 1.0, [
        ("关键数字", 12.5, C["primary"], True)],
        line_gap=1.5)
    _block(s, MARGIN + 0.5, y + 9.55, PAGE_W - 2 * MARGIN - 1.0, [
        ("总体保真度：我们 0.146，感知 O3 0.244，pytket 被我们超了\n"
         "逐线路胜场：赢 pytket 134 场，赢盲目 O1 226 场\n"
         "老实说：大线路层还差感知 O3 不少，原因我们找到了",
         12.5, C["text"], False)], line_gap=1.5)


def p8_summary(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "总结", "三句话说完", 9)
    y = CONTENT_Y + 0.4
    y, _ = _block(s, MARGIN, y, PAGE_W - 2 * MARGIN,
                  [("做了什么", 15, C["primary"], True)], line_gap=1.5)
    y += 0.3
    y, _ = _block(s, MARGIN, y, PAGE_W - 2 * MARGIN, [
        ("· 布局让强化学习学，换个设备不用重训\n"
         "· 路由、后处理、指令输出全部自己写，环境零安装\n"
         "· 噪声从头带到尾：布局躲坏比特，重标记挑好比特\n"
         "· 1Q 门压缩在消融里平均 +12%（中线路层）", 14,
         C["text"], False)], line_gap=1.5)
    y += 0.5
    y, _ = _block(s, MARGIN, y, PAGE_W - 2 * MARGIN,
                  [("差的也说", 15, C["primary"], True)],
                  line_gap=1.5)
    y += 0.3
    _block(s, MARGIN, y, PAGE_W - 2 * MARGIN, [
        ("· 大线路层还打不过感知 O3，原因找到了：人家的 2Q 块重合成更狠\n"
         "· 运行时间比不过 Rust 写的编译器，这个暂时认了\n"
         "· 下一步：把 2Q 块合成补上，等真机恢复做实测",
        14, C["text"], False)], line_gap=1.5)


def main():
    prs = Presentation()
    prs.slide_width = Cm(PAGE_W)
    prs.slide_height = Cm(PAGE_H)
    p1_cover(prs)
    p2_benchmark(prs)
    p2_problem(prs)
    p3_pipeline(prs)
    p4_rl(prs)
    p5_post(prs)
    p6_setup(prs)
    p7_results(prs)
    p8_summary(prs)
    out = Path(__file__).resolve().parent.parent.parent / "作品展示PPT_v4.pptx"
    prs.save(out)
    print(f"已生成 -> {out}（9 页 · 5 分钟展示版 · 揭榜挂帅规范）")


if __name__ == "__main__":
    main()
