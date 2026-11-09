# -*- coding: utf-8 -*-
"""QTrial 5 分钟展示 PPT v3（2026-11-10）。

排版约定（修复 v2 的布局问题）：
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
    _block(slide, MARGIN, TITLE_Y, 24, title, size=24,
           color=C["primary"], bold=True)
    if subtitle:
        _block(slide, MARGIN, TITLE_Y + 0.85, 24, subtitle, size=11,
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
    y, _ = _block(s, 1.8, y, 26, "QTrail", size=58,
                  color=C["white"], bold=True)
    y += 0.5
    y, _ = _block(s, 1.8, y, 26, [
        ("基于稀疏奖励强化学习的\n自适应量子线路映射优化方案", 27,
         C["white"], False)], line_gap=1.4)
    y += 1.6
    _rect(s, 1.8, y, 3.2, 0.09, C["highlight"], radius=False)
    y += 0.7
    _block(s, 1.8, y, 26, [
        ("面向天衍-287 超导量子处理器 · 零依赖自主实现管线", 14,
         C["white"], False)])
    y += 0.8
    _block(s, 1.8, y, 26, [
        ("2026「揭榜挂帅」擂台赛 · 量子+AI 双向赋能榜题", 12,
         "9FB6D8", False)])


def p2_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "研究背景：问题定义", "量子线路映射 = 布局 + 路由的联合优化", 2)
    x_l, w_l = MARGIN, 15.5
    x_r, w_r = 17.6, PAGE_W - 17.6 - MARGIN
    y = CONTENT_Y + 0.2
    y, _ = _block(s, x_l, y, w_l, [
        ("什么是量子线路映射", 16, C["primary"], True)], line_gap=1.5)
    y += 0.3
    y, _ = _block(s, x_l, y, w_l, [
        ("逻辑线路的双比特门要求作用比特在芯片上相邻；不满足时须插入\n"
         "SWAP 搬运量子态。映射 = 确定比特布局（layout）并插入尽量少\n"
         "的 SWAP（routing），使线路可直接在真机执行。", 12.5,
         C["text"], False)], line_gap=1.5)
    y += 0.5
    y, _ = _block(s, x_l, y, w_l, [
        ("为什么难", 16, C["primary"], True)], line_gap=1.5)
    y += 0.3
    y, _ = _block(s, x_l, y, w_l, [
        ("· 布局是加权二次分配问题（QAP），NP-难；\n"
         "· 布局不当可使 SWAP 数达数万，2Q 门误差 ~0.4% 下保真度\n"
         "  完全崩塌；\n"
         "· 天衍-287：105 比特、182 耦合器，平均距离 6.7 步。", 12.5,
         C["text"], False)], line_gap=1.5)
    y += 0.5
    y, _ = _block(s, x_l, y, w_l, [
        ("本方案：AI 布局 + 自主编译全栈", 16, C["highlight"], True)],
        line_gap=1.5)
    y += 0.3
    _block(s, x_l, y, w_l, [
        ("强化学习求解布局；路由与后处理完全自主实现（零 qiskit）。",
         12.5, C["text"], False)], line_gap=1.5)
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
           "零 qiskit 依赖 · 输入 QASM → 输出 QCIS + 指标", 3)
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
        ("· 表示层：解析器 / 电路 IR / 程序图（纯 Python + numpy）", 11.5,
         C["text"], False)], line_gap=1.5)
    y2, _ = _block(s, MARGIN, y2, PAGE_W - 2 * MARGIN, [
        ("· 学习层：强化学习布局（蓝色块）· 路由层：自研 SABRE（front "
         "layer + 衰减 + 前瞻）", 11.5, C["text"], False)], line_gap=1.5)
    _block(s, MARGIN, y2, PAGE_W - 2 * MARGIN, [
        ("· 变换层：后处理栈 v5（蓝色块）——全部为保语义等价变换，"
         "态矢量验证 + 硬件合法性断言", 11.5, C["text"], False)],
        line_gap=1.5)


def p4_rl(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "技术方案（一）：强化学习布局", "CO-MAP 范式复现 + 三项噪声感知扩展", 4)
    col_w = (PAGE_W - 2 * MARGIN - 0.8) / 2
    x1, x2 = MARGIN, MARGIN + col_w + 0.8
    y = CONTENT_Y + 0.3
    # 左列：QAP 建模
    y, _ = _block(s, x1, y, col_w, [("QAP 建模与稀疏奖励", 15,
                                    C["primary"], True)], line_gap=1.5)
    y += 0.25
    y, _ = _block(s, x1, y, col_w, [
        ("代价 C(π) = Σ w_ij · 2 · d(π(qi), π(qj))\n"
         "奖励 R = −C(π) / Σw_ij（终局一次性，稀疏）\n"
         "噪声感知：D_eff = D_topo + λn·(D_noise − D_topo)", 12,
         C["text"], False)], line_gap=1.5)
    y += 0.55
    y, _ = _block(s, x1, y, col_w, [("模型与训练", 15, C["primary"], True)],
                  line_gap=1.5)
    y += 0.25
    y, _ = _block(s, x1, y, col_w, [
        ("GAT 编码器（4 层 × 8 头）+ 指针注意力解码器\n"
         "REINFORCE + 自批评贪心基线（零额外网络）\n"
         "训练数据：ER 随机图 + MQTBench/QUEKO 图池混合", 12,
         C["text"], False)], line_gap=1.5)
    y += 0.55
    y, _ = _block(s, x1, y, col_w, [("三项扩展（QTrail 优化）", 15,
                                    C["highlight"], True)], line_gap=1.5)
    y += 0.25
    _block(s, x1, y, col_w, [
        ("① 噪声感知映射：7 维校准特征 + 噪声加权距离\n"
         "② 多起点解码（5 采样 + 4 乱序贪心）+ 两阶段局部搜索\n"
         "③ 深度感知静态奖励 + 时序感知程序图", 12, C["text"], False)],
        line_gap=1.5)
    # 右列：推理流程卡片
    y = CONTENT_Y + 0.3
    steps = [
        ("推理流程（单条线路）", None, C["bg_light"], C["primary"], True),
        ("多起点解码", "程序图 → GAT 编码 → 指针解码\n10 起点 × 局部搜索 5000 步",
         C["bg_gray"], C["text"], False),
        ("候选池竞技", "RL 候选 + λ0 拓扑候选\n+ 启发式 + 平凡布局",
         C["bg_gray"], C["text"], False),
        ("路由实测评分", "自研 SABRE 实测\nswap / fidelity / depth 三规则决胜",
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
    _decor(s, "技术方案（二）：自主路由与后处理栈 v5",
           "SABRE 机制完整复现 · 置换共轭后处理 · 本轮改进组件", 5)
    col_w = (PAGE_W - 2 * MARGIN - 0.8) / 2
    x1, x2 = MARGIN, MARGIN + col_w + 0.8
    y = CONTENT_Y + 0.3
    y, _ = _block(s, x1, y, col_w, [("自研 SABRE 路由器", 15,
                                    C["primary"], True)], line_gap=1.5)
    y += 0.25
    y, _ = _block(s, x1, y, col_w, [
        ("front layer 并行执行 + extended set 前瞻\n"
         "逐门衰减 w = 0.5^t（防困难门绑架）+ 交换历史惩罚\n"
         "增量维护 + Numba 融合内核（v1/v2 逐位等价 A/B 门禁）",
         12, C["text"], False)], line_gap=1.5)
    y += 0.55
    y, _ = _block(s, x1, y, col_w, [("后处理栈（保语义等价变换）", 15,
                                    C["primary"], True)], line_gap=1.5)
    y += 0.25
    y, _ = _block(s, x1, y, col_w, [
        ("SWAP 共轭推挤（连通性感知）+ 同向 CX 消解\n"
         "尾部置换吸收 + 平台基分解（ZSXXZ 模板，2.5e-16）\n"
         "QCIS 原生发射（X2P/RZ/CZ/M，真机标签）", 12,
         C["text"], False)], line_gap=1.5)
    y += 0.55
    _block(s, x1, y, col_w, [
        ("正确性：态矢量等价测试 + 硬件合法性断言 + 非劣化开关",
         11.5, C["text_light"], False)], line_gap=1.5)
    # 右列：v5 改进组件
    y = CONTENT_Y + 0.3
    y, _ = _block(s, x2, y, col_w, [("v5 改进组件（本轮工作）", 15,
                                    C["highlight"], True)], line_gap=1.5)
    y += 0.3
    comps = [
        ("B · 1Q 游程优化", "游程矩阵合并 → u3 再合成\nH² 奇偶消解（消融：收益主体）", C["bg_light"]),
        ("C1 · 噪声感知重标记", "贪心合法重定位\n（真机校准下 +1.4%~219%）", C["bg_light"]),
        ("L3 · 学习式重标记", "复用 QAPolicy 解码候选\n零重训练", C["bg_light"]),
        ("A2 · 2Q 块重合成", "KAK 分解 + 参数拟合\n（同对块上限 1.3%，已记录）", C["bg_light"]),
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
    _decor(s, "实验设置：587 条线路 × 六方法", "三层统计口径 · 原始数据逐线路落盘", 6)
    col_w = (PAGE_W - 2 * MARGIN - 1.4) / 3
    y = CONTENT_Y + 0.4
    cards = [
        ("基准数据集（587 条）",
         "Benchpress（IBM 官方集）\n≤105 比特全部纳入\n"
         "MQTBench（36 类基准）\nQUEKO（高密度映射极端场景）\n"
         "分层：≤10（129）· 11-50（259）· 51-105（199）", C["bg_light"]),
        ("对比方法（六方法）",
         "本方案 QTrial（零依赖管线）\n"
         "噪声感知 O1 / O3（Target 注入）\n"
         "盲目 O1 / O3（SABRE 编译）\n"
         "pytket（工业编译器）", C["bg_light"]),
        ("统计口径",
         "均值 / 截尾均值（去两端 10%）/ 中位数\n"
         "指标：SWAP 数 · 深度 · 2Q 深度 · 估计保真度\n"
         "逐线路原始数据 JSONL 落盘（可回溯）\n"
         "断点续跑 · 一键复现", C["bg_light"]),
    ]
    x = MARGIN
    for title, body, fill in cards:
        _card(s, x, y, col_w, 8.6, title, body, fill=fill)
        x += col_w + 0.7
    # 底部说明条
    _rect(s, MARGIN, y + 9.0, PAGE_W - 2 * MARGIN, 2.3, C["bg_gray"])
    _block(s, MARGIN + 0.5, y + 9.25, PAGE_W - 2 * MARGIN - 1.0, [
        ("真机口径：天衍-287 平台 API 实测——182 耦合器弯曲排布（非矩形\n"
         "网格）、4 禁用比特、168 可用耦合器；平台真实校准快照。"
         "合成口径 = 评测基线（188 边矩形网格 + 合成校准）。", 11.5,
         C["text"], False)], line_gap=1.5)


def p7_results(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "实验结果：分层估计保真度", "QTrial vs 感知 O3 / pytket（同集合口径）", 7)
    col_w = (PAGE_W - 2 * MARGIN - 1.4) / 3
    x = MARGIN
    y = CONTENT_Y + 0.4
    data = [
        ("≤10 比特（128 条）",
         [("QTrial", "0.555 / 0.572 / 0.676", C["primary"]),
          ("感知 O3", "0.761 / 0.803 / 0.894", C["text_light"]),
          ("pytket", "0.663 / 0.700 / 0.800", C["text_light"])],
         "vs 盲目 O1 胜场 86 条"),
        ("11-50 比特（258 条）",
         [("QTrial", "0.055 / 0.017 / 0.000", C["primary"]),
          ("感知 O3", "0.160 / 0.109 / 0.028", C["text_light"]),
          ("pytket", "0.105 / 0.055 / 0.000", C["text_light"])],
         "vs 盲目 O1 胜场 126 条"),
        ("51-105 比特（199 条）",
         [("QTrial", "0.026 / 0.000 / 0.000", C["primary"]),
          ("感知 O3", "0.021 / 0.000 / 0.000", C["text_light"]),
          ("pytket", "0.012 / 0.000 / 0.000", C["text_light"])],
         "真机拓扑口径（弯曲结构）"),
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
        ("关键数字（587 条，均值口径）", 12.5, C["primary"], True)],
        line_gap=1.5)
    _block(s, MARGIN + 0.5, y + 9.55, PAGE_W - 2 * MARGIN - 1.0, [
        ("总体保真度 QTrial 0.146 vs 感知 O3 0.244（0.60×）· 超越 pytket\n"
         "逐线路胜场：vs pytket 134 条 · vs 盲目 O1 226 条\n"
         "诚实边界：大线路层与感知 O3 差距为管线级机制差异（2Q 块重合成）",
         11.5, C["text"], False)], line_gap=1.5)


def p8_summary(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "总结与展望", "创新点 · 诚实边界 · 后续方向", 8)
    y = CONTENT_Y + 0.4
    y, _ = _block(s, MARGIN, y, PAGE_W - 2 * MARGIN,
                  [("创新点", 15, C["primary"], True)], line_gap=1.5)
    y += 0.3
    y, _ = _block(s, MARGIN, y, PAGE_W - 2 * MARGIN, [
        ("· 学习式布局：QAP 建模 + GAT/指针网络，设备特征自归一化 → 跨设备零重训迁移\n"
         "· 噪声感知贯穿全链：距离插值 + 校准特征 + QuEst 预测器（T1/T2 感知）\n"
         "· 零依赖自主实现：解析器 → 路由器 → 后处理 → QCIS 全自研（评委环境零安装）\n"
         "· v5 后段改进：1Q 游程优化（消融 +2%~+12%）· 噪声感知重标记（真机口径显著）",
         12, C["text"], False)], line_gap=1.5)
    y += 0.5
    y, _ = _block(s, MARGIN, y, PAGE_W - 2 * MARGIN,
                  [("诚实边界与后续方向", 15, C["primary"], True)],
                  line_gap=1.5)
    y += 0.3
    _block(s, MARGIN, y, PAGE_W - 2 * MARGIN, [
        ("· 边界：大线路层与感知 O3 的差距来自管线级机制（2Q 块重合成 / 近似合成），\n"
         "  运行时间为 Python 常数（感知 O3 为 Rust 实现）——报告书已如实披露\n"
         "· 方向：跨对 2Q 块合成（ConsolidateBlocks 语义）· 噪声感知近似合成 · "
         "真机恢复后实测验证",
        12, C["text"], False)], line_gap=1.5)


def main():
    prs = Presentation()
    prs.slide_width = Cm(PAGE_W)
    prs.slide_height = Cm(PAGE_H)
    p1_cover(prs)
    p2_problem(prs)
    p3_pipeline(prs)
    p4_rl(prs)
    p5_post(prs)
    p6_setup(prs)
    p7_results(prs)
    p8_summary(prs)
    out = Path(__file__).resolve().parent.parent.parent / "作品展示PPT_v3.pptx"
    prs.save(out)
    print(f"已生成 -> {out}（8 页 · 5 分钟展示版）")


if __name__ == "__main__":
    main()
