# -*- coding: utf-8 -*-
"""QTrial 决赛 6 分钟展示 PPT（2026-11-12）。

设计继承 make_ppt_v5.py 的浅色学术视觉体系（投影友好）：
  页面 33.87×19.05cm，边距 1.2cm，显式换行流式布局。
内容口径：v8 生产管线 + tables/submission/summary.md；
对外表述遵循竞赛口径（不出现内部修复史，数字真实可复现）。
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "tables" / "submission" / "figs"
OUT = ROOT / "作品展示PPT_决赛.pptx"

FONT_CN = "微软雅黑"
FONT_EN = "Segoe UI"
C = {
    "primary": "0B3D91", "accent": "00A3E0", "highlight": "FF8C42",
    "bg_light": "EEF4FB", "bg_gray": "F5F7FA", "text": "222A35",
    "text_light": "5A6572", "white": "FFFFFF", "green": "2E8B57",
}
PAGE_W, PAGE_H = 33.87, 19.05
MARGIN = 1.2
TITLE_Y, TITLE_H = 0.8, 1.3
CONTENT_Y = TITLE_Y + TITLE_H + 0.65
FOOTER_Y = PAGE_H - 0.8


def _rgb(h):
    return RGBColor.from_string(h)


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


def _block(slide, x, y, w, lines, size=14, color=C["text"], bold=False,
           align=PP_ALIGN.LEFT, line_gap=1.55):
    if isinstance(lines, str):
        lines = [(lines, size, color, bold)]
    total = sum(len(t[0].split("\n")) for t in lines)
    h = total * size * line_gap * 1.38 / 72 * 2.54
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
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
    _block(slide, MARGIN, TITLE_Y, 30, title, size=26,
           color=C["primary"], bold=True)
    if subtitle:
        _block(slide, MARGIN, TITLE_Y + 1.28, 30, subtitle, size=13,
               color=C["text_light"])
    _block(slide, MARGIN, FOOTER_Y, 20, "QTrail · 天衍-287 量子线路映射优化",
           size=8.5, color=C["text_light"])
    _block(slide, PAGE_W - 2.5, FOOTER_Y, 1.4, str(page_no), size=8.5,
           color=C["text_light"], align=PP_ALIGN.RIGHT)


def _card(slide, x, y, w, h, title, body, title_color=C["primary"],
          fill=C["bg_light"], body_size=12, title_size=15):
    _rect(slide, x, y, w, h, fill)
    _block(slide, x + 0.4, y + 0.3, w - 0.8, title, size=title_size,
           color=title_color, bold=True)
    _block(slide, x + 0.4, y + 1.15, w - 0.8, body, size=body_size,
           color=C["text"], line_gap=1.55)
    return y + h


def _picture(slide, path, x, y, width_cm):
    """按宽度插入图片，返回底部 y。"""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    height = width_cm * h / w
    slide.shapes.add_picture(str(path), Cm(x), Cm(y), Cm(width_cm), Cm(height))
    return y + height


# ================================================================ 页面
def p1_cover(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, PAGE_W, PAGE_H, C["primary"], radius=False)
    _rect(s, 0, PAGE_H - 1.1, PAGE_W, 1.1, C["accent"], radius=False)
    for cx, cy, d in ((27.0, 1.2, 4.4), (29.6, 8.6, 2.2), (26.4, 14.4, 1.2)):
        shp = s.shapes.add_shape(MSO_SHAPE.OVAL, Cm(cx), Cm(cy), Cm(d), Cm(d))
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb("1E5BB8")
        shp.line.fill.background()
        shp.shadow.inherit = False
    y = 3.4
    y, _ = _block(s, 1.8, y, 28, "QTrail", size=56, color=C["white"], bold=True)
    y += 0.3
    y, _ = _block(s, 1.8, y, 28, [
        ("基于稀疏奖励强化学习的\n自适应量子线路映射优化方案", 26, C["white"], False)],
        line_gap=1.4)
    y += 1.0
    _rect(s, 1.8, y, 3.2, 0.09, C["highlight"], radius=False)
    y += 0.55
    y, _ = _block(s, 1.8, y, 28, [
        ("583 条线路分层实测 · 31/31 真机合法 · 端到端保真度 1.000000", 17,
         C["white"], False)], line_gap=1.5)
    y += 0.1
    y, _ = _block(s, 1.8, y, 28, [
        ("映射内核零 qiskit，输出天衍-287 可直接执行的 QCIS", 17,
         "CDE3FF", False)], line_gap=1.5)
    y += 1.1
    _block(s, 1.8, y, 28, [
        ("2026 年度中国青年科技创新「揭榜挂帅」擂台赛 · 量子+AI 双向赋能榜题", 13,
         "9FB6D8", False)])


def p2_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "量子线路映射：AI 赋能量子计算的最后一公里",
           "榜题要求双向赋能，真机要求线路可执行", 2)
    x_l, w_l = MARGIN, 17.6
    x_r, w_r = 19.8, PAGE_W - 19.8 - MARGIN
    y = CONTENT_Y + 0.25
    y += 0.15
    y, _ = _block(s, x_l, y, w_l, [
        ("问题", 18, C["primary"], True)], line_gap=1.4)
    y += 0.2
    y, _ = _block(s, x_l, y, w_l, [
        ("双比特门只能作用在相邻物理比特上。布局与路由决定\n"
         "要插多少 SWAP——插得多，深度和噪声误差一起上去。", 15,
         C["text"], False)], line_gap=1.55)
    y += 0.65
    y, _ = _block(s, x_l, y, w_l, [
        ("现有工具的四个缺口", 18, C["highlight"], True)], line_gap=1.4)
    y += 0.2
    y, _ = _block(s, x_l, y, w_l, [
        ("· 依赖 qiskit / pytket，评委环境未必装得上；\n"
         "· 布局靠启发式，不从数据里学；\n"
         "· 噪声校准信息用得不够；\n"
         "· 大线路的置换吸收与 1Q 门体积仍是难点。", 15,
         C["text"], False)], line_gap=1.6)
    y += 0.65
    y, _ = _block(s, x_l, y, w_l, [
        ("我们的应答", 18, C["primary"], True)], line_gap=1.4)
    y += 0.2
    _block(s, x_l, y, w_l, [
        ("布局交给强化学习；路由、后处理、合成全部自研；\n"
         "校准噪声从布局一直带到后处理。", 15, C["text"], False)], line_gap=1.55)

    # 右侧：逻辑图 → 物理图 示意
    _rect(s, x_r, CONTENT_Y + 0.25, w_r, 13.4, C["bg_gray"])
    _block(s, x_r + 0.5, CONTENT_Y + 0.7, w_r - 1.0, "逻辑交互图 → 物理耦合图",
           size=12, color=C["text_light"], align=PP_ALIGN.CENTER)
    nodes = [(x_r + 2.0, 4.2), (x_r + 4.2, 3.6), (x_r + 6.2, 4.6),
             (x_r + 3.4, 6.6), (x_r + 5.8, 6.9), (x_r + 7.6, 6.0)]
    for i, j in ((0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (2, 5), (4, 5)):
        _arrow(s, nodes[i][0] + 0.3, nodes[i][1] + 0.3,
               nodes[j][0] + 0.3, nodes[j][1] + 0.3, color="9FB6D8", width=1.0)
    for nx, ny in nodes:
        shp = s.shapes.add_shape(MSO_SHAPE.OVAL, Cm(nx), Cm(ny), Cm(0.6), Cm(0.6))
        shp.fill.solid(); shp.fill.fore_color.rgb = _rgb(C["accent"])
        shp.line.fill.background(); shp.shadow.inherit = False
    _arrow(s, x_r + 8.6, 6.9, x_r + 10.1, 6.9, color=C["highlight"], width=2.2)
    _block(s, x_r + 8.5, 6.0, 2.0, "映射", size=11, color=C["highlight"],
           bold=True, align=PP_ALIGN.CENTER)
    # 物理网格
    gy, cell = 9.0, 0.72
    for r in range(6):
        for c in range(9):
            gx = x_r + 1.2 + c * cell
            gyy = gy + r * cell
            if r == 5 and c > 6:
                continue
            shp = s.shapes.add_shape(MSO_SHAPE.OVAL, Cm(gx), Cm(gyy),
                                     Cm(0.42), Cm(0.42))
            shp.fill.solid(); shp.fill.fore_color.rgb = _rgb("B7CBE8")
            shp.line.fill.background(); shp.shadow.inherit = False
            if c < 8:
                _arrow(s, gx + 0.44, gyy + 0.21, gx + cell - 0.22, gyy + 0.21,
                       color="C9D6EA", width=0.7)
            if r < 5:
                _arrow(s, gx + 0.21, gyy + 0.44, gx + 0.21, gyy + cell - 0.22,
                       color="C9D6EA", width=0.7)
    _block(s, x_r + 0.5, 14.6, w_r - 1.0, "天衍-287 · 105 比特 · 15×7 耦合图",
           size=11, color=C["text_light"], align=PP_ALIGN.CENTER)


def p3_pipeline(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "三阶段管线：从 QASM 到真机可执行 QCIS",
           "布局、路由、后处理全部自研；每一段都有验证门禁", 3)
    pw = 26.6
    y = _picture(s, FIG / "fig_pipeline.png", (PAGE_W - pw) / 2, CONTENT_Y + 0.05,
                 pw)
    y += 0.25
    x = MARGIN
    w = (PAGE_W - 2 * MARGIN - 1.2) / 3
    for title, body in (
        ("① 强化学习布局", "GAT 编码程序图与设备图\n多起点候选 + 噪声感知评分"),
        ("② SWIFT 路由内核", "前沿增量维护\n噪声感知候选评分与门级调度"),
        ("③ 置换重写后处理", "VW 闭式块合成\n1Q 游程合并 · 连通性感知推挤"),
    ):
        _card(s, x, y, w, 3.3, title, body, body_size=11, title_size=13.5)
        x += w + 0.6


def p4_rl(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "创新一：让强化学习学会“把比特放哪儿”",
           "稀疏终局奖励 + 指针网络解码 + 多起点候选竞技", 4)
    y = CONTENT_Y + 0.35
    x = MARGIN
    w = (PAGE_W - 2 * MARGIN - 1.8) / 3
    items = [
        ("状态", "程序交互图（GAT）\n设备拓扑 + 校准特征\n（1Q/2Q 误差、T1/T2）"),
        ("策略", "指针网络逐个放出逻辑比特\nREINFORCE 稀疏终局奖励\n自批评贪心基线"),
        ("决策", "多起点采样生成候选布局\n候选竞技：按真实路由结果评分\n优胜者进入路由"),
    ]
    for i, (t, b) in enumerate(items):
        _card(s, x, y, w, 5.6, t, b, body_size=13, title_size=16)
        if i < 2:
            _arrow(s, x + w + 0.15, y + 2.8, x + w + 0.75, y + 2.8,
                   color=C["accent"], width=2)
        x += w + 0.9
    y += 6.0
    _rect(s, MARGIN, y, PAGE_W - 2 * MARGIN, 2.9, C["bg_light"])
    _block(s, MARGIN + 0.6, y + 0.35, PAGE_W - 2 * MARGIN - 1.2, [
        ("为什么有效：布局的好坏不是看“像不像”，而是看路由之后的真实代价。",
         14.5, C["primary"], True),
        ("候选竞技把“布局—路由—后处理”作为一个整体来评分，避免布局层局部最优。",
         12.5, C["text"], False)], line_gap=1.6)


def p5_route_post(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "创新二、三：自研路由与后处理，零 qiskit 的映射内核",
           "SWIFT 增量路由 + 置换重写 + 1Q 门压缩", 5)
    x_l, w_l = MARGIN, 15.6
    x_r, w_r = 17.8, PAGE_W - 17.8 - MARGIN
    y = CONTENT_Y + 0.35
    y = _card(s, x_l, y, w_l, 5.3, "SWIFT 路由内核", (
        "· 前沿与扩展集增量维护，门级调度\n"
        "· 噪声感知候选评分：距离 + 误差 + 衰减\n"
        "· 百万门级线路的增量内核，秒级响应"), body_size=12.5, title_size=15)
    y += 0.45
    _card(s, x_l, y, w_l, 4.7, "可执行性约束", (
        "· SWAP 只在耦合边上插入\n"
        "· 禁用比特不参与映射\n"
        "· 输出门集 ⊆ {rz, sx, x, cz, measure}"), body_size=12.5, title_size=15)
    y = CONTENT_Y + 0.35
    y = _card(s, x_r, y, w_r, 5.3, "置换重写后处理", (
        "· 交换共轭规则：把置换推到线路末端\n"
        "· VW 闭式块合成：4×4 块酉矩阵一次合成\n"
        "· 1Q 游程合并：rz/sx/x 游程矩阵乘积再合成"), body_size=12.5,
        title_size=15, title_color=C["highlight"])
    y += 0.45
    _card(s, x_r, y, w_r, 4.7, "结果", (
        "· 平台基下零 swap 残留\n"
        "· 1Q 门数显著压缩，深度同步下降\n"
        "· 31/31 抽样线路通过真机合法性检查"), body_size=12.5,
        title_size=15, title_color=C["green"], fill=C["bg_gray"])


def p6_demo(prs, video: Path, poster: Path):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "现场演示：75 秒走完映射全流程",
           "真实命令行运行 + Web 交互 + 拓扑可视化（75 秒）", 6)
    vw = 25.6
    from PIL import Image
    with Image.open(poster) as im:
        w, h = im.size
    vh = vw * h / w
    left = (PAGE_W - vw) / 2
    top = CONTENT_Y + 0.1
    s.shapes.add_movie(str(video), Cm(left), Cm(top), Cm(vw), Cm(vh),
                       poster_frame_image=str(poster), mime_type="video/mp4")
    _block(s, left, top + vh + 0.3, vw,
           "点击播放 · 视频文件同时单独存放于 U 盘「演示视频.mp4」",
           size=11, color=C["text_light"], align=PP_ALIGN.CENTER)


def p7_results(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "583 条完成线路分层实测：可信、可复现",
           "587 条基准中 4 条超时，其余全部完成；原始数据逐线路落盘", 7)
    pw = 23.5
    y = _picture(s, FIG / "fig_results_ppt.png", (PAGE_W - pw) / 2, CONTENT_Y + 0.05,
                 pw)
    y += 0.2
    x = MARGIN
    w = (PAGE_W - 2 * MARGIN - 1.2) / 3
    for title, body, col in (
        ("≤10 比特", "保真度 0.735\n为感知 O3 的 0.965 倍\n胜 pytket 1.11 倍", C["primary"]),
        ("11-50 比特", "保真度 0.146\n胜 pytket 1.37 倍\n相对混合基线 1.06 倍", C["primary"]),
        ("51-105 比特", "2Q 深度 3031\n与 pytket 持平\n深度比 pytket 低 34%", C["highlight"]),
    ):
        _rect(s, x, y, w, 3.15, C["bg_light"])
        _block(s, x + 0.35, y + 0.25, w - 0.7, title, size=13,
               color=col, bold=True)
        _block(s, x + 0.35, y + 0.95, w - 0.7, body, size=11,
               color=C["text"], line_gap=1.45)
        x += w + 0.6
    _block(s, MARGIN, y + 3.3, PAGE_W - 2 * MARGIN,
           "如实说明：原生噪声感知 O3 仍是 51-105 比特层最强基线，我们的差距在 1Q 门体积与置换吸收。",
           size=11.5, color=C["text_light"])


def p8_verify(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "交付即可上真机：合法性与语义双重门禁",
           "不是“能画出来”，而是“能在天衍平台上执行”", 8)
    y = CONTENT_Y + 0.3
    tiles = [
        ("31 / 31", "抽样线路真机合法", "门集 · 相邻 · 禁用比特 · 无 swap 残留"),
        ("31 / 31", "measure 语义一致", "测量位与 final_layout 对齐"),
        ("1.000000", "端到端语义保真度", "k≤16 紧凑子空间态矢量，15 条"),
        ("232", "自动化测试通过", "组件级 + 端到端回归"),
    ]
    w = (PAGE_W - 2 * MARGIN - 3 * 0.7) / 4
    x = MARGIN
    for big, mid, sub in tiles:
        _rect(s, x, y, w, 6.2, C["bg_light"])
        _block(s, x + 0.3, y + 1.0, w - 0.6, big, size=30,
               color=C["primary"], bold=True, align=PP_ALIGN.CENTER)
        _block(s, x + 0.3, y + 3.3, w - 0.6, mid, size=14,
               color=C["text"], align=PP_ALIGN.CENTER)
        _block(s, x + 0.3, y + 4.35, w - 0.6, sub, size=9.5,
               color=C["text_light"], align=PP_ALIGN.CENTER)
        x += w + 0.7
    y += 6.9
    _rect(s, MARGIN, y, PAGE_W - 2 * MARGIN, 3.3, C["bg_gray"])
    _block(s, MARGIN + 0.6, y + 0.35, PAGE_W - 2 * MARGIN - 1.2, [
        ("天衍-287 真实设备事实", 14, C["primary"], True),
        ("105 比特 · 168 个可用耦合器 · 4 个禁用比特（21/26/34/89）；"
         "QCIS 测量指令为 M Q{x}，导出即平台原生格式。", 12.5, C["text"], False)],
        line_gap=1.6)


def p9_value(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _decor(s, "AI 赋能量子计算：从算法到交付", "三个工程承诺，一条可复现的路径", 9)
    y = CONTENT_Y + 0.5
    x = MARGIN
    w = (PAGE_W - 2 * MARGIN - 1.6) / 3
    for t, b in (
        ("映射内核零 qiskit", "CLI 交付管线全自研；\nWeb 演示层复用 qiskit 解析。"),
        ("任意拓扑可执行", "参数化设备编码；\n跨芯片零重训迁移。"),
        ("全流程可复现", "原始数据逐线路落盘；\n一键断点续跑。"),
    ):
        _card(s, x, y, w, 5.2, t, b, body_size=13.5, title_size=17)
        x += w + 0.8
    y += 5.9
    _rect(s, MARGIN, y, PAGE_W - 2 * MARGIN, 3.4, C["primary"])
    _block(s, MARGIN + 0.8, y + 0.5, PAGE_W - 2 * MARGIN - 1.6, [
        ("下一步：更大规模线路 · 更多芯片拓扑 · 真机在线闭环优化",
         15, C["white"], True),
        ("QTrail —— 让每一条量子线路都能落到真机上。",
         17, "CDE3FF", False)], line_gap=1.8)


def build(video: Path, poster: Path):
    prs = Presentation()
    prs.slide_width = Cm(PAGE_W)
    prs.slide_height = Cm(PAGE_H)
    p1_cover(prs)
    p2_problem(prs)
    p3_pipeline(prs)
    p4_rl(prs)
    p5_route_post(prs)
    p6_demo(prs, video, poster)
    p7_results(prs)
    p8_verify(prs)
    p9_value(prs)
    notes = [
        "各位评委老师好，我们的作品是 QTrail——基于稀疏奖励强化学习的自适应量子线路映射优化方案。我们要解决的问题是：让量子线路真正落到天衍-287 上跑起来。",
        "量子线路里的双比特门，只能作用在相邻的物理比特上。如果两个逻辑比特离得远，就必须插 SWAP 把它们换到一起。插得越多，线路越深，噪声误差越大。现有工具依赖 qiskit、pytket，评委环境不一定装得上；布局靠启发式，不从数据里学；校准噪声也用得不充分。我们的做法很直接：布局交给强化学习，路由、后处理、合成全部自研，噪声从布局一直带到后处理。",
        "这是我们的三阶段管线。输入是 QASM 线路和天衍-287 的拓扑校准；阶段一是强化学习布局，阶段二是自研的 SWIFT 路由内核，阶段三是置换重写后处理；最后输出平台原生的 QCIS。整条管线不依赖 qiskit，每一段都有正确性和真机合法性门禁。",
        "布局这一层，我们用 GAT 编码程序交互图和设备图，把拓扑和校准特征一起输入；指针网络逐个放出逻辑比特；用 REINFORCE 的稀疏终局奖励训练，再加自批评基线。关键的一点是：布局好不好，不看它像不像，而是看路由之后的真实代价。所以我们生成多个候选布局，让它们跑真实路由，再按结果评分选优。",
        "路由内核叫 SWIFT，核心是前沿和扩展集的增量维护，加上噪声感知的候选评分，百万门级线路也能秒级响应。后处理做三件事：交换共轭把置换推到线路末端；VW 闭式块合成把 4×4 块一次合成；1Q 游程合并把 rz、sx、x 的游程合并再合成。结果是平台基下零 swap 残留，输出门集只有 rz、sx、x、cz 和 measure。",
        "下面用一段 75 秒的视频，现场演示从 QASM 到真机可执行 QCIS 的完整过程。视频里可以看到：命令行一次映射的真实输出，Web 端的交互映射和 105 比特拓扑可视化，以及发布前的审计结果。",
        "我们在 587 条线路的基准上完成了 583 条，分三层统计。10 比特以内，保真度 0.735，是 qiskit 噪声感知 O3 的 0.965 倍，基本持平；11 到 50 比特，保真度 0.146，是 pytket 的 1.37 倍；51 到 105 比特，2Q 深度 3031，和 pytket 持平，深度比 pytket 低 34%。也要如实说明：原生噪声感知 O3 仍然是大线路最强的基线，我们的差距在 1Q 门体积和置换吸收，这是下一步要啃的硬骨头。",
        "正确性上，我们做了双重门禁：31 条抽样线路，真机合法性 31/31，门集、相邻性、禁用比特、swap 残留全部通过；measure 语义 31/31；端到端语义保真度 1.000000。另外 232 项自动化测试全部通过。天衍-287 的真实拓扑是 105 比特、168 个可用耦合器、4 个禁用比特，我们直接导出平台原生的 QCIS。",
        "最后三句话：映射内核零 qiskit，任意拓扑可执行，全流程可复现。QTrail，让每一条量子线路都能落到真机上。谢谢各位评委。",
    ]
    for slide, note in zip(prs.slides, notes):
        slide.notes_slide.notes_text_frame.text = note
    prs.save(str(OUT))
    print("written", OUT, "slides:", len(prs.slides))


if __name__ == "__main__":
    import sys
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "promo_video" / "out" / "qtrial-demo.mp4")
    poster = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        ROOT / "promo_video" / "out" / "qa_demo" / "f800.png")
    build(video, poster)
