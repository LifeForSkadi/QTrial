# -*- coding: utf-8 -*-
"""生成逐页讲稿 doc + 配图制作清单 doc（与 PPT/报告书编号一致）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.shared import Cm, Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from format import set_font

ROOT = Path(__file__).resolve().parent.parent.parent

# 每页：(页号, 标题, 建议秒数, 讲稿正文)
PAGES = [
    (1, "问题、方案与价值", 45,
     "各位老师好。我们的问题很直接：量子线路要上芯片，布局是 NP 难问题，路由还得插 SWAP 门。"
     "我们的做法是：**强化学习布局加自研 SABRE 路由，再加置换重写后处理**，全链路不依赖 "
     "qiskit。价值一句话：在天衍-287 上真机可执行，587 条线路评测里，保真度超过了 pytket "
     "和盲目基线。"),
    (2, "系统总体架构", 50,
     "刚才说的是问题与价值，接下来看整体结构。整个方案是五段管线：解析、图构建、布局、路由、"
     "后处理。程序图和设备规格两条输入，分别带 6 维和 7 维特征。右边这张架构图请大家看两个"
     "地方：一个是贯穿全链路的噪声感知机制，另一个是底部的三层真机验证链。另外，任何输入"
     "都不会崩，因为我们有失败阶梯兜底。"),
    (3, "布局：QAP 形式化与强化学习策略", 50,
     "先看最核心的布局环节。布局本质是二次分配问题，NP 难。式(2-1) 的物理含义很简单：把所有"
     "交互对的次数，乘上它们被分开的距离，总和越小，布局越好。我们用 GAT 加指针网络来学这个"
     "问题的构造策略，训练信号只有一个，就是线路完整排好之后的一次性奖励，也就是式(4-2)。"
     "推理的时候十起点解码加局部搜索，再叠上失败阶梯。"),
    (4, "路由：自研 SABRE", 50,
     "布局定好了，怎么把不相邻的门挪到一起，就是路由。我们完整复现了 SABRE 的机制：前沿层"
     "并行执行，扩展集前瞻。式(4-1) 是它的交换评分，三项分别管距离代价、前瞻和历史惩罚。"
     "工程上做了增量维护和 Numba 内核，**百比特线路秒级完成**，而且用逐位 A/B 验证保证加速"
     "不改变行为。"),
    (5, "后处理栈与平台基分解", 45,
     "路由完的线路里全是 SWAP，后处理栈做四层消化：推挤、消解、尾部吸收、平台基分解。铁律是"
     "每一步都保语义，有态矢量测试守着；红线是重标记后比特对必须还在耦合图上，这来自一次"
     "真机被拒的事故。式(4-3) 是 U3 分解模板，误差 2.5e-16，最终落到平台基。"),
    (6, "噪声感知与跨设备零重训", 45,
     "接下来是噪声感知。式(2-2) 的意思是：把噪声距离当作拓扑距离的扰动，揉进奖励和评分。"
     "关键的工程细节是设备特征按自己的统计量归一化，所以网络学的是相对质量差异，换芯片"
     "不用重训。图5.7 就是证据：同一套权重，三种拓扑上我们的 2Q 误差都显著低于盲目 O1。"),
    (7, "训练过程与模型侧改进", 50,
     "训练层面有两项改进。LAUREL 可学习残差把收敛轮数从 400 轮压到 89 轮，**单次训练约"
     "五分钟**；效果不降反升，保真度加 3.1%，30 条里 21 条更优，就是图5.2。另一个是 QuEst "
     "风格的预测器，把 T1/T2 退相干算进保真度估计，补上乘积模型的盲区。"),
    (8, "数据集与评测体系", 40,
     "评测体系一句话：用 Qiskit 官方测评集 Benchpress 打主战场，加 MQTBench 和 QUEKO，一共 "
     "587 条，分三层、三种统计口径，原始数据逐条落盘。另外我们还自建了 2 万条 287 拓扑原生"
     "样本，每一条都带已知最优映射，可以当布局质量的上界参考。"),
    (9, "分层性能", 50,
     "先看分层性能。小线路层，我们 0.619，**显著高于盲目 O1 的 0.454**；中等层 0.129，超过 "
     "pytket 和全部盲目基线，图5.3 的柱子就是各层的保真度对比。大线路层目前是短板，0.0089 "
     "对 0.0209，差距的根因我们定位清楚了，稍后在第 13 页讲。"),
    (10, "深度与运行时间", 40,
     "深度和时间一起看。深度的差距来自残余 SWAP 展开成三个 CNOT，大层大约 8.7 倍。时间上，"
     "单条线路 2.5 到 27.7 秒，满足分钟级目标；图5.5 是对比曲线，平均慢 51 倍是对方多年工程"
     "积累的差距，如实呈现。"),
    (11, "噪声感知证据", 40,
     "噪声感知到底有没有用？看这个缺陷扫描：把 2% 比特的误差放大五倍，盲目 O1 的 2Q 误差从 "
     "0.0035 恶化到 0.0110，**我们稳定在 0.003 左右**，和感知 O3 接近。图5.6 四条线就是四个"
     "方法。这说明缺陷越严重，噪声感知的价值越明显。"),
    (12, "与成熟方案对比", 45,
     "与成熟方案总对比：587 条线路，总体保真度 0.196，超过 pytket 的 0.187 和全部盲目基线；"
     "胜场是**对感知 O3 二十九场、对 pytket 一百三十四场、对盲目 O1 二百二十六场**。是感知 "
     "O3 的 0.80 倍，差距集中在大线路层，我们不回避。"),
    (13, "布局贡献与结论展望", 45,
     "最后一个实验回答：优势是不是布局带来的？把路由器都换成 qiskit SABRE，只比布局——"
     "我们保真度三个基准全面第一，Benchpress 上 0.068 对 O3 的 0.003。结论是：**噪声感知"
     "布局这条路线成立**；下一步做门级再合成和学习化路由，补齐大线路深度的短板。谢谢。"),
]

# 配图制作清单：(位置, 图号, 建议类型, 数据来源, 制作工具建议)
FIGURE_LIST = [
    ("第 2 页", "图3.1", "系统架构图（五段管线流程图）",
     "方案设计（报告书 3.2 节）", "PowerPoint 形状/Visio；已随 PPT 嵌入 PNG"),
    ("第 3 页", "式(2-1)(4-2)", "OMML 原生公式（PPT 内置公式编辑器）",
     "报告书 2.1/4.3.2 节", "PPT 公式编辑器；已按 OMML 嵌入，可直接双击编辑"),
    ("第 4 页", "式(4-1)", "OMML 原生公式", "报告书 4.4 节", "同上"),
    ("第 5 页", "式(4-3)", "OMML 原生公式", "报告书 4.5 节", "同上"),
    ("第 6 页", "图5.7", "分组柱状图（跨拓扑 2Q 误差对比）",
     "tables/cross_topo/*.jsonl", "matplotlib（scripts/make_report_figures.py）"),
    ("第 6 页", "式(2-2)", "OMML 原生公式", "报告书 2.5 节", "PPT 公式编辑器"),
    ("第 7 页", "图5.2", "双面板柱状图（训练轮数 + 保真度对比）",
     "tables/compare_models/*.jsonl", "matplotlib"),
    ("第 8 页", "表5.1 节选", "表格（数据集构成）",
     "报告书 3.4 节/附录 D", "PPT 表格；已随 PPT 嵌入"),
    ("第 9 页", "图5.3", "分组柱状图（分层保真度对比，含基线）",
     "tables/final_report/report.md（v4 定稿表）", "matplotlib"),
    ("第 10 页", "图5.5", "折线对比图（运行时间，对数刻度）",
     "tables/speed_large_vs_o3.jsonl", "matplotlib"),
    ("第 11 页", "图5.6", "折线对比图（缺陷敏感性扫描）",
     "tables/defect_sweep/f*.jsonl", "matplotlib"),
    ("第 12 页", "图5.8", "双面板对比图（总体保真度 + 胜场数）",
     "tables/final_report/report.md", "matplotlib"),
    ("第 13 页", "图5.10", "3×3 网格对比图（布局贡献，三基准 × 三指标）",
     "tables/compare_models/layout_contribution_*.jsonl", "matplotlib"),
]


def _para(doc, text, size=12, bold=False, indent=0, space_after=6,
          color=None):
    from docx.shared import RGBColor
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(space_after)
    if indent:
        pf.left_indent = Cm(indent)
    # 解析 **加粗** 标记
    parts = text.split("**")
    for i, seg in enumerate(parts):
        if not seg:
            continue
        run = p.add_run(seg)
        set_font(run, size=size, bold=bold or (i % 2 == 1))
        if color is not None:
            run.font.color.rgb = RGBColor(*color)
    return p


def build_script():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.5)
    sec.top_margin = sec.bottom_margin = Cm(2.5)

    _para(doc, "方案展示讲稿（逐页）", size=18, bold=True, space_after=4)
    _para(doc, "配套文件：方案展示.pptx ｜ 作品报告书.docx ｜ 配图制作清单.docx",
          size=11, space_after=2, color=(0x89, 0x87, 0x81))
    _para(doc, "总时长约 9 分 55 秒（13 页正文，含过渡与致谢），预留 5 秒余量。"
               "正文加粗处建议重读强调。", size=11, space_after=16,
          color=(0x89, 0x87, 0x81))

    total = 0
    for idx, title, secs, body in PAGES:
        total += secs
        mm, ss = divmod(total, 60)
        _para(doc, f"【第{idx}页｜建议{secs}秒】　{title}", size=13, bold=True,
              space_after=2, color=(0x2A, 0x78, 0xD6))
        _para(doc, body, size=12, space_after=2)
        _para(doc, f"（至此约{mm}分{ss:02d}秒）", size=10, space_after=14,
              color=(0x89, 0x87, 0x81))

    # 时长自检表
    _para(doc, "时长自检表", size=16, bold=True, space_after=6)
    tbl = doc.add_table(rows=1 + len(PAGES), cols=4)
    tbl.style = "Table Grid"
    headers = ["页码", "主题", "建议秒数", "累计"]
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                set_font(r, size=10.5, bold=True)
    acc = 0
    for i, (idx, title, secs, _) in enumerate(PAGES, 1):
        acc += secs
        row = tbl.cell(i, 0), tbl.cell(i, 1), tbl.cell(i, 2), tbl.cell(i, 3)
        vals = [str(idx), title, str(secs), f"{acc//60}分{acc%60:02d}秒"]
        for cell, v in zip(row, vals):
            cell.text = v
            for p in cell.paragraphs:
                for r in p.runs:
                    set_font(r, size=10.5)
    _para(doc, f"合计 {acc} 秒 = {acc//60} 分 {acc%60} 秒 ≤ 10 分钟，达标。",
          size=11, bold=True, space_after=2)

    out = ROOT / "讲稿.docx"
    doc.save(str(out))
    print(f"saved -> {out}")


def build_figure_list():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.5)

    _para(doc, "配图制作清单", size=18, bold=True, space_after=4)
    _para(doc, "配套文件：方案展示.pptx ｜ 讲稿.docx ｜ 作品报告书.docx。"
               "图号与报告书正文完全一致；数据图表均为对比性可视化（含基线方法对照）。",
          size=11, space_after=12, color=(0x89, 0x87, 0x81))

    tbl = doc.add_table(rows=1 + len(FIGURE_LIST), cols=5)
    tbl.style = "Table Grid"
    for j, h in enumerate(["位置", "图号/编号", "建议类型", "数据来源", "制作工具建议"]):
        cell = tbl.cell(0, j)
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                set_font(r, size=10.5, bold=True)
    for i, row_data in enumerate(FIGURE_LIST, 1):
        for j, v in enumerate(row_data):
            cell = tbl.cell(i, j)
            cell.text = v
            for p in cell.paragraphs:
                for r in p.runs:
                    set_font(r, size=10)
    # 列宽
    widths = [Cm(1.8), Cm(2.2), Cm(4.6), Cm(4.6), Cm(4.4)]
    for j, w in enumerate(widths):
        for row in tbl.rows:
            row.cells[j].width = w

    _para(doc, "说明：全部图表 PNG 已随仓库提供（tables/report_figures/），"
               "运行 python scripts/make_report_figures.py 可一键重新生成。"
               "当前 PPT 采用整页图片版式（由方案展示.pdf 渲染，保证任意环境可打开），"
               "公式以 mathtext 渲染并标注式编号（与报告书一致），如需编辑文本与公式，"
               "请使用作品报告书.docx 或方案展示.pdf 源文件。", size=10.5,
          space_after=2, color=(0x89, 0x87, 0x81))

    out = ROOT / "配图制作清单.docx"
    doc.save(str(out))
    print(f"saved -> {out}")


if __name__ == "__main__":
    build_script()
    build_figure_list()
