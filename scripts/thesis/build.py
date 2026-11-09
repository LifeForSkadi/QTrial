# -*- coding: utf-8 -*-
"""构建作品报告书 docx（读 JSON 内容文件）：页面/样式/摘要/目录/正文/图表/
参考文献/自检。内容文件由 restructure.py 生成。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from format import (cjk_compact, equation_para, figure, heading1, heading2, heading3,
                    insert_toc, page_break, para, para_with_math, ref_para,
                    set_font, setup_page, table_title, three_line_table)

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "作品报告书.docx"
HERE = Path(__file__).resolve().parent

TITLE_CN = "QTrail——基于稀疏奖励强化学习的自适应量子线路映射优化方案"


def load(name):
    with open(HERE / name, encoding="utf-8") as f:
        return json.load(f)


def render_blocks(doc, blocks):
    heading_ids = []
    for kind, *args in blocks:
        if kind == "h1":
            heading1(doc, args[0], page_break=False)
            heading_ids.append(("h1", args[0]))
        elif kind == "h1x":
            page_break(doc)
            heading1(doc, args[0])
        elif kind == "h2":
            heading2(doc, args[0])
            heading_ids.append(("h2", args[0]))
        elif kind == "h3":
            heading3(doc, args[0])
            heading_ids.append(("h3", args[0]))
        elif kind == "p":
            para_with_math(doc, args[0])
        elif kind == "eq":
            equation_para(doc, args[0])
        elif kind == "fig":
            figure(doc, str(ROOT / args[1]), args[0], args[2])
        elif kind == "tab":
            table_title(doc, args[0])
            p = doc.add_paragraph()
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(ROOT / args[1]), width=Cm(14.5))
        elif kind == "tbl3":
            # Word 原生三线表：("tbl3", 表题, 表头, 数据行, 表注, 对齐)
            caption, header, rows = args[0], args[1], args[2]
            note = args[3] if len(args) > 3 else None
            aligns = args[4] if len(args) > 4 else None
            three_line_table(doc, caption, header, rows, note=note,
                             aligns=aligns)
    return heading_ids


def build():
    d1 = load("content1.json")
    d2 = load("content2.json")
    d3 = load("content3.json")

    doc = Document()
    setup_page(doc)

    # ---------------- 封面
    for _ in range(5):
        para(doc, "", indent_chars=0)
    para(doc, "作品报告书", align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
         size=26, bold=True, chinese="黑体")
    para(doc, "", indent_chars=0)
    para(doc, TITLE_CN, align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
         size=16, bold=True, chinese="黑体")
    para(doc, "", indent_chars=0)
    for line in ("2026 年度中国青年科技创新“揭榜挂帅”擂台赛（学生赛道）",
                 "量子+AI 双向赋能的研究与应用探索 榜题",
                 "（单位：　　　　　姓名：　　　　　手机号：　　　　　）"):
        para(doc, line, align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
             size=14, chinese="宋体")
    page_break(doc)

    # ---------------- 中文摘要（四自然段）
    para(doc, "摘  要", align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
         size=16, bold=True, chinese="黑体")
    for seg in d1["abstract_cn"].split("\n"):
        para(doc, seg)
    p = para(doc, "关键词：", indent_chars=2)
    run = p.add_run(cjk_compact(d1["kw_cn"]))
    set_font(run, size=12)
    page_break(doc)

    # ---------------- 英文摘要（四自然段对应）
    para(doc, "Abstract", align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
         size=16, bold=True)
    for seg in d1["abstract_en"].split("\n"):
        para(doc, seg, chinese="Times New Roman")
    p = para(doc, "Keywords: ", indent_chars=0)
    run = p.add_run(d1["kw_en"])
    set_font(run, size=12)
    page_break(doc)

    # ---------------- 目录
    para(doc, "目  录", align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
         size=16, bold=True, chinese="黑体")
    insert_toc(doc)
    page_break(doc)

    # ---------------- 正文
    blocks = (d1["ch1"] + d1["ch2"] + d2["ch3"] + d2["ch4"] + d3["ch5"]
              + d3["conclusion"])
    heading_ids = render_blocks(doc, blocks)

    # ---------------- 参考文献
    page_break(doc)
    heading1(doc, "参考文献")
    for i, entry in enumerate(d1["refs"], 1):
        ref_para(doc, f"[{i}] {entry}")

    # ---------------- 附录
    render_blocks(doc, d3["appendix"])

    # ---------------- 格式自检说明
    page_break(doc)
    heading1(doc, "格式自检说明")
    for line in run_self_checks(blocks, heading_ids, d3):
        para(doc, line, indent_chars=0)

    # ---------------- 修改自检说明
    page_break(doc)
    heading1(doc, "修改自检说明")
    for line in REVISION_CHECK:
        para(doc, line, indent_chars=0)
    _revision_table(doc)

    try:
        doc.save(str(OUT))
        print(f"saved -> {OUT}")
    except PermissionError:
        alt = ROOT / "作品报告书_更新版.docx"
        doc.save(str(alt))
        print(f"[提示] {OUT} 被占用，已保存到 {alt}；关闭 Word 后重跑 build.py 覆盖。")


def run_self_checks(blocks, heading_ids, d3):
    out = []
    text_all = "\n".join(a[0] for k, *a in blocks if k == "p")
    text_all += "\n" + "\n".join(a[0] for k, *a in d3["appendix"] if k == "p")
    abstract = d1_abstract_cn()
    cjk = len(re.findall(r"[一-鿿]", text_all + abstract))
    out.append(f"（1）正文字数：摘要、正文、结论与附录中文字符共 {cjk} 个，"
               f"{'达到' if cjk >= 30000 else '未达到'}建议的 30000 字要求。")
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text_all)})
    n_refs = 11
    missing = [i for i in range(1, n_refs + 1) if i not in cited]
    extra = [i for i in cited if i > n_refs]
    out.append(f"（2）参考文献标注：共 {n_refs} 条；正文标注编号集合为 {cited}；"
               f"{'全部文献均在正文中有对应标注' if not missing and not extra else f'异常：缺失 {missing}，越界 {extra}'}。")
    figs = [(a[0].split()[0], a[0]) for k, *a in blocks
            if k in ("fig", "tab", "tbl3")]
    bad = []
    seen = ""
    for k, *a in blocks:
        if k == "p":
            seen += a[0]
        elif k in ("fig", "tab", "tbl3"):
            fid = a[0].split()[0]
            if fid not in seen:
                bad.append(a[0])
    out.append(f"（3）图表先行引用：共 {len(figs)} 张图/表，"
               f"{'全部在正文中被先行引用' if not bad else f'未先行引用：{bad}'}。")
    abs_bad = re.findall(r"\[\d+\]", abstract) or re.findall(r"Σ", abstract)
    out.append(f"（4）摘要检查：中文摘要 {'不含' if not abs_bad else '含'}公式、图表或文献编号。")
    h1s = [int(re.match(r"^第(\d+)章", t).group(1)) for lv, t in heading_ids
           if lv == "h1" and re.match(r"^第(\d+)章", t)]
    ok = h1s == list(range(1, len(h1s) + 1))
    out.append(f"（5）标题编号：一级标题编号 {h1s}，"
               f"{'连续无跳号' if ok else '存在跳号'}。")
    return out


def d1_abstract_cn():
    d1 = load("content1.json")
    return d1["abstract_cn"]


# ---------------------------------------------------------------- 修改自检说明
REVISION_CHECK = [
    "本次修改对第 5 章做独立可读化与去重：删除与第 2、3、4 章重复的内容与第 5 章内部"
    "重复的描述，保留必要的数据集/实验设置/评测指标；表格按内容归属重排——训练设置表与"
    "运行时间评测设置表删除，新增测评基准汇总表与硬件环境表（处置细节见第 10 条）。修改"
    "仅涉及第 5 章与附录 B，第 1、2、3、4 章、摘要、结论与参考文献未改动。逐条核对如下。",
    "（1）删除的与前文重复的内容：5.1 中“各模块接口的契约”段（模块接口内容，见第 3 章）；"
    "5.3 中“验证协议的标准步骤”段（三层验证链协议见 4.3.2 节）与真机验证过程描述中的"
    "协议步骤；5.4.2 深度分析中的后处理机制展开（改为引用 4.3 节）；5.4.3 中奖励塑形与"
    "设备特征构造的机制展开（改为引用 4.1.7 节）；5.4.6 中“51-105 层深度约 8.7 倍”具体"
    "数值（改为“约一个数量级”并指向 5.4.2 节，归纳不再罗列数值）。",
    "（2）删除的第 5 章内部重复：5.2 实验设置与 5.1 重复的数据集列举与统计口径描述（改为"
    "“见 5.1 节”）；5.2 中两段硬件环境描述合并为一段。",
    "（3）保留的必要内容：数据集与来源（5.1：三基准 + 287 原生池 + 运行时间测试集 90 条，"
    "含划分与生成方式）；实验设置（5.2：硬件/软件/种子/基线定义/公平性措施，环境清单与"
    "基线设置两段、评测流程一段）；评测指标（分层统计口径在 5.1；端到端运行时间与优化前"
    "SWAP 门数定义在 5.1）；真机验证结果与局限（5.3）。",
    "（4）各板块图表（无跨板块重复使用）：5.1 数据集与评测体系——表5.1；5.2 实验设置——"
    "表5.2；5.4.1 训练过程——图5.1；5.4.2 分层性能——图5.2、图5.3；5.4.3 噪声感知有效性"
    "证据——图5.4、图5.5、图5.6；5.4.4 运行时间——图5.7、图5.8；5.4.5 与成熟方案的综合"
    "对比——图5.9、图5.10；5.4.6 结果归纳与讨论——无图。每张图/表仅在一个板块出现一次。",
    "（5）引用前文的方式：统一使用“见第 X 章”“见 X.X 节”“机制详见 X.X 节”“协议见 "
    "4.3.2 节”等指向性短句；不重新推导、不重新解释符号。",
    "（6）特殊现象分析：5.4.4 的“低难度线路编译反而更慢”——已验证部分（105 比特低难度"
    "档 8-17 秒 vs 高难度档 0.04-0.1 秒，五个种子稳定）与推测性解释（线性函数再合成成本，"
    "文中明确标注“推测”）分别呈现；5.4.6 的“盲目 O3 保真度 0.003”矛盾解释以实测放置"
    "质量差异为依据。",
    "（7）不占优结果如实呈现：51-105 层保真度低于噪声感知 O3、大线路层深度差距、自研纯"
    "管线部分线路慢于 qiskit 编译栈、布局贡献实验中感知 O3 全面领先，均如实保留并分析原因，"
    "未使用“显著优于”等无数据支撑的形容词。",
    "（8）文献：未新增、未删除、未重新编号——去重未改变任何文献的首次出现顺序"
    "（[1]-[8] 第 1 章，[9]、[10] 第 2 章，[11] 第 4 章）。",
    "（9）表格：第 5 章现含表5.1 测评基准汇总（5.1 节）与表5.2 实验硬件与软件环境（5.2 节），"
    "均为 Word 原生三线表（顶线、底线粗线，表头线细线，无竖线；表题在上、表注在下）。表编号 "
    "5.1-5.2 连续，图编号 5.1-5.10 连续，图题置于图下方居中、表题置于表上方居中。",
    "（10）表格的删除与新增：其一，训练设置表（原表5.1）与 4.1.3 节（网络结构）、4.1.4 节"
    "（训练算法）、4.1.8 节（LAUREL 残差）重复，删除并改为引用式衔接，原“表5.1中值得解读的"
    "三处设置”段随表删除；其二，运行时间评测设置表（原表5.2）中测试集构成、被测方案、计时口径"
    "与运行环境已见于 5.1/5.2 节正文，计时口径并入图5.7/图5.8 图题注释，时间预算（600 秒红线）"
    "等图中无体现的内容不再提出；其三，本轮新增表5.1 测评基准汇总（5.1 节五个数据来源的规模、"
    "构成与用途）与表5.2 实验硬件与软件环境（5.2 节硬件、软件与依赖清单），正文介绍文字保留、"
    "表格作速查汇总；统计概貌段由 5.4.1 移入 5.1 节，与表5.1 同板块呈现。",
]

REVISION_TABLE = [
    ["原编号", "新编号", "变更类型"],
    ["图5.1 训练学习曲线", "图5.1", "不变"],
    ["图5.2 LAUREL 收敛效率", "（删除，改文字表述）", "删除"],
    ["图5.3 分层保真度", "图5.2", "顺延"],
    ["图5.4 分层深度", "图5.3", "顺延"],
    ["图5.5 运行时间（18 条）", "（删除）", "删除"],
    ["图5.6 缺陷敏感性扫描", "图5.4", "顺延"],
    ["图5.7 跨拓扑噪声规避", "图5.5", "顺延"],
    ["（无）", "图5.7 运行时间（按比特数，90 条）", "新增"],
    ["（无）", "图5.8 运行时间（按优化前 SWAP，90 条）", "新增"],
    ["图5.8 总体对比", "图5.9", "顺延"],
    ["图5.9 LAUREL 逐线路对比", "图5.10", "顺延"],
    ["图5.10 布局贡献对比", "图5.6（移入 5.4.3）", "顺延+移动"],
    ["表5.1 训练设置（Word 三线表）", "（删除；编号另用于测评基准汇总表）", "删除"],
    ["表5.2 运行时间评测设置（三线表）", "（删除；编号另用于硬件环境表）", "删除"],
    ["5.4.1 “表5.1中值得解读的三处设置”段", "（删除）", "随表删除"],
    ["（无）", "表5.1 测评基准汇总（三线表）", "新增"],
    ["（无）", "表5.2 实验硬件与软件环境（三线表）", "新增"],
    ["统计概貌段（原在 5.4.1 训练过程）", "5.1 数据集与评测体系", "移动"],
    ["5.4.3 运行时间 / 5.4.4 噪声感知证据", "5.4.3 噪声感知证据 / 5.4.4 运行时间", "对调"],
    ["布局贡献实验（5.4.5）", "5.4.3 噪声感知的有效性证据", "移动"],
    ["5.1 接口契约段 / 5.3 协议步骤段 / 5.2 重复环境段等", "（删除，改引用前文）", "删除"],
]


def _revision_table(doc):
    tbl = doc.add_table(rows=1, cols=3)
    tbl.style = "Table Grid"
    for j, h in enumerate(REVISION_TABLE[0]):
        cell = tbl.cell(0, j)
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                set_font(r, size=10, bold=True)
    for i, row in enumerate(REVISION_TABLE[1:], 1):
        cells = tbl.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = v
            for p in cells[j].paragraphs:
                for r in p.runs:
                    set_font(r, size=9.5)


if __name__ == "__main__":
    build()
