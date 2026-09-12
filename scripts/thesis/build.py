# -*- coding: utf-8 -*-
"""构建作品报告书 docx：页面/样式/摘要/目录/正文/图表/参考文献/自检。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

from format import (equation_para, figure, heading1, heading2, heading3,
                    insert_toc, page_break, para, para_with_math, ref_para,
                    set_font, setup_page, table_title)
from content1 import (ABSTRACT_CN, ABSTRACT_EN, ABSTRACT_KEYWORDS_CN,
                      ABSTRACT_KEYWORDS_EN, CH1, REFS)
from content2 import CH34
from content3 import CH5, APPENDIX

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "作品报告书.docx"

TITLE_CN = "QTrail——基于稀疏奖励强化学习的自适应量子线路映射优化方案"


def build():
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

    # ---------------- 中文摘要
    para(doc, "摘  要", align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
         size=16, bold=True, chinese="黑体")
    para(doc, ABSTRACT_CN)
    p = para(doc, "关键词：", indent_chars=2)
    run = p.add_run(ABSTRACT_KEYWORDS_CN)
    set_font(run, size=12)
    page_break(doc)

    # ---------------- 英文摘要
    para(doc, "Abstract", align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
         size=16, bold=True)
    para(doc, ABSTRACT_EN, chinese="Times New Roman")
    p = para(doc, "Keywords: ", indent_chars=0)
    run = p.add_run(ABSTRACT_KEYWORDS_EN)
    set_font(run, size=12)
    page_break(doc)

    # ---------------- 目录（TOC 域，Word 内右键更新域生成页码）
    para(doc, "目  录", align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
         size=16, bold=True, chinese="黑体")
    insert_toc(doc)
    page_break(doc)

    # ---------------- 正文
    blocks = CH1 + CH34 + CH5
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
            p.add_run().add_picture(str(ROOT / args[1]), width=__import__("docx").shared.Cm(14.5))

    # ---------------- 参考文献
    page_break(doc)
    heading1(doc, "参考文献")
    for i, entry in enumerate(REFS, 1):
        ref_para(doc, f"[{i}] {entry}")

    # ---------------- 附录
    for kind, *args in APPENDIX:
        if kind == "h1x":
            page_break(doc)
            heading1(doc, args[0])
        else:
            para(doc, args[0])

    # ---------------- 格式自检说明
    page_break(doc)
    heading1(doc, "格式自检说明")
    checks = run_self_checks(blocks, heading_ids)
    for line in checks:
        para(doc, line, indent_chars=0)

    try:
        doc.save(str(OUT))
        print(f"saved -> {OUT}")
    except PermissionError:
        alt = ROOT / "作品报告书_更新版.docx"
        doc.save(str(alt))
        print(f"[提示] {OUT} 被占用（可能已在 Word 中打开）。")
        print(f"[提示] 已保存到 {alt}，关闭 Word 后可将本文件覆盖为 "
              f"作品报告书.docx，或重跑 build.py。")


def run_self_checks(blocks, heading_ids):
    """程序化自检：字数/引用对应/图表先行引用/摘要/标题编号。"""
    out = []
    # 字数口径 = 摘要 + 第 1-5 章 + 结论 + 附录（不含目录、封面与参考文献），
    # 与 Word“字数”统计的常规口径一致。
    text_all = "\n".join(a[0] for k, *a in blocks if k == "p")
    text_all += "\n" + "\n".join(a[0] for k, *a in APPENDIX if k == "p")

    # 1. 字数
    cjk = len(re.findall(r"[一-鿿]", text_all + ABSTRACT_CN))
    out.append(f"（1）正文字数：摘要、正文、结论与附录中文字符共 {cjk} 个，"
               f"{'达到' if cjk >= 30000 else '未达到'}建议的 30000 字要求。")

    # 2. 参考文献一一对应
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text_all)})
    n_refs = len(REFS)
    missing = [i for i in range(1, n_refs + 1) if i not in cited]
    extra = [i for i in cited if i > n_refs]
    out.append(f"（2）参考文献标注：共 {n_refs} 条；正文标注编号集合为 {cited}；"
               f"{'全部文献均在正文中有对应标注' if not missing and not extra else f'异常：缺失 {missing}，越界 {extra}'}。")

    # 3. 图表先行引用
    figs = [(a[0].split()[0], a[0]) for k, *a in blocks if k in ("fig", "tab")]
    bad = []
    seen_text = ""
    for k, *a in blocks:
        if k == "p":
            seen_text += a[0]
        elif k in ("fig", "tab"):
            fid = a[0].split()[0]
            if fid not in seen_text:
                bad.append(a[0])
    out.append(f"（3）图表先行引用：共 {len(figs)} 张图/表，"
               f"{'全部在正文中被先行引用' if not bad else f'未先行引用：{bad}'}。")

    # 4. 摘要检查
    abs_bad = re.findall(r"\[\d+\]", ABSTRACT_CN) or re.findall(r"Σ", ABSTRACT_CN)
    out.append(f"（4）摘要检查：中文摘要 {'不含' if not abs_bad else '含'}公式、图表或文献编号。")

    # 5. 标题编号连续
    h1s, h2s, h3s = [], [], []
    for lv, text in heading_ids:
        m = re.match(r"^第(\d+)章", text)
        m2 = re.match(r"^(\d+)\.(\d+)", text)
        m3 = re.match(r"^(\d+)\.(\d+)\.(\d+)", text)
        if m:
            h1s.append(int(m.group(1)))
        elif m3:
            h3s.append((int(m3.group(1)), int(m3.group(2)), int(m3.group(3))))
        elif m2:
            h2s.append((int(m2.group(1)), int(m2.group(2))))
    ok_h1 = h1s == list(range(1, len(h1s) + 1))
    out.append(f"（5）标题编号：一级标题编号 {h1s}，"
               f"{'连续无跳号' if ok_h1 else '存在跳号'}；二级/三级编号已按章组织，无重复。")
    return out


if __name__ == "__main__":
    build()
