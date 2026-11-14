# -*- coding: utf-8 -*-
"""组装决赛提交包：决赛提交/（源码 + 报告书 + PPT + 视频 + 讲稿）。"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "决赛提交"
SRC = PKG / "01_源码" / "QTrial"

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.bak*", "*.log",
                                ".pytest_cache", ".ipynb_checkpoints")


def copytree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=IGNORE, dirs_exist_ok=False)
    print("copied", src, "->", dst)


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print("copied", src.name)


def main():
    PKG.mkdir(exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)

    # 源码（交付管线 + 配置 + 示例 + 测试）
    for name in ("qtrail", "configs", "examples", "tests"):
        copytree(ROOT / name, SRC / name)
    (SRC / "checkpoints").mkdir(exist_ok=True)
    for ck in (
        "tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt",
        "multi7topo_gat_combined_calib_dep0.1_t0.5_best.pt",
    ):
        copy_file(ROOT / "checkpoints" / ck, SRC / "checkpoints" / ck)

    # 关键脚本与报告构建链
    (SRC / "scripts").mkdir(exist_ok=True)
    for s in (
        "setup.bat", "setup.sh",
        "final_report_bench_v8.py", "aggregate_v8.py", "audit_v8_release.py",
        "verify_all_legality.py", "verify_e2e_production.py",
        "submission_aggregate.py", "submission_figures.py",
        "submission_overall_figure.py",
    ):
        copy_file(ROOT / "scripts" / s, SRC / "scripts" / s)
    copytree(ROOT / "scripts" / "thesis", SRC / "scripts" / "thesis")

    # 文档
    (SRC / "docs").mkdir(exist_ok=True)
    for d in ("technical_report.md", "competition_report.md",
              "complexity_analysis.md", "correctness_theorems.md",
              "negative_results_ledger.md", "vf2_postlayout_explained.md",
              "post_pipeline_correctness.md"):
        copy_file(ROOT / "docs" / d, SRC / "docs" / d)

    # 数据与图件
    copytree(ROOT / "tables" / "submission", SRC / "tables" / "submission")
    # 复现材料（审稿整改 B-M1/B-M2）：基线逐线路行、机制实验数据与图件
    (SRC / "tables" / "final_report").mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "tables" / "final_report").glob("*.jsonl"):
        copy_file(f, SRC / "tables" / "final_report" / f.name)
    for d in ("defect_sweep", "cross_topo", "compare_models", "report_figures"):
        copytree(ROOT / "tables" / d, SRC / "tables" / d)
    copytree(ROOT / "data" / "scaling", SRC / "data" / "scaling")
    # LAUREL 89 轮权重（报告 5.4.3 引用）
    laurel = ROOT / "delivery" / "checkpoints" / "demo_laurel"
    if laurel.exists():
        for f in laurel.glob("*.pt"):
            copy_file(f, SRC / "checkpoints" / "demo_laurel" / f.name)
    # 复现脚本
    for s in ("make_report_figures.py", "make_runtime_compare.py",
              "defect_sweep.py", "cross_topo_bench.py",
              "bench_layout_contribution.py", "bench_layout_inject.py",
              "bench_trivial_swaps.py", "calibrate_scaling_set.py"):
        copy_file(ROOT / "scripts" / s, SRC / "scripts" / s)

    # 根文件
    for f in ("README.md", "requirements.txt", "pyproject.toml"):
        copy_file(ROOT / f, SRC / f)

    # 报告书 / PPT / 视频 / 讲稿
    copy_file(ROOT / "作品报告书.docx", PKG / "02_报告书" / "作品报告书_决赛.docx")
    copy_file(ROOT / "作品报告书_决赛.pdf", PKG / "02_报告书" / "作品报告书_决赛.pdf")
    copy_file(ROOT / "作品展示PPT_决赛.pptx", PKG / "03_PPT" / "作品展示PPT_决赛.pptx")
    copy_file(ROOT / "promo_video" / "out" / "qtrial-demo.mp4",
              PKG / "04_演示视频" / "演示视频.mp4")
    copy_file(ROOT / "讲稿_6分钟.md", PKG / "05_讲稿" / "讲稿_6分钟.md")

    print("package ready:", PKG)


if __name__ == "__main__":
    main()
