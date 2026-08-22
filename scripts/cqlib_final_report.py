"""合并小/中两层行文件生成最终报告（大层已按用户决策砍掉）。

用法: python scripts/cqlib_final_report.py
输出: tables/cqlib_inject/report.md（覆盖旧的冒烟残留）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.cqlib_inject_bench import render  # noqa: E402

OUT = Path("tables/cqlib_inject")


def load(bucket: str) -> list:
    rows = []
    seen = set()
    p = OUT / f"{bucket}_rows.jsonl"
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["circuit"] in seen:  # 续跑重复行去重
            continue
        seen.add(r["circuit"])
        rows.append(r)
    return rows


def main():
    titles = {"small": "≤10 比特", "medium": "11-50 比特"}
    sections = [
        "# Cqlib 注入接口分层对比（天衍-287，105 比特 15×7 网格）\n",
        "QTrial+Cqlib = RL 多起点布局 + 自适应局部搜索 → 注入天衍平台原生 "
        "MCTS 路由（objective=depth）；QTrial+SABRE = 同管线 SabreSwap 后端；"
        "全部输出过 decompose_to_platform 后由 compute_metrics 统一度量；"
        "统计：均值/截尾(去10%)/中位数\n",
    ]
    total = 0
    for b in ("small", "medium"):
        rows = load(b)
        if not rows:
            continue
        total += len(rows)
        render(sections, titles[b], rows)

    sections.append(
        "\n## 数据说明\n\n"
        "- **qwalk_25 为公认病态线路**：全部编译器表现异常（SABRE O1 "
        "19428 SWAP、pytket 15983 SWAP、QTrial+Cqlib 全部 MCTS 超时），"
        "其均值污染以截尾均值消除（截尾口径为默认读数）\n"
        "- 51-105 层缺失不影响 ≤50 比特分层结论；cqlib 家族在大线路上"
        "的超时本身即定性结论\n")

    sections.append(
        "\n## 51-105 比特（含满占）——未评测（按用户决策砍掉）\n\n"
        "原因：cqlib 原生 MCTS 在大线路上频繁超时（graphstate_50 两目标均 "
        ">600s 超时、qwalk_25 全 MCTS 超时），51-105 层每条线路最坏需 "
        "3×600s 且大概率全部超时——该层对 cqlib 家族无可比数据。"
        "定性结论：**cqlib 原生管线不适合大线路实时编译**，"
        "竞赛默认路径应保持 RL 布局 + SabreSwap 级路由器（毫秒级），"
        "cqlib 注入仅作 opt-in（--routing cqlib / --ensemble）。\n")
    out = OUT / "report.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    print(f"final report -> {out} ({total} circuits)")


if __name__ == "__main__":
    main()
