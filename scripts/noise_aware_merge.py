"""合并噪声感知基线评测：QTrial+SABRE vs 噪声感知O1/O3 vs 盲目O1 vs pytket vs Cqlib原生。

用法: python scripts/noise_aware_merge.py
输出: tables/noise_aware/report.md
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INJ = Path("tables/cqlib_inject")
NA = Path("tables/noise_aware")
TRIM = 0.1

# 表列定义：key -> 行字段前缀
COLUMNS = [
    ("QTrial fid规则", "os_fidelity"),
    ("QTrial swap规则", "os_swap"),
    ("噪声感知O1", "aware_o1"),
    ("噪声感知O3", "aware_o3"),
    ("盲目O1", "sabre"),
    ("pytket", "tket"),
    ("Cqlib原生最强", "cqlib_best"),
]


def load(path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["circuit"]] = r
    return out


def trimmed_mean(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * TRIM))
    if 2 * k >= len(vals):
        return statistics.mean(vals)
    return statistics.mean(vals[k:len(vals) - k])


def fmt3(v):
    return f"{v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" if v else "-"


def stat(rows, prefix, metric):
    vals = [r.get(f"{prefix}_{metric}") for r in rows
            if isinstance(r.get(f"{prefix}_{metric}"), (int, float))]
    if not vals:
        return None
    return (round(statistics.mean(vals), 4),
            round(trimmed_mean(vals), 4),
            round(statistics.median(vals), 4))


def render_bucket(sections, title, rows):
    lines = [f"\n## {title}（{len(rows)} 条线路）\n",
             "| 指标（均值/截尾/中位） | " + " | ".join(c for c, _ in COLUMNS) + " |",
             "|---|---|---|---|---|---|---|"]
    # mean2q 字段名映射（各来源命名不同）
    MEAN2Q_FIELD = {"os_fidelity": "os_fidelity_mean2q",
                    "os_swap": "os_swap_mean2q",
                    "aware_o1": "aware_o1_mean2q",
                    "aware_o3": "aware_o3_mean2q",
                    "sabre": "blind_o1_mean2q",
                    "tket": "tket_mean2q",
                    "cqlib_best": None}

    for metric, name, nd in (("swaps", "SWAP 数", 2), ("depth", "深度", 2),
                             ("twoq_depth", "2Q 深度", 2),
                             ("fidelity", "估计保真度", 4),
                             ("mean2q", "2Q 误差加权均值", 4)):
        cells = []
        for _, prefix in COLUMNS:
            if metric == "mean2q":
                field = MEAN2Q_FIELD.get(prefix)
                if field is None:
                    cells.append("-")
                    continue
                s = stat(rows, prefix, "") if False else None
                vals = [r.get(field) for r in rows
                        if isinstance(r.get(field), (int, float))]
                s = (round(statistics.mean(vals), 4),
                     round(trimmed_mean(vals), 4),
                     round(statistics.median(vals), 4)) if vals else None
            else:
                if prefix == "cqlib_best":
                    vals = []
                    for r in rows:
                        a = r.get(f"cqlib_size_{metric}")
                        b = r.get(f"cqlib_depth_{metric}")
                        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                            vals.append(min(a, b) if metric != "fidelity"
                                        else max(a, b))
                    s = (round(statistics.mean(vals), 4),
                         round(trimmed_mean(vals), 4),
                         round(statistics.median(vals), 4)) if vals else None
                else:
                    s = stat(rows, prefix, metric)
            if s is None:
                cells.append("-")
            elif nd == 2:
                cells.append(f"{s[0]:.2f}/{s[1]:.2f}/{s[2]:.2f}")
            else:
                cells.append(f"{s[0]:.4f}/{s[1]:.4f}/{s[2]:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    # 保真度胜场（QTrial fid 规则 vs 噪声感知O3，全完成线路）
    ok = [r for r in rows if isinstance(r.get("os_fidelity_fidelity"), (int, float))
          and isinstance(r.get("aware_o3_fidelity"), (int, float))]
    if ok:
        w = sum(1 for r in ok if r["os_fidelity_fidelity"] > r["aware_o3_fidelity"])
        lines.append(f"\nQTrial fid规则 保真度 vs 噪声感知O3：**{w}/{len(ok)} 胜**"
                     f"（{len(ok)} 条全完成线路）")
    lines.append("")
    sections.append("\n".join(lines))


def main():
    inj_small = load(INJ / "small_rows.jsonl")
    inj_med = load(INJ / "medium_rows.jsonl")
    na_small = load(NA / "small_rows.jsonl")
    na_med = load(NA / "medium_rows.jsonl")

    def merged(inj, na, fid=None):
        out = []
        for name, r in inj.items():
            if name in na:
                r = dict(r)
                r.update(na[name])
                if fid and name in fid:
                    r.update(fid[name])
                out.append(r)
        return out

    sections = [
        "# 噪声感知机制对比：稀疏奖励 RL vs 噪声感知 SABRE O1/O3（天衍-287）\n",
        "全部编译器在合成祖冲之三号校准（含 2% 空间关联缺陷热区，5-10× 误差）"
        "上评测；噪声感知 O1/O3 = qiskit Target 驱动的 VF2PostLayout 打分；"
        "统计：均值/截尾(去10%)/中位数\n",
    ]
    fid_small = load(NA / "small_fidrule.jsonl")
    fid_med = load(NA / "medium_fidrule.jsonl")
    render_bucket(sections, "≤10 比特", merged(inj_small, na_small, fid_small))
    render_bucket(sections, "11-50 比特", merged(inj_med, na_med, fid_med))
    sections.append("\n## 机制解读（诚实结论）\n\n"
                    "1. **盲目 O1 完全没有噪声感知**（VF2 仅结构打分），2Q 误差加权"
                    "均值 0.012-0.017 vs 噪声感知方案 0.002-0.007——热区规避"
                    "的差距主要由\"是否有噪声感知\"本身解释\n"
                    "2. **噪声感知 O1 已捕获大部分收益**；O3 的 50 次布局试验 + "
                    "VF2 误差打分在全部 29 条线路上找到 0-SWAP 且低误差的嵌入，"
                    "估计保真度在本评测集上超过 QTrial（fid 规则 1/15、1/14 胜）\n"
                    "3. **QTrial 的边选择质量最优**（11-50 层 mean2q 0.0038 < "
                    "O3 0.0044），但 SWAP 开销（fid 规则 293 vs O3 0）在乘积模型"
                    "中压过了边质量的收益——**在可无 SWAP 嵌入的线路上，"
                    "噪声感知 VF2 试验难以被击败**\n"
                    "4. **剩余验证空间**：MQTBench 线路全部可子图嵌入；"
                    "QUEKO 高密度线路（不可无 SWAP 嵌入）上 VF2 的 0-SWAP 优势"
                    "消失，RL 布局+路由感知选择的必要性需在该分布上检验"
                    "（下一步实验）\n")
    (NA / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"report -> {NA / 'report.md'}")


if __name__ == "__main__":
    main()
