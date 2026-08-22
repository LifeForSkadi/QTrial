"""Consolidate eval JSONL outputs into a final comparison report (markdown)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import TABLES_DIR


def load_rows(bench: str) -> list[dict]:
    p = TABLES_DIR / f"{bench}_rows.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _median(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    return (vals[mid] + vals[~mid]) / 2 if len(vals) % 2 == 0 else vals[mid]


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def render_section(f, title, rows, baselines=("qiskit-o1", "qiskit-o3")):
    ok = [r for r in rows if "error" not in r and "swaps" in r]
    if not ok:
        f.write(f"\n## {title}\n\n(无有效数据)\n")
        return
    f.write(f"\n## {title}（{len(ok)}/{len(rows)} 条线路）\n\n")
    f.write("| 指标（中位数） | QTrial | qiskit-o1 | qiskit-o3 | 相对 o1 | 相对 o3 |\n")
    f.write("|---|---|---|---|---|---|\n")

    def cell(key):
        m = _median([r.get(key) for r in ok])
        return f"{m:.2f}" if m == m else "-"

    def pct(base_key):
        ours = _median([r.get("swaps") for r in ok])
        base = _median([r.get(base_key) for r in ok])
        if ours and base:
            red = 100 * (1 - ours / base)
            return f"{red:+.1f}%"
        return "-"

    f.write(f"| SWAP 门数 | {cell('swaps')} | "
            f"{cell('qiskit-o1_swaps')} | {cell('qiskit-o3_swaps')} | "
            f"{pct('qiskit-o1_swaps')} | {pct('qiskit-o3_swaps')} |\n")
    f.write(f"| 2Q 门数 | {cell('twoq')} | {cell('qiskit-o1_twoq')} | "
            f"{cell('qiskit-o3_twoq')} | - | - |\n")
    f.write(f"| 深度 | {cell('depth')} | {cell('qiskit-o1_depth')} | "
            f"{cell('qiskit-o3_depth')} | - | - |\n")
    f.write(f"| 估计保真度 | {cell('fidelity')} | "
            f"{cell('qiskit-o1_fidelity')} | {cell('qiskit-o3_fidelity')} | - | - |\n")
    m2 = mean([r.get("mean_2q_err") for r in ok])
    m2o1 = mean([r.get("qiskit-o1_mean_2q_err") for r in ok])
    if m2 == m2 and m2o1 == m2o1:
        f.write(f"| 加权 2Q 误差（均值, ×1e-3） | {m2*1e3:.2f} | {m2o1*1e3:.2f} | "
                f"{mean([r.get('qiskit-o3_mean_2q_err') for r in ok])*1e3:.2f} | - | - |\n")
    sc = mean([r.get("static_cost") for r in ok])
    sc1 = mean([r.get("qiskit-o1_static_cost") for r in ok])
    if sc == sc and sc1 == sc1:
        f.write(f"| 静态代价 CO-MAP 指标（均值） | {sc:.1f} | {sc1:.1f} | "
                f"{mean([r.get('qiskit-o3_static_cost') for r in ok]):.1f} | "
                f"{100*(1-sc/sc1):+.1f}% | - |\n")
    f.write("\n")


def main():
    benches = [p.stem.replace("_rows", "") for p in TABLES_DIR.glob("*_rows.jsonl")]
    out = TABLES_DIR / "report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# QTrial 最终评测报告\n\n")
        f.write("设备：天衍-287（105 比特，15×7 网格，合成校准数据，含空间缺陷热区）\n")
        f.write("方法：GAT+噪声感知 RL 初始映射（multistart×10 + 自适应局部搜索）+ SabreSwap 路由\n\n")
        for bench in sorted(benches):
            rows = load_rows(bench)
            render_section(f, bench, rows)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
