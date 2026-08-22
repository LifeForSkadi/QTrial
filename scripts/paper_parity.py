"""CO-MAP paper Table 1 parity analysis (8x8 grid setting).

Paper (arXiv 2605.13638, Table 1, 8x8 grid / N=64):
  MQTBench (n=15, 166 circuits):  RL 88.82  RL+PP 45.16  Qiskit 264.35
  Queko-16 (180 circuits):       RL 22.87  RL+PP 0.15   Qiskit 0.116
  Queko-20 (450 circuits):       RL 52.74  RL+PP 6.14   Qiskit 147.98

Our reproduction columns:
  static_cost        = CO-MAP objective sum 2*d(pi) of the FINAL layout
                       (RL multistart + adaptive local search; the paper's
                       RL+PP column is the same metric after their LS)
  swaps              = ACTUAL SabreSwap-routed swap count from our layout
  qiskit-o1_swaps    = ACTUAL SABRE O1 layout+routing swap count

Note the paper's RL/RL+PP numbers are the static proxy, while their Qiskit
number is actual routing — mixed-metric comparison (see analysis notes).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import TABLES_DIR

PAPER = {
    "mqtbench_n15": {"rl": 88.82, "rl_pp": 45.16, "qiskit": 264.35},
    "queko_BNTF": {"rl": 22.87, "rl_pp": 0.15, "qiskit": 0.116},
    "queko_BIGD": {"rl": 52.74, "rl_pp": 6.14, "qiskit": 147.98},
}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")
            if l.strip() and "error" not in json.loads(l)]


def main():
    base = TABLES_DIR / "paper_8x8"
    print(f"{'benchmark':<16} {'n':>4} | {'our static(=RL+PP)':>18} | "
          f"{'paper RL+PP':>11} | {'our routed':>10} | {'paper Qiskit':>12} | "
          f"{'our SABRE O1':>12}")
    print("-" * 100)
    for bench, p in PAPER.items():
        rows = load(base / f"{bench}_rows.jsonl")
        if not rows:
            print(f"{bench:<16} (no data)")
            continue
        static = statistics.mean(r["static_cost"] for r in rows)
        swaps = statistics.mean(r["swaps"] for r in rows)
        o1 = statistics.mean(r["qiskit-o1_swaps"] for r in rows)
        print(f"{bench:<16} {len(rows):>4} | {static:>18.2f} | {p['rl_pp']:>11} | "
              f"{swaps:>10.2f} | {p['qiskit']:>12} | {o1:>12.2f}")
    print()
    print("口径说明：论文 RL/RL+PP 列 = 静态 2d 代价（无路由假设）；"
          "论文 Qiskit 列 = 实际路由 SWAP。")
    print("我们同时给出 静态(=论文RL+PP口径) 与 实际路由 两个口径。")


if __name__ == "__main__":
    main()
