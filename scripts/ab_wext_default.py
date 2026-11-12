# -*- coding: utf-8 -*-
"""P1-6 裁决 A/B（2026-11-12 外来审查捕获）：路由器 w_ext 生产默认
实际 = 0.0（mapper 三处调用都不传、签名默认 0.0），文档声称 0.5/n。

同布局同种子对比 w_ext=0.0 vs None（→0.5/n，qiskit decay 语义）：
纯路由 swap 数（5 条混合线路 × seeds 3）。判据：多数线路不劣化
且合计 swap 下降 → 改 mapper 传 w_ext=None；否则改文档。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route

spec = build_tianyan287_spec(load_device_config())
TARGETS = [
    ("hwb10", "data/benchpress/qasm/feynman/hwb10.qasm"),
    ("gf2^10", "data/benchpress/qasm/feynman/gf2^10_mult.qasm"),
    ("dtc_100", "data/benchpress/qasm/dtc/dtc_100_cx_12345.qasm"),
    ("barenco", "data/benchpress/qasm/feynman/barenco_tof_10.qasm"),
    ("adder_8", "data/benchpress/qasm/feynman/adder_8.qasm"),
]


def main():
    total = {0.0: 0, "wext": 0}
    for name, path in TARGETS:
        p = Path(path)
        if not p.exists():
            print(f"[跳过] {name}: {p} 不存在")
            continue
        circ = parse_qasm(p.read_text(encoding="utf-8", errors="replace"))
        # 平凡布局：每个逻辑比特放同一物理位（只测路由评分差异）
        layout = {i: i for i in range(circ.n)}
        row = {}
        for tag, wx in (("w_ext=0", 0.0), ("w_ext=0.5/n", None)):
            t0 = time.time()
            r, s, fin = sabre_route(circ, spec, dict(layout), seed=42,
                                    lam_ms=0.0, trials=3,
                                    mix_rollout=(circ.n > 10), w_ext=wx)
            row[tag] = s
            print(f"  {name} {tag}: swaps={s} ({time.time()-t0:.0f}s)",
                  flush=True)
        total[0.0] += row["w_ext=0"]
        total["wext"] += row["w_ext=0.5/n"]
        mark = ("✓" if row["w_ext=0.5/n"] <= row["w_ext=0"]
                else "✗")
        print(f"[{name}] 0 → 0.5/n: {row['w_ext=0']} → "
              f"{row['w_ext=0.5/n']} {mark}", flush=True)
    print(f"\n合计: w_ext=0 {total[0.0]} vs 0.5/n {total['wext']}")
    better = total["wext"] <= total[0.0]
    print(f"裁决: {'0.5/n 不劣（改 mapper 传 None）' if better else '0 更优（改文档）'}")


if __name__ == "__main__":
    main()
