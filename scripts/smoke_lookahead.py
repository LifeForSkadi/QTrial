# -*- coding: utf-8 -*-
"""lookahead 层（qiskit rust 语义）冒烟：5 线路 × 5 配置 × 3 种子
纯路由 swap 对比（平凡布局，无后处理，隔离路由器贡献）。

配置：D 无前瞻 / A 现状 rollout0.5 / B 仅 lookahead 层 /
      C 层+rollout / E 完整 qiskit（层 + decay 0.001/5）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.qasm_io import sanitize_qasm
from qtrail.pure.router import sabre_route

NAMES = {
    "gf2^7": "data/benchpress/qasm/feynman/gf2^7_mult.qasm",
    "dtc_100": "data/benchpress/qasm/dtc/dtc_100_cx_12345.qasm",
    "barenco_tof_10": "data/benchpress/qasm/feynman/barenco_tof_10.qasm",
    "adder_8": "data/benchpress/qasm/feynman/adder_8.qasm",
    "queko_20QBT": "data/Queko/BIGD/20QBT_100C_T0.5_c5_0.qasm",
}

CONFIGS = {
    "D_none": dict(lookahead_w=0.0, w_layer=0.0, decay_inc=0.0),
    "A_rollout": dict(lookahead_w=0.5, w_layer=0.0, decay_inc=0.0),
    "B_layer": dict(lookahead_w=0.0, w_layer=None, decay_inc=0.0),
    "C_both": dict(lookahead_w=0.5, w_layer=None, decay_inc=0.0),
    "E_qiskit": dict(lookahead_w=0.0, w_layer=None, decay_inc=0.001,
                     decay_reset=5),
    # qiskit 2.5 rust 纯语义：无 ext、无 rollout，仅 basic + 层 + decay
    "F_noext": dict(lookahead_w=0.0, w_layer=None, w_ext=0.0,
                    decay_inc=0.0),
    "G_fulldecay": dict(lookahead_w=0.0, w_layer=None, w_ext=0.0,
                        decay_inc=0.001, decay_reset=5),
}


def main():
    spec = build_tianyan287_spec(load_device_config())
    print(f"device n={spec.n}")
    rows = []
    for name, path in NAMES.items():
        p = Path(path)
        if not p.exists():
            print(f"  [skip] {name}: {path} 不存在")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "queko" in name:
            text = sanitize_qasm(text)
        circ = parse_qasm(text)
        layout = {i: i for i in range(circ.n)}
        for cfg_name, kw in CONFIGS.items():
            best = None
            t0 = time.time()
            for k in range(3):
                routed, swaps, _ = sabre_route(
                    circ, spec, layout, seed=42 + k, trials=1, **kw)
                if best is None or swaps < best:
                    best = swaps
            wall = time.time() - t0
            rows.append((name, cfg_name, best, round(wall, 1)))
            print(f"  {name:16s} {cfg_name:10s} swap={best:7d}  "
                  f"wall={wall:.1f}s", flush=True)
    print("\n汇总（min over 3 seeds）:")
    for name, *_ in rows[:1]:
        pass
    names = []
    for r in rows:
        if r[0] not in names:
            names.append(r[0])
    hdr = "circuit          " + "".join(
        f"{c:>12s}" for c in CONFIGS) + "  best"
    print(hdr)
    for name in names:
        vals = {}
        for (n_, c_, s_, w_) in rows:
            if n_ == name:
                vals[c_] = s_
        bestc = min(vals, key=lambda c: vals[c])
        line = f"{name:16s}" + "".join(f"{vals[c]:>12d}" for c in CONFIGS)
        print(line + f"  <- {bestc}")


if __name__ == "__main__":
    main()
