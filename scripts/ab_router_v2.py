"""路由器 v1/v2 逐位等价验证（增量更新优化回归门禁）。

对每条例线路：相同布局/种子/上限下分别运行 router_v1 与 router（v2），
断言 swap 数、最终布局、路由后指令序列（名称/比特/参数）完全一致。
v2 必须零差异——增量更新是纯工程加速，不允许任何语义漂移。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qtrail.devices import build_tianyan287_spec
from qtrail.config import load_device_config
from qtrail.pure.qasm import parse_qasm
from qtrail.pure import router as R2          # 增量版（当前）
from qtrail.pure import router_v1 as R1      # 全量重算版（基准）

CASES = [
    # (路径, max_swaps, 说明)
    ("data/mqtbench/stratified/bv_10.qasm", 100000, "small"),
    ("data/mqtbench/stratified/dj_25.qasm", 100000, "small"),
    ("data/benchpress/qasm/square-heisenberg/square_heisenberg_N16.qasm",
     100000, "medium"),
    ("data/benchpress/qasm/qasmbench-medium/cat_state_n22/cat_state_n22.qasm",
     100000, "medium"),
    ("data/benchpress/qasm/qasmbench-large/QV_n32/32.qasm",
     100000, "medium-large 全路由"),
    ("data/benchpress/qasm/square-heisenberg/square_heisenberg_N100.qasm",
     3000, "large 截断"),
    ("data/benchpress/qasm/qaoa/qaoa_barabasi_albert_N100_3reps.qasm",
     3000, "large 截断"),
    ("data/benchpress/qasm/qaoa/qaoa_barabasi_albert_N105_3reps.qasm",
     3000, "large 截断"),
]


def ops_sig(circ):
    return [(i.name, tuple(i.qubits), tuple(i.params)) for i in circ.ops]


def main():
    dev_cfg = load_device_config()
    spec = build_tianyan287_spec(dev_cfg)
    n_missing = 0
    for path, cap, label in CASES:
        if not Path(path).exists():
            print(f"[skip] {path} not found")
            n_missing += 1
            continue
        circ = parse_qasm(Path(path).read_text(encoding="utf-8"))
        layout = {i: int(np.arange(circ.n)[i] % spec.n) for i in range(circ.n)}
        for seed in (0, 7):
            t0 = time.time()
            o1, s1, fl1 = R1.sabre_route(circ, spec, layout, seed=seed,
                                         max_swaps=cap)
            t1 = time.time() - t0
            t0 = time.time()
            o2, s2, fl2 = R2.sabre_route(circ, spec, layout, seed=seed,
                                         max_swaps=cap)
            t2 = time.time() - t0
            ok = (s1 == s2 and fl1 == fl2 and ops_sig(o1) == ops_sig(o2))
            print(f"{label:12s} seed={seed} swaps v1={s1} v2={s2} "
                  f"layout_eq={fl1 == fl2} ops_eq={ops_sig(o1) == ops_sig(o2)} "
                  f"time v1={t1:.2f}s v2={t2:.2f}s speedup={t1 / max(t2, 1e-9):.1f}x "
                  f"{'OK' if ok else 'MISMATCH!'}")
            if not ok:
                if s1 != s2 or fl1 != fl2:
                    print(f"  swap/layout mismatch: {s1} vs {s2}; {fl1} vs {fl2}")
                else:
                    a, b = ops_sig(o1), ops_sig(o2)
                    for k in range(min(len(a), len(b))):
                        if a[k] != b[k]:
                            print(f"  first op diff at {k}: v1={a[k]} v2={b[k]}")
                            break
                raise SystemExit(1)
    print(f"\nALL EQUAL (missing cases skipped: {n_missing})")


if __name__ == "__main__":
    main()
