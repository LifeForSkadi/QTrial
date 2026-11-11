# -*- coding: utf-8 -*-
"""路由器 Python 主循环 vs Numba 内核逐位等价验证（2026-11-12 门禁）。

对每条例线路：相同布局/种子/上限下分别运行 Python 路径（numa_loop=False）
与 Numba 内核路径（默认），断言 swap 数、最终布局、路由后指令序列
（名称/比特/参数）完全一致。内核必须零差异——纯工程加速，不允许任何
语义漂移（含逃逸路径：故意用小 attempt 上限触发防卡死回卷）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import numpy as np

from qtrail.devices import build_tianyan287_spec
from qtrail.config import load_device_config
from qtrail.pure.qasm import parse_qasm
from qtrail.pure import router as R

CASES = [
    # (路径, max_swaps, 说明)
    ("data/mqtbench/stratified/bv_10.qasm", 100000, "small"),
    ("data/mqtbench/stratified/dj_25.qasm", 100000, "small"),
    ("data/benchpress/qasm/square-heisenberg/square_heisenberg_N16.qasm",
     100000, "medium（heis N16 口径锚点）"),
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
    # 链结构压力（hwb 类同对链密集）
    ("data/benchpress/qasm/feynman/hwb8.qasm", 100000, "hwb8 链"),
]


def ops_sig(circ):
    return [(i.name, tuple(i.qubits), tuple(i.params)) for i in circ.ops]


def main():
    dev_cfg = load_device_config()
    spec = build_tianyan287_spec(dev_cfg)
    n_fail = 0
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
            o_py, s_py, fl_py = R.sabre_route(circ, spec, dict(layout),
                                              seed=seed, max_swaps=cap,
                                              numa_loop=False)
            t_py = time.time() - t0
            t0 = time.time()
            # 小线路默认走 Python 路径（固定开销门槛）——测试时强制
            # 内核，保证小线路上的内核语义也被门禁覆盖
            os.environ["QTRAIL_ROUTER_NUMBA_ALL"] = "1"
            o_k, s_k, fl_k = R.sabre_route(circ, spec, dict(layout),
                                           seed=seed, max_swaps=cap,
                                           numa_loop=True)
            os.environ.pop("QTRAIL_ROUTER_NUMBA_ALL", None)
            t_k = time.time() - t0
            sig_py = ops_sig(o_py)
            sig_k = ops_sig(o_k)
            ok = (s_py == s_k and fl_py == fl_k and sig_py == sig_k)
            status = "OK " if ok else "FAIL"
            if not ok:
                n_fail += 1
                for idx, (a, b) in enumerate(zip(sig_py, sig_k)):
                    if a != b:
                        print(f"  首个分歧 @{idx}: py={a} kernel={b}")
                        break
                if s_py != s_k:
                    print(f"  swaps: py={s_py} kernel={s_k}")
                if fl_py != fl_k:
                    print(f"  layout 分歧: py={fl_py} kernel={fl_k}")
            print(f"[{status}] {label} seed={seed} swaps={s_py} "
                  f"py={t_py:.3f}s kernel={t_k:.3f}s "
                  f"(x{t_py / max(t_k, 1e-9):.1f})", flush=True)
    print(f"\n失败 {n_fail} / 缺失 {n_missing}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
