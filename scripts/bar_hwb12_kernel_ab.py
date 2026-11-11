# -*- coding: utf-8 -*-
"""hwb12 单种子 Python 主循环 vs Numba 内核 A/B（2026-11-12）。

同布局（qiskit 0-swap 布局）/同种子/同上限，测两条路径的墙钟
与 swap 数——逐位一致性由 ab_router_kernel.py 门禁保证，这里
只量速度（含触顶 RuntimeError 的时间到异常口径）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route

LAYOUT = {0: 74, 1: 80, 2: 75, 3: 82, 4: 68, 5: 59, 6: 73, 7: 60,
          8: 76, 9: 66, 10: 61, 11: 67, 12: 69, 13: 62, 14: 83,
          15: 81, 16: 55, 17: 54, 18: 53, 19: 52}


def main():
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    p = None
    for cand in Path("data/benchpress/qasm").rglob("*.qasm"):
        if "hwb12" in cand.name:
            p = cand
            break
    circ = parse_qasm(p.read_text(encoding="utf-8", errors="replace"))
    print(f"hwb12: {len(circ.ops)} 门", flush=True)
    for cap in (2000, 20000):
        for numa in (False, True):
            label = "kernel" if numa else "python"
            t0 = time.time()
            try:
                _, swaps, _ = sabre_route(circ, spec, dict(LAYOUT),
                                          seed=42, lam_ms=None, trials=1,
                                          mix_rollout=False,
                                          max_swaps=cap, numa_loop=numa)
                print(f"max_swaps={cap} {label}: swaps={swaps} "
                      f"{time.time()-t0:.2f}s", flush=True)
            except RuntimeError as e:
                print(f"max_swaps={cap} {label}: 触顶 "
                      f"{time.time()-t0:.2f}s（{str(e)[:40]}...）",
                      flush=True)


if __name__ == "__main__":
    main()
