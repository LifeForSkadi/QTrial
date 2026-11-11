# -*- coding: utf-8 -*-
"""hwb12 fold 阶段诊断（2026-11-13）：10 万 swap 折叠后
保留多少 swap 分支（3cx）、链化多少长程门——30 万 cx 的来源
与空位穿越的关系。
"""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route
from qtrail.pure.mapper import post_route

LAYOUT = {0: 74, 1: 80, 2: 75, 3: 82, 4: 68, 5: 59, 6: 73, 7: 60,
          8: 76, 9: 66, 10: 61, 11: 67, 12: 69, 13: 62, 14: 83,
          15: 81, 16: 55, 17: 54, 18: 53, 19: 52}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
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

    # 路由后：非相邻 2Q 门统计（空位穿越的直接度量）
    t0 = time.time()
    routed, swaps, fl = sabre_route(circ, spec, dict(LAYOUT), seed=42,
                                    lam_ms=0.0, trials=16,
                                    mix_rollout=True)
    print(f"路由 {time.time()-t0:.0f}s swaps={swaps}", flush=True)
    nonadj = 0
    maxq = 0
    for inst in routed.ops:
        if inst.nq == 2 and inst.name != "swap":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            maxq = max(maxq, a, b)
            if not spec.adj[a, b]:
                nonadj += 1
    print(f"路由后 2Q={sum(1 for i in routed.ops if i.nq==2 and i.name!='swap')}"
          f" 非相邻={nonadj} 最大坐标={maxq}（spec.n={spec.n}）",
          flush=True)

    # fold 阶段
    t0 = time.time()
    routed2, removed = post_route(routed, fl, spec.adj)
    from qtrail.pure.post_v2 import fold_and_relink
    folded, fin = fold_and_relink(routed2, spec)
    c = Counter(i.name for i in folded.ops)
    print(f"fold {time.time()-t0:.0f}s: {dict(c)}", flush=True)


if __name__ == "__main__":
    main()
