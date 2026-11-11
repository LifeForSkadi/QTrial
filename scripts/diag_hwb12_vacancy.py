# -*- coding: utf-8 -*-
"""hwb12 空位穿越量化（2026-11-13）：路由后非相邻=0 但 fold 后
非相邻=139536——矛盾指向路由器空位穿越（swap 进空物理位）使
fold 的 arr 跟踪错位。量化：路由坐标分布、空位坐标使用、
fold 后非相邻门的坐标特征。
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
from qtrail.pure.post_v2 import fold_only

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

    t0 = time.time()
    routed, swaps, fl = sabre_route(circ, spec, dict(LAYOUT), seed=42,
                                    lam_ms=0.0, trials=16, mix_rollout=True)
    print(f"路由 {time.time()-t0:.0f}s swaps={swaps}", flush=True)

    # 路由后坐标分布
    qs = Counter()
    for inst in routed.ops:
        for q in inst.qubits:
            qs[int(q)] += 1
    print(f"路由后坐标范围 {min(qs)}-{max(qs)}、使用 {len(qs)} 个坐标"
          f"（spec.n={spec.n}、初始布局坐标={sorted(LAYOUT.values())}）",
          flush=True)
    print(f"布局坐标外的新坐标（空位证据）: "
          f"{sorted(set(qs) - set(LAYOUT.values()))[:20]}", flush=True)

    # swap 涉及坐标
    sw_qs = Counter()
    for inst in routed.ops:
        if inst.name == "swap":
            for q in inst.qubits:
                sw_qs[int(q)] += 1
    print(f"swap 涉及坐标数 {len(sw_qs)}、其中布局外: "
          f"{sorted(set(sw_qs) - set(LAYOUT.values()))[:20]}", flush=True)

    # fold 后非相邻门坐标特征
    routed2, removed = post_route(routed, fl, spec.adj)
    folded, arr = fold_only(routed2)
    nonadj_qs = Counter()
    for inst in folded.ops:
        if inst.nq == 2 and inst.name != "swap":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            if not spec.adj[a, b]:
                nonadj_qs[a] += 1
                nonadj_qs[b] += 1
    print(f"fold 后非相邻门涉及坐标数 {len(nonadj_qs)}、"
          f"其中布局外: {sorted(set(nonadj_qs) - set(LAYOUT.values()))[:20]}",
          flush=True)


if __name__ == "__main__":
    main()
