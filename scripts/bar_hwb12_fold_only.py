# -*- coding: utf-8 -*-
"""hwb12 fold_only 验证（2026-11-13）：纯置换折叠（恒正确代数恒等式）
vs fold_and_relink（保守逐 swap 判定保留 99999 swap）。
假设：路由后非相邻=0 → 全折叠零成本 → fold_only 产物 = 原始 cx
188674（无 30 万 swap 展开）。
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
from qtrail.pure.post_v2 import fold_only, fold_and_relink, chain_long_gates

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
    routed, removed = post_route(routed, fl, spec.adj)

    # 路径 A：fold_only（纯折叠）→ 折叠后非相邻检查 → 链化
    t0 = time.time()
    folded, arr = fold_only(routed)
    nonadj = sum(1 for i in folded.ops
                 if i.nq == 2 and i.name != "swap"
                 and not spec.adj[int(i.qubits[0]), int(i.qubits[1])])
    c = Counter(i.name for i in folded.ops)
    print(f"fold_only {time.time()-t0:.0f}s: {dict(c)} "
          f"折叠后非相邻={nonadj}", flush=True)
    # 链化长程门
    t0 = time.time()
    chained, _ = chain_long_gates(folded, spec)
    cc = Counter(i.name for i in chained.ops)
    print(f"chain_long_gates {time.time()-t0:.0f}s: {dict(cc)}",
          flush=True)

    # 路径 B 对照：fold_and_relink（保守）
    t0 = time.time()
    folded2, fin2 = fold_and_relink(routed, spec)
    c2 = Counter(i.name for i in folded2.ops)
    print(f"fold_and_relink {time.time()-t0:.0f}s: {dict(c2)}",
          flush=True)


if __name__ == "__main__":
    main()
