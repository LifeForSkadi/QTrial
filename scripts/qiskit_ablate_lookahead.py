# -*- coding: utf-8 -*-
"""qiskit 侧 lookahead 权重消融：用 Heuristic 对象直接注入 SabreSwap，
区分"评分效应"（权重渐进）vs"结构效应"（层存在与否改变行为）。

w = 0（无层）/ 1e-9 / 1e-4 / 0.5/105（默认）/ 0.05 / 0.5，
gf2^7 平凡布局，trials=1 seed=42。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (SabreSwap, TrivialLayout, ApplyLayout,
                                      FullAncillaAllocation, EnlargeWithAncilla)
from qiskit._accelerate.sabre import (Heuristic, BasicHeuristic,
                                      LookaheadHeuristic, SetScaling,
                                      DecayHeuristic)

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec

WEIGHTS = [0.0, 1e-9, 1e-4, 0.5 / 105, 0.005, 0.05, 0.5]


def main():
    spec = build_tianyan287_spec(load_device_config())
    edges = [(int(a), int(b)) for a in range(spec.n)
             for b in range(a + 1, spec.n) if spec.adj[a, b]]
    cm = CouplingMap(edges)
    qc = qasm2.load("data/benchpress/qasm/feynman/gf2^7_mult.qasm")
    qc = qc.decompose(reps=3)
    n = qc.num_qubits
    print(f"gf2^7 n={n} gates={qc.size()}")
    for w in WEIGHTS:
        h = Heuristic(attempt_limit=10 * n)
        h = h.with_basic(1.0, SetScaling.Constant)
        if w > 0:
            h = h.with_lookahead([w], SetScaling.Constant)
        t0 = time.time()
        pm = PassManager([
            TrivialLayout(cm),
            FullAncillaAllocation(cm),
            EnlargeWithAncilla(),
            ApplyLayout(),
            SabreSwap(cm, heuristic=h, seed=42, trials=1),
        ])
        out = pm.run(qc)
        swaps = out.count_ops().get("swap", 0)
        print(f"  lookahead_w={w:>10.6f}  swaps={swaps:5d}  wall={time.time()-t0:.1f}s",
              flush=True)
    # 对照：默认字符串 heuristic
    for heur in ("basic", "lookahead", "decay"):
        pm = PassManager([
            TrivialLayout(cm),
            FullAncillaAllocation(cm),
            EnlargeWithAncilla(),
            ApplyLayout(),
            SabreSwap(cm, heuristic=heur, seed=42, trials=1),
        ])
        out = pm.run(qc)
        print(f"  str-{heur:10s} swaps={out.count_ops().get('swap', 0)}",
              flush=True)


if __name__ == "__main__":
    main()
