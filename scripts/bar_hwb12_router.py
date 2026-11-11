# -*- coding: utf-8 -*-
"""hwb12 同布局路由对照（2026-11-13）：qiskit 布局（0-swap 布局）
下，qiskit SabreSwap vs 我们 sabre_route 的 swap 数。
裁决：qiskit 路由器 ~0 → 我们路由器实现差距；qiskit 也 10 万 →
qiskit 的 0 来自 O3 swap 吸收（修复方向不同）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"

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

    # 1) qiskit SabreSwap（同布局）
    from qiskit import qasm2, transpile
    from qiskit.transpiler import CouplingMap, PassManager
    from qiskit.transpiler.passes import (SabreSwap, ApplyLayout,
                                          FullAncillaAllocation,
                                          EnlargeWithAncilla)
    from qiskit.transpiler import Layout as QLayout
    from qtrail.pipeline.baselines import build_target
    from qtrail.utils.qasm_io import load_qasm2, strip_measurements
    qc = load_qasm2(p)
    qc_clean, _ = strip_measurements(qc)
    cm = CouplingMap(list(spec.adj_edges()) if hasattr(spec, 'adj_edges')
                     else [(a, b) for a in range(spec.n)
                           for b in range(spec.n) if spec.adj[a, b]])
    t0 = time.time()
    for trials, label in ((3, "trials=3"), (16, "trials=16")):
        from qiskit.transpiler.passes import SetLayout
        pm = PassManager([
            SetLayout(QLayout({qc_clean.qubits[i]: LAYOUT[i]
                               for i in range(qc_clean.num_qubits)})),
            FullAncillaAllocation(cm),
            EnlargeWithAncilla(),
            ApplyLayout(),
            SabreSwap(cm, heuristic="basic", trials=trials, seed=42),
        ])
        out = pm.run(qc_clean)
        sw = out.count_ops().get("swap", 0)
        print(f"qiskit SabreSwap({label}) 同布局: swaps={sw}"
              f"（{time.time()-t0:.0f}s）", flush=True)

    # 2) 我们的路由器（同布局，1 种子）
    torch.manual_seed(42)
    t0 = time.time()
    routed, swaps, fl = sabre_route(circ, spec, dict(LAYOUT), seed=42,
                                    lam_ms=None, trials=1,
                                    mix_rollout=False)
    print(f"我们 sabre_route(trials=1) 同布局: swaps={swaps}"
          f"（{time.time()-t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
