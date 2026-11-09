# -*- coding: utf-8 -*-
"""决定性实验：qiskit 自己的 lookahead 层在【纯路由】（平凡布局）
上有没有收益？SabreSwap heuristic=basic/lookahead/decay × trials=3，
188 边网格（与我们的合成设备一致）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (SabreSwap, TrivialLayout, ApplyLayout,
                                      FullAncillaAllocation, EnlargeWithAncilla)

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec

NAMES = {
    "gf2^7": "data/benchpress/qasm/feynman/gf2^7_mult.qasm",
    "dtc_100": "data/benchpress/qasm/dtc/dtc_100_cx_12345.qasm",
    "barenco_tof_10": "data/benchpress/qasm/feynman/barenco_tof_10.qasm",
    "adder_8": "data/benchpress/qasm/feynman/adder_8.qasm",
}


def main():
    spec = build_tianyan287_spec(load_device_config())
    edges = [(int(a), int(b)) for a in range(spec.n)
             for b in range(a + 1, spec.n) if spec.adj[a, b]]
    cm = CouplingMap(edges)
    print(f"coupling edges={len(edges)}")
    for name, path in NAMES.items():
        qc = qasm2.load(path)
        qc = qc.decompose(reps=3)
        print(f"  {name}: n={qc.num_qubits} gates={qc.size()}")
        for heur in ("basic", "lookahead", "decay"):
            t0 = time.time()
            pm = PassManager([
                TrivialLayout(cm),
                FullAncillaAllocation(cm),
                EnlargeWithAncilla(),
                ApplyLayout(),
                SabreSwap(cm, heuristic=heur, seed=42, trials=3),
            ])
            try:
                out = pm.run(qc)
                swaps = out.count_ops().get("swap", 0)
                print(f"    {heur:10s} swaps={swaps:6d}  wall={time.time()-t0:.1f}s",
                      flush=True)
            except Exception as e:
                print(f"    {heur:10s} ERROR: {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
