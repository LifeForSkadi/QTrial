# -*- coding: utf-8 -*-
"""对拍：我们的 sabre_route 与 qiskit SabreSwap 在同一平凡布局、同 seed
下的逐 swap 序列，找第一个分歧点（basic 与 lookahead 两配置）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (SabreSwap, TrivialLayout, ApplyLayout,
                                      FullAncillaAllocation, EnlargeWithAncilla)

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route


def qiskit_swaps(qc, cm, heuristic, seed):
    pm = PassManager([
        TrivialLayout(cm),
        FullAncillaAllocation(cm),
        EnlargeWithAncilla(),
        ApplyLayout(),
        SabreSwap(cm, heuristic=heuristic, seed=seed, trials=1),
    ])
    out = pm.run(qc)
    seq = []
    for item in out.data:
        inst = item.operation
        if inst.name == "swap":
            qargs = item.qubits
            seq.append(tuple(sorted(out.find_bit(q).index for q in qargs)))
    return seq


def our_swaps(circ, spec, layout, seed, **kw):
    routed, swaps, fl = sabre_route(circ, spec, layout, seed=seed, trials=1,
                                    **kw)
    seq = []
    for inst in routed.ops:
        if inst.name == "swap":
            seq.append(tuple(sorted(inst.qubits)))
    return seq


def main():
    spec = build_tianyan287_spec(load_device_config())
    edges = [(int(a), int(b)) for a in range(spec.n)
             for b in range(a + 1, spec.n) if spec.adj[a, b]]
    cm = CouplingMap(edges)
    for name, path, heur, kw in [
        ("gf2^7", "data/benchpress/qasm/feynman/gf2^7_mult.qasm",
         "basic", dict(lookahead_w=0.0, w_layer=0.0, w_ext=0.0,
                       cand_scope="front")),
        ("gf2^7", "data/benchpress/qasm/feynman/gf2^7_mult.qasm",
         "lookahead", dict(lookahead_w=0.0, w_layer=None, w_ext=0.0,
                           cand_scope="front")),
        ("adder_8", "data/benchpress/qasm/feynman/adder_8.qasm",
         "basic", dict(lookahead_w=0.0, w_layer=0.0, w_ext=0.0,
                       cand_scope="front")),
        ("adder_8", "data/benchpress/qasm/feynman/adder_8.qasm",
         "lookahead", dict(lookahead_w=0.0, w_layer=None, w_ext=0.0,
                           cand_scope="front")),
    ]:
        qc = qasm2.load(path)
        qc = qc.decompose(reps=3)
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        circ = parse_qasm(text)
        layout = {i: i for i in range(circ.n)}
        qs = qiskit_swaps(qc, cm, heur, 42)
        os_ = our_swaps(circ, spec, layout, 42, **kw)
        print(f"{name} [{heur}] qiskit={len(qs)} ours={len(os_)}")
        n = min(len(qs), len(os_))
        diff = None
        for k in range(n):
            if qs[k] != os_[k]:
                diff = k
                break
        if diff is None and len(qs) != len(os_):
            diff = n
        if diff is None:
            print("  序列完全一致" if len(qs) == len(os_) else
                  f"  前 {n} 步一致，长度不同")
        else:
            print(f"  第 {diff} 步分歧: qiskit={qs[diff]} ours={os_[diff]}")
            print(f"  上下文 qiskit[{max(0,diff-2)}:{diff+2}]={qs[max(0,diff-2):diff+2]}")
            print(f"  上下文 ours  [{max(0,diff-2)}:{diff+2}]={os_[max(0,diff-2):diff+2]}")


if __name__ == "__main__":
    main()
