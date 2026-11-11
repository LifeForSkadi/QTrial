# -*- coding: utf-8 -*-
"""qiskit 消融：禁用 ConsolidateBlocks / Split2QUnitaries 后跑 O3，
确认 hwb10 的 73→24 边支撑图重写是哪个 pass 的贡献。

变体：full_o3 / no_consolidate / no_split / no_both。
指标：swap 数、输出 2Q 支撑图边数、最大度、depth、墙钟。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2, transpile
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import ConsolidateBlocks, Split2QUnitaries

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from scripts.final_report_bench import build_target


def support_stats(out):
    """输出电路 2Q 支撑图的 (边数, 最大度, 节点数)。"""
    pairs = set()
    deg = {}
    for item in out.data:
        inst = item.operation
        if inst.num_qubits == 2:
            q0, q1 = (out.find_bit(item.qubits[0]).index,
                      out.find_bit(item.qubits[1]).index)
            pairs.add((min(q0, q1), max(q0, q1)))
            deg[q0] = deg.get(q0, 0) + 1
            deg[q1] = deg.get(q1, 0) + 1
    return len(pairs), (max(deg.values()) if deg else 0), len(deg)


def main():
    spec = build_tianyan287_spec(load_device_config())
    cm = CouplingMap([(a, b) for a in range(spec.n)
                      for b in range(a + 1, spec.n) if spec.adj[a, b]])
    target = build_target(spec)
    qc = qasm2.load("data/benchpress/qasm/feynman/hwb10.qasm")
    qc = qc.decompose(reps=2)
    print(f"hwb10 原始: n={qc.num_qubits} 2Q门={qc.num_nonlocal_gates()}"
          f" 支撑图={support_stats(qc) if False else ''}", flush=True)

    orig_cb = ConsolidateBlocks.run
    orig_sp = Split2QUnitaries.run

    variants = [
        ("full_o3", False, False),
        ("no_consolidate", True, False),
        ("no_split", False, True),
        ("no_both", True, True),
    ]
    for name, no_cb, no_sp in variants:
        if no_cb:
            ConsolidateBlocks.run = lambda self, dag: dag
        else:
            ConsolidateBlocks.run = orig_cb
        if no_sp:
            Split2QUnitaries.run = lambda self, dag: dag
        else:
            Split2QUnitaries.run = orig_sp
        t0 = time.time()
        out = transpile(qc, target=target, coupling_map=cm,
                        basis_gates=["cx", "u3"], optimization_level=3,
                        seed_transpiler=42)
        edges, maxdeg, nodes = support_stats(out)
        print(f"[{name:16s}] swaps={out.count_ops().get('swap', 0)} "
              f"支撑图边={edges} 最大度={maxdeg} 节点={nodes} "
              f"depth={out.depth()} wall={time.time()-t0:.0f}s", flush=True)
    ConsolidateBlocks.run = orig_cb
    Split2QUnitaries.run = orig_sp


if __name__ == "__main__":
    main()
