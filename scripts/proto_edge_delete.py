# -*- coding: utf-8 -*-
"""原型：奇环/不可嵌入交互图的"删边精确嵌入 + 残边链化"。

对 graphstate_105（105 节点奇环）的**原始逻辑图**：
  1. H 非二分（奇环），设备 7×15 网格二分 → 任何嵌入都不可能；
  2. BFS 二着色找出同色边（奇环恰好 1 条），删除之；
  3. 对 H' 做 VF2 精确嵌入（Warnsdorff + 噪声打分）；
  4. 残边用 chain_long_gates 链化（最短路径 CX 链）；
  5. 结果：2Q 681→124、2Q 深度 75→23、logf −7.504→−2.668。

⚠️ 未接入生产：生产 `global_relabel` 作用在 `fold_only(routed)` 上，
折叠后的内容图是 16-环 + 89-环（不是逻辑 105 环），删边后成为
"16-环 + 89-路径覆盖全部 105 个设备节点"的精确覆盖问题，VF2 10s
内无解。下一步是**预路由布局先验**（在路由前嵌入逻辑图，把唯一
长程门交给 SABRE），详见 docs/qft_graphstate_analysis.md §2.6。
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.metrics import compute_metrics
from qtrail.pure.post_v2 import chain_long_gates, decompose_to_platform_v2
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.vf2pp import VF2Config, find_embedding


def interaction_graph(circ):
    edges = {}
    for inst in circ.ops:
        if inst.nq == 2 and inst.name != "swap":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            e = (a, b) if a < b else (b, a)
            edges[e] = edges.get(e, 0) + 1
    return edges


def bipartite_conflicts(nodes, edges):
    """BFS 二着色；返回同色边列表（删除它们可使 H 二分）。"""
    adj = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    color = {}
    bad = []
    for start in nodes:
        if start in color:
            continue
        color[start] = 0
        dq = collections.deque([start])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    dq.append(v)
                elif color[v] == color[u]:
                    e = (u, v) if u < v else (v, u)
                    if e not in bad:
                        bad.append(e)
    return bad


def main():
    spec = build_tianyan287_spec(load_device_config())
    calib = spec.calib
    path = Path("data/mqtbench/stratified/graphstate_105.qasm")
    circ = parse_qasm(path.read_text(encoding="utf-8"))
    edges = interaction_graph(circ)
    active = sorted({p for e in edges for p in e})
    print(f"n={circ.n} 2Q_edges={len(edges)} active={len(active)} "
          f"measures={len(circ.measures)}")

    bad = bipartite_conflicts(active, edges.keys())
    print("bipartite conflicts:", bad)
    h2 = {e: c for e, c in edges.items() if e not in set(bad)}
    if len(h2) == len(edges):
        print("H 已二分，无需删边")
        return

    usable = [q for q in range(spec.n) if q not in set(spec.disabled_qubits)]
    err2q = calib.err_2q
    med = float(np.median(list(err2q.values())))
    w1q = {p: 0 for p in active}
    for inst in circ.ops:
        if inst.nq == 1:
            q = int(inst.qubits[0])
            if q in w1q:
                w1q[q] += 1
    wro = {p: 0 for p in active}
    for m in circ.measures:
        q = int(m.qubits[0])
        if q in wro:
            wro[q] += 1

    mapping, score = find_embedding(
        active, h2, spec.adj, usable,
        calib.err_1q, calib.err_ro, err2q, med,
        w1q, wro, VF2Config(wall_limit=8.0))
    print("embedding:", "found" if mapping else None, "score", score)
    if mapping is None:
        return

    out = Circuit(circ.n, name=circ.name)
    for inst in circ.ops:
        qs = tuple(int(mapping[int(q)]) for q in inst.qubits)
        out.append(Inst(inst.name, qs, inst.params))
    for m in circ.measures:
        out.measures.append(
            Inst("measure", (int(mapping[int(m.qubits[0])]),),
                 cbits=m.cbits))

    chained, n_chained = chain_long_gates(out, spec)
    final = decompose_to_platform_v2(chained, spec=spec, linear=True)
    m = compute_metrics(final, 0, calib)
    print(f"deleted={len(bad)} chained={n_chained} "
          f"2Q={m['twoq_count']} 2Qdepth={m['twoq_depth']} "
          f"depth={m['depth']} logf={m['log_est_fidelity']:.3f} "
          f"fid={m['est_fidelity']:.3e}")
    # 残边端点距离（链化成本）
    for a, b in bad:
        print("  deleted edge", a, b, "->", mapping[a], mapping[b],
              "dist", int(spec.dist[mapping[a], mapping[b]]))


if __name__ == "__main__":
    main()
