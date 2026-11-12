# -*- coding: utf-8 -*-
"""VF2++ 精确嵌入覆盖率扫描（第 6 项判据 C4 的真实样本测量）。

用法：python scripts/scan_vf2pp_coverage.py [--n 20] [--budget 2.0]

对 large 层（n ≥ 25）抽样：从 benchpress/mqtbench 的 large 目录按
排序后等距抽取 N 条，计算交互图（2Q 门对，不含 swap）→ 活跃节点数/
最大度 → 自研 VF2++ 在设备图上找精确嵌入（墙钟预算 budget 秒）。
报告：总体成功率、21-40 活跃（管线激活区间）成功率、按活跃数分层。

注：不可嵌入的图（10×10 格子/稠密 QV/长环）无论算法多强都不可能有解，
覆盖率应结合"可嵌入性"解读——本脚本同时报告度序列上界预检结果。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qtrail.devices import build_tianyan287_spec
from qtrail.pure import vf2pp as V
from qtrail.pure.qasm import load_qasm_file


def interaction_graph(circ):
    edges = {}
    for inst in circ.ops:
        if inst.nq == 2 and inst.name != "swap":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            e = (a, b) if a < b else (b, a)
            edges[e] = edges.get(e, 0) + 1
    return edges


def sample_paths(n):
    roots = [Path("data/benchpress/qasm"), Path("data/mqtbench/stratified")]
    files = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.rglob("*.qasm")))
    picked = []
    for p in files:
        try:
            circ = load_qasm_file(p)
        except Exception:
            continue
        if circ.n >= 25:
            picked.append((p, circ.n))
    picked.sort(key=lambda t: (t[1], str(t[0])))
    if not picked:
        return []
    if len(picked) <= n:
        return picked
    step = len(picked) / n
    return [picked[int(i * step)] for i in range(n)]


def main():
    n = 20
    budget = 2.0
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    if "--budget" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--budget") + 1])
    spec = build_tianyan287_spec()
    G = spec.adj
    usable = [q for q in range(spec.n)
              if q not in set(spec.disabled_qubits)]
    err2q = spec.calib.err_2q
    med = float(np.median(list(err2q.values())))
    rows = []
    for p, _n in sample_paths(n):
        circ = load_qasm_file(p)
        edges = interaction_graph(circ)
        if not edges:
            continue
        active = sorted({x for e in edges for x in e})
        deg = {u: 0 for u in active}
        for (a, b) in edges:
            deg[a] += 1
            deg[b] += 1
        w1q = {u: 1 for u in active}
        wro = {u: 0 for u in active}
        t0 = time.time()
        mp, _ = V.find_embedding(
            active, edges, G, usable, spec.calib.err_1q, spec.calib.err_ro,
            err2q, med, w1q, wro,
            V.VF2Config(wall_limit=budget))
        dt = time.time() - t0
        rows.append((p.name, circ.n, len(active), max(deg.values()),
                     mp is not None, dt))
        print(f"{p.name:28s} n={circ.n:3d} active={len(active):3d} "
              f"maxdeg={max(deg.values()):2d} embed={'YES' if mp is not None else 'no':3s} "
              f"t={dt:.2f}s", flush=True)
    tot = len(rows)
    if not tot:
        print("无样本")
        return
    succ = sum(1 for r in rows if r[4])
    mid = [r for r in rows if 21 <= r[2] <= 40]
    mid_succ = sum(1 for r in mid if r[4])
    print(f"\n总体：{succ}/{tot} = {100*succ/tot:.0f}%")
    if mid:
        print(f"21-40 活跃（管线激活区间）：{mid_succ}/{len(mid)} = "
              f"{100*mid_succ/len(mid):.0f}%")
    print(f"最大单次耗时 {max(r[5] for r in rows):.2f}s")


if __name__ == "__main__":
    main()
