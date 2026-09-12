"""针对天衍-287 拓扑的大规模样本池生成器（目标 ~2 万条）。

两个生成器，全部以 287 的耦合图（105 比特，合成 182 边或 live 实测
168 边）为生成源：

1. queko287 —— QUEKO 风格"已知最优映射"基准（文献口径：QUEKO
   给任意耦合图生成映射基准）。用**自研无拒绝生成器**实现：官方
   queko.py 的拒绝采样在 20+ 比特高密度下退化死循环（实测 3.8 亿次
   随机尝试未完成一条 20 比特样本），本生成器改为直接放置，构造保证
   与官方一致——每个周期内门不共享节点、2Q 门全部落在耦合边上、
   最优映射已知（= 生成的随机排列，按它布局全部 2Q 门天然相邻，
   最优深度 = 周期数）。在 287 全芯片及 4 档连通子图（20/40/60/80
   比特）上扫 深度×密度×种子，每条样本附带其最优映射。
2. native287 —— 原生层化随机线路：每层从 287 耦合图随机最大匹配
   采样放 cx（随机控制方向），按密度撒 1Q 门。无已知最优，但 2Q 门
   100% 落在 287 真实耦合器上，拓扑特征与芯片一致。

输出结构（data/tianyan287/）：
  queko287/qasm/*.qasm + queko287/manifest.jsonl（含 optimal_mapping）
  native287/qasm/*.qasm + native287/manifest.jsonl
  graph_pool.pkl —— 全部样本的程序图（直接进训练图池）

用法：
  python tools/generate_tianyan287_pool.py --n-queko 10000 --n-native 10000
  python tools/generate_tianyan287_pool.py --live   # 用真机实测拓扑（需 API key）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import numpy as np

from qtrail.devices import build_tianyan287_spec
from qtrail.problems.program_graph import build_program_graph

# 深度 × 密度网格（queko287）
DEPTHS = [10, 20, 40, 80]
DENSITIES = [(0.1, 0.9), (0.3, 0.7), (0.5, 0.5)]   # (1Q 密度, 2Q 密度)
SIZES = [20, 40, 60, 80, 105]                        # 105 = 全芯片
NATIVE_DEPTHS = [20, 40, 80, 120]


def usable_edges(spec) -> list:
    """287 耦合边（内部索引，排除禁用比特）。"""
    bad = set(spec.disabled_qubits)
    return [(a, b) for a, b in spec.edges if a not in bad and b not in bad]


def random_connected_subgraph(spec, rng, size: int) -> list:
    """从 287 可用比特中随机生长 size 个节点的连通子图。

    每步从"边界节点"（已有未见邻居的节点）中随机挑一个向外生长，
    保证子图连通（287 可用图连通，size ≤ 可用节点数时边界恒非空）。
    """
    nodes = [q for q in range(spec.n) if q not in set(spec.disabled_qubits)]
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(usable_edges(spec))
    start = int(rng.choice(nodes))
    seen = {start}
    while len(seen) < size:
        boundary = [u for u in seen
                    if any(v not in seen for v in g.neighbors(u))]
        u = int(rng.choice(boundary))
        v = int(rng.choice([x for x in g.neighbors(u) if x not in seen]))
        seen.add(v)
    return sorted(seen)


def queko_to_qasm(cycles: list, size: int) -> str:
    """QUEKO cycles → QASM 2.0（QUEKO 官方同款：QASM 用局部逻辑 id
    0..size-1，最优映射单独在 manifest 里给出；1Q 用 x，2Q 用 cx）。"""
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f"qreg q[{size}];"]
    for cycle in cycles:
        for g in cycle:
            if len(g) == 1:
                lines.append(f"x q[{g[0]}];")
            else:
                lines.append(f"cx q[{g[0]}],q[{g[1]}];")
    return "\n".join(lines) + "\n"


def gen_queko_sample(rng, sub_edges: list, subset: list, depth: int,
                     dens: tuple) -> tuple:
    """自研无拒绝 QUEKO 风格生成器：返回 (cycles, optimal_mapping)。

    cycles = 每周期一个门列表（1Q=(q,)，2Q=(a,b)，q/a/b 为 subset 内
    位置 0..size-1）；optimal_mapping[k] = 287 物理索引。构造保证
    （与官方 QUEKO 同款属性）：
      1. 每周期内门不共享节点；
      2. 2Q 门全部落在耦合边（sub_edges）上；
      3. 按 optimal_mapping 布局时全部 2Q 门相邻 → 最优深度 = 周期数。

    算法：① 关键路径——每周期放 1 个门，相邻周期共享一个关键节点
    （官方同款，保证深度下界）；② 剩余 2Q 门按周期容量（该周期空闲
    节点子图的最大匹配）直接分配、从匹配中随机抽子集放置（天然不
    共享节点）；③ 剩余 1Q 门在空闲节点上不放回抽样直接放置。
    无拒绝循环，任意密度/规模线性时间完成。
    """
    size = len(subset)
    g = nx.Graph()
    g.add_nodes_from(range(size))
    g.add_edges_from(sub_edges)
    edges = list(g.edges())

    n_1q = max(int(dens[0] * size * depth), depth)
    n_2q = max(int(dens[1] * size * depth / 2), 0)
    n_1q = min(n_1q, depth * size - 2 * n_2q)          # 总门数 ≤ 深度×节点

    cycles = [[] for _ in range(depth)]
    used = [set() for _ in range(depth)]               # 每周期已占节点

    # ---- ① 关键路径（每周期恰好 1 个门，与上一周期共享关键节点）
    critical = {rng.randrange(size)}
    for c in range(depth):
        node = rng.choice(list(critical))
        cand_edges = [e for e in edges if node in e and
                      e[0] not in used[c] and e[1] not in used[c]]
        # 1Q 配额耗尽时强制 2Q；否则随机（需有可用边才可放 2Q）
        want_2q = n_2q > 0 and (n_1q == 0 or
                                (cand_edges and rng.random() < 0.5))
        if want_2q:
            e = rng.choice(cand_edges)
            cycles[c].append(e)
            used[c].update(e)
            critical = set(e)
            n_2q -= 1
        else:
            cycles[c].append((node,))
            used[c].add(node)
            critical = {node}
            n_1q -= 1

    # ---- ② 剩余 2Q 门：按周期容量直接分配
    caps = []
    for c in range(depth):
        h = g.copy()
        h.remove_nodes_from(used[c])
        caps.append(len(nx.maximal_matching(h)))
    if sum(caps) < n_2q:
        n_2q = sum(caps)                               # 密度收缩保护（不拒绝）
    order = list(range(depth))
    rng.shuffle(order)
    for c in order:
        if n_2q <= 0:
            break
        take = min(caps[c], n_2q)      # 确定性排空；随机性来自周期顺序
        if take == 0:
            continue
        h = g.copy()
        h.remove_nodes_from(used[c])
        match = list(nx.maximal_matching(h))
        rng.shuffle(match)
        for e in match[:take]:
            cycles[c].append(e)
            used[c].update(e)
        n_2q -= take

    # ---- ③ 剩余 1Q 门：空闲节点不放回抽样
    for c in range(depth):
        free = [q for q in range(size) if q not in used[c]]
        while n_1q > 0 and free:
            q = int(rng.choice(free))
            cycles[c].append((q,))
            used[c].add(q)
            free.remove(q)
            n_1q -= 1
    assert n_1q == 0 and n_2q == 0, (n_1q, n_2q)

    # ---- ④ 随机置换：门标签按 perm 重标记，最优映射为非平凡置换。
    # 语义：重标记后的电路比特 perm[i] 应放在物理 subset[i]（该对在
    # 287 耦合边上），故 optimal_mapping[j] = subset[perm⁻¹(j)]。
    perm = list(range(size))
    rng.shuffle(perm)
    perm_inv = [0] * size
    for i, p in enumerate(perm):
        perm_inv[p] = i
    relabeled = [[(perm[g[0]], perm[g[1]]) if len(g) == 2 else (perm[g[0]],)
                  for g in cyc] for cyc in cycles]
    opt_map = [subset[perm_inv[j]] for j in range(size)]
    return relabeled, opt_map


def gen_queko287(spec, n_total: int, out_dir: Path, rng) -> list:
    """QUEKO 风格生成器扫网格。返回程序图池列表。"""
    edges = usable_edges(spec)
    all_nodes = [q for q in range(spec.n) if q not in set(spec.disabled_qubits)]
    qasm_dir = out_dir / "qasm"
    qasm_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    pool, n_done = [], 0
    n_combos = len(SIZES) * len(DEPTHS) * len(DENSITIES)
    per = max((n_total + n_combos - 1) // n_combos, 1)   # 向上取整达目标量
    t0 = time.time()
    with open(manifest, "w", encoding="utf-8") as mf:
        for size in SIZES:
            for depth in DEPTHS:
                for dens in DENSITIES:
                    for k in range(per):
                        seed = 100000 * size + 1000 * depth + 10 * k
                        r = random.Random(seed)
                        subset = (all_nodes if size >= len(all_nodes) else
                                  random_connected_subgraph(
                                      spec, random.Random(seed), size))
                        sub_edges = [(a, b) for a, b in edges
                                     if a in set(subset) and b in set(subset)]
                        local = {node: i for i, node in enumerate(subset)}
                        sub_edges_local = [(local[a], local[b])
                                           for a, b in sub_edges]
                        cycles, opt_map = gen_queko_sample(
                            r, sub_edges_local, subset, depth, dens)
                        # 自检：最优映射下全部 2Q 门必须落在 287 耦合边
                        for cyc in cycles:
                            for g in cyc:
                                if len(g) == 2 and \
                                        not spec.adj[opt_map[g[0]],
                                                      opt_map[g[1]]]:
                                    raise AssertionError(
                                        f"optimal mapping invalid: {fname}")
                        n_2q = sum(1 for cyc in cycles
                                   for g in cyc if len(g) == 2)
                        fname = f"q287_s{size}_d{depth}_p{dens[1]}_k{k}.qasm"
                        (qasm_dir / fname).write_text(
                            queko_to_qasm(cycles, size), encoding="utf-8")
                        graph = build_program_graph(
                            size,
                            [(("cx" if len(g) == 2 else "x"), g)
                             for cyc in cycles for g in cyc],
                            circuit_id=f"queko287/{fname[:-5]}")
                        mf.write(json.dumps({
                            "file": fname, "seed": seed, "size": size,
                            "depth": depth, "dens": list(dens),
                            "n_2q": n_2q,
                            "optimal_mapping": list(opt_map),
                            "subset": subset,
                        }, ensure_ascii=False) + "\n")
                        pool.append(graph)
                        n_done += 1
                        if n_done % 500 == 0:
                            print(f"  queko287 {n_done}/{n_total} "
                                  f"({time.time()-t0:.0f}s)", flush=True)
                        if n_done >= n_total:
                            return pool
    return pool


def gen_native287(spec, n_total: int, out_dir: Path, rng) -> list:
    """原生层化随机线路（2Q 门从 287 耦合图匹配采样）。"""
    edges = usable_edges(spec)
    g = nx.Graph()
    g.add_nodes_from(range(spec.n))
    g.add_edges_from(edges)
    qasm_dir = out_dir / "qasm"
    qasm_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    pool, n_done = [], 0
    n_combos = len(SIZES) * len(NATIVE_DEPTHS)
    per = max((n_total + n_combos - 1) // n_combos, 1)
    t0 = time.time()
    with open(manifest, "w", encoding="utf-8") as mf:
        for size in SIZES:
            for depth in NATIVE_DEPTHS:
                for k in range(per):
                    r = np.random.default_rng(100000 * size + depth + k)
                    subset = random_connected_subgraph(
                        spec, random.Random(int(r.integers(0, 2**31))), size)
                    local = {node: i for i, node in enumerate(subset)}
                    sub = g.subgraph(subset)
                    ops = []
                    for layer in range(depth):
                        match = nx.maximal_matching(sub)
                        for a, b in match:
                            if r.random() < 0.5:
                                a, b = b, a
                            ops.append(("cx", (local[int(a)],
                                               local[int(b)])))
                        for q in subset:
                            u = float(r.random())
                            if u < 0.2:
                                ops.append(("rz", (local[q],)))
                            elif u < 0.3:
                                ops.append(("sx", (local[q],)))
                            elif u < 0.35:
                                ops.append(("x", (local[q],)))
                    fname = f"n287_s{size}_d{depth}_k{k}.qasm"
                    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";',
                             f"qreg q[{size}];"]
                    for name, qs in ops:
                        if len(qs) == 1:
                            if name == "rz":
                                ang = float(r.random() * 2 * np.pi)
                                lines.append(f"rz({ang:.4f}) q[{qs[0]}];")
                            else:
                                lines.append(f"{name} q[{qs[0]}];")
                        else:
                            lines.append(f"cx q[{qs[0]}],q[{qs[1]}];")
                    (qasm_dir / fname).write_text(
                        "\n".join(lines) + "\n", encoding="utf-8")
                    graph = build_program_graph(size, ops,
                                                circuit_id=f"native287/{fname[:-5]}")
                    mf.write(json.dumps({"file": fname, "size": size,
                                         "depth": depth,
                                         "subset": subset,
                                         "n_2q": sum(1 for n, q in ops
                                                     if len(q) == 2)},
                                        ensure_ascii=False) + "\n")
                    pool.append(graph)
                    n_done += 1
                    if n_done % 1000 == 0:
                        print(f"  native287 {n_done}/{n_total} "
                              f"({time.time()-t0:.0f}s)", flush=True)
                    if n_done >= n_total:
                        return pool
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queko", type=int, default=10000)
    ap.add_argument("--n-native", type=int, default=10000)
    ap.add_argument("--out", default="data/tianyan287")
    ap.add_argument("--live", action="store_true",
                    help="用真机实测拓扑（需天衍 API key，失败自动回退合成）")
    ap.add_argument("--no-pool", action="store_true",
                    help="不构建程序图池（只要 qasm）")
    args = ap.parse_args()

    spec = build_tianyan287_spec()
    if args.live:
        try:
            from qtrail.devices.adapter import download_tianyan287_spec
            spec = download_tianyan287_spec()
            print(f"live topology: {spec.name} "
                  f"n={spec.n} edges={len(spec.edges)}")
        except Exception as e:
            print(f"live 下载失败，回退合成配置：{e}")
    print(f"topology: n={spec.n} usable_edges="
          f"{len(usable_edges(spec))} disabled={spec.disabled_qubits}")

    out = Path(args.out)
    rng = random.Random(20260908)

    pool = []
    if args.n_queko > 0:
        print(f"queko287 target {args.n_queko} ...")
        pool += gen_queko287(spec, args.n_queko, out / "queko287", rng)
    if args.n_native > 0:
        print(f"native287 target {args.n_native} ...")
        pool += gen_native287(spec, args.n_native, out / "native287", rng)
    print(f"total samples: {len(pool)}")

    if pool and not args.no_pool:
        import pickle
        pool_path = out / "graph_pool.pkl"
        with open(pool_path, "wb") as f:
            pickle.dump(pool, f)
        print(f"saved graph pool ({len(pool)} graphs) -> {pool_path}")


if __name__ == "__main__":
    main()
