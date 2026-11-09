"""规模-难度网格测试集生成器（天衍-287 拓扑，6×3×5 = 90 条）。

网格：6 比特数 {10, 20, 40, 60, 80, 105} × 3 难度 {low, mid, high} × 5 种子。

难度由线路结构在生成期控制（验收口径 = 本方案+O1 路由的实测 SWAP 门数，
目标区间见 scripts/calibrate_scaling_set.py 的 BANDS）：
  low  : 交互对 = 子集网格生成树的边 + 少量近邻弦（全部网格相邻）
         → 可完全嵌入，实测 SWAP ≈ 0
  mid  : 子集内网格距离 2-3 的随机相异对，门数 ≈1.5n → 实测 ≈0.3n
  high : 远距对（网格距离 ≥4）优先、相异对密度高（≈1.8n 相异对）、
         门数 ≈2n → 小比特数靠稠密不可嵌入、大比特数靠长程张力逼出
         SWAP，实测 ≈ n~2.5n

1Q 门 = 随机角度 u1；逻辑标签随机置换（恒等布局不再平凡）。
输出：data/scaling/qasm/*.qasm + data/scaling/manifest.jsonl

用法：python tools/generate_scaling_set.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx

from qtrail.devices import build_tianyan287_spec

SIZES = [10, 20, 40, 60, 80, 105]
DIFFS = ["low", "mid", "high"]
SEEDS = [0, 1, 2, 3, 4]


def usable_edges(spec) -> list:
    bad = set(spec.disabled_qubits)
    return [(a, b) for a, b in spec.edges if a not in bad and b not in bad]


def grow_compact(spec, rng: random.Random, size: int) -> list:
    """边界生长连通子图（紧凑型，低/中难度用）。"""
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


def grow_spread(spec, size: int) -> list:
    """最远点采样子图（分散型，高难度用）：每次加入离当前集合
    最小距离最大的节点，保证子集内部存在大量远距对。"""
    nodes = [q for q in range(spec.n) if q not in set(spec.disabled_qubits)]
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(usable_edges(spec))
    start = random.Random(size * 31 + 7).choice(nodes)
    seen = {start}
    while len(seen) < size:
        # 到已选集的多源 BFS 距离（无权）
        lengths = nx.multi_source_dijkstra_path_length(g, seen, weight=None)
        far = max((v for v in nodes if v not in seen),
                  key=lambda v: lengths.get(v, 0))
        seen.add(far)
    return sorted(seen)


def grid_dist(spec, subset: list, local_i: int, local_j: int) -> int:
    c0, c1 = spec.coords[subset[local_i]], spec.coords[subset[local_j]]
    return int(abs(c0[0] - c1[0]) + abs(c0[1] - c1[1]))


def snake_order(spec) -> list:
    """全网格牛耕式（boustrophedon）哈密顿路径：逐行交替方向，
    相邻节点全部网格相邻——低难度档的交互图 = 纯路径（Fiedler 完美）。"""
    rows = {}
    for q in range(spec.n):
        rows.setdefault(int(spec.coords[q][0]), []).append(q)
    order = []
    flip = False
    for r in sorted(rows):
        line = sorted(rows[r], key=lambda q: spec.coords[q][1])
        if flip:
            line = line[::-1]
        order.extend(line)
        flip = not flip
    return order


def layerize(pairs: list) -> list:
    """贪心分层：每层两两不相交。"""
    layers = []
    remaining = list(pairs)
    while remaining:
        used, layer, rest = set(), [], []
        for a, b in remaining:
            if a not in used and b not in used:
                used.add(a)
                used.add(b)
                layer.append((a, b))
            else:
                rest.append((a, b))
        layers.append(layer)
        remaining = rest
    return layers


def build_circuit(spec, size: int, diff: str, rng: random.Random) -> tuple:
    """返回 (layers_1q, layers_2q, subset, n_2q)。"""
    n = size if size < spec.n else spec.n
    if diff == "high":
        subset = grow_spread(spec, n)
    else:
        subset = grow_compact(spec, rng, n)

    if diff == "low":
        # 牛耕式蛇形路径：交互图 = 纯路径（Fiedler 序即路径序，
        # 螺旋布局可完全嵌入）→ 实测 SWAP ≈ 0；种子差异来自置换
        # 与近邻弦的选取
        order = snake_order(spec)
        seg = order[:n]
        local_of = {q: i for i, q in enumerate(seg)}
        path = [(i, i + 1) for i in range(n - 1)]
        chords = [(a, b) for a in range(n) for b in range(a + 1, n)
                  if spec.adj[seg[a], seg[b]] and b - a > 1]
        rng.shuffle(chords)
        pairs = path + chords[:max(0, n - len(path))]
        pairs = [(int(a), int(b)) for a, b in pairs]
        n_2q_target = n
        subset = seg
    else:
        # 候选相异对按难度档的距离约束过滤
        cands = [(a, b) for a in range(n) for b in range(a + 1, n)]
        if diff == "mid":
            pool = [(a, b) for a, b in cands
                    if 2 <= grid_dist(spec, subset, a, b) <= 3]
        else:  # high
            pool = [(a, b) for a, b in cands
                    if grid_dist(spec, subset, a, b) >= 4]
        if len(pool) < 3:  # 小比特数兜底：取距离最大的若干对
            pool = sorted(cands, key=lambda e: -grid_dist(
                spec, subset, e[0], e[1]))[:max(6, n // 2)]
        n_distinct = int(1.2 * n) if diff == "mid" else int(1.8 * n)
        # 高难度小比特数加密：n≤20 时门数提到 3n，逼出稠密不可嵌入
        n_2q_target = int(1.5 * n) if diff == "mid" else (
            int(3.0 * n) if n <= 20 else int(2.0 * n))
        rng.shuffle(pool)
        pairs = pool[:min(n_distinct, len(pool))]
        while len(pairs) < n_2q_target:  # 重复对补足门数
            pairs.append(rng.choice(pool[:min(n_distinct, len(pool))]))
        pairs = pairs[:n_2q_target]

    layers = layerize(pairs)
    # 每层随机撒 1Q 门（u1 随机角度）
    layers_1q = []
    for layer in layers:
        ones = [q for q in range(n) if rng.random() < 0.25]
        if not ones:
            ones = [int(rng.randrange(n))]
        layers_1q.append([(q, round(rng.random() * 6.2832, 4)) for q in ones])
    # 逻辑标签随机置换
    perm = list(range(n))
    rng.shuffle(perm)
    rel_2q = [[(perm[a], perm[b]) for a, b in layer] for layer in layers]
    rel_1q = [[(perm[q], ang) for q, ang in layer] for layer in layers_1q]
    return rel_1q, rel_2q, subset, len(pairs)


def to_qasm(rel_1q, rel_2q, size: int) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{size}];"]
    for l1, l2 in zip(rel_1q, rel_2q):
        for q, ang in l1:
            lines.append(f"u1({ang}) q[{q}];")
        for a, b in l2:
            lines.append(f"cx q[{a}],q[{b}];")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scaling")
    args = ap.parse_args()

    spec = build_tianyan287_spec()
    out = Path(args.out)
    qasm_dir = out / "qasm"
    qasm_dir.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    n_total = 0
    with open(manifest, "w", encoding="utf-8") as mf:
        for size in SIZES:
            for diff in DIFFS:
                for k in SEEDS:
                    seed = 100000 * size + 1000 * DIFFS.index(diff) + k
                    rng = random.Random(seed)
                    rel_1q, rel_2q, subset, n_2q = build_circuit(
                        spec, size, diff, rng)
                    fname = f"s{size}_{diff}_k{k}.qasm"
                    (qasm_dir / fname).write_text(
                        to_qasm(rel_1q, rel_2q, size), encoding="utf-8")
                    mf.write(json.dumps({
                        "file": fname, "size": size, "difficulty": diff,
                        "seed": seed, "k": k, "n_2q": n_2q,
                        "n_layers": len(rel_2q), "subset": subset,
                    }, ensure_ascii=False) + "\n")
                    n_total += 1
    print(f"generated {n_total} circuits -> {qasm_dir}")


if __name__ == "__main__":
    main()
