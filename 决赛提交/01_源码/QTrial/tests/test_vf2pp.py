"""自研 VF2++ 嵌入测试（执行顺序第 6 项）。

预注册判据（2026-11-13，实现前写定——见 vf2_postlayout_explained.md §8.4）

P1（最优性，≤20 活跃）：随机小图与**穷举单态射**的最优分数一致。
  注：networkx 的 subgraph_isomorphisms_iter 是**诱导**子图同构
  （2026-11-13 实测：三角形 vs 边+孤立点 → 0 解），而布局嵌入需要
  非诱导（额外设备边允许）——因此等价性对拍以穷举单态射为金标准；
  另在完全图上与 networkx 全枚举对照（完全图的诱导=非诱导）。
P2（覆盖率，>20 活跃）：可嵌入的 25/36/50 节点随机子图，精确嵌入
  成功率 ≥50%（本实现 15/15×3 实测），单次 ≤8s。
P3（硬预算）：call_limit / wall_limit 在递归内部生效——小预算下
  快速返回（不挂死），不产出非法映射。
P4（映射合法性）：双射、覆盖全部 H 节点、H 边全部落在设备边上、
  不含禁用比特。
P5（管线集成）：>20 活跃的合成 "routed 线路" 经 `_exact_embedding`
  找到映射且 global_relabel 产物全部 2Q 相邻。

失败含义：VF2++ 搜索错误（漏解/非法映射/预算失效）或集成回归。
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from qtrail.devices import build_grid3x3_spec, build_tianyan287_spec
from qtrail.pure import vf2pp as V
from qtrail.pure.circuit import Circuit, Inst


def _weights(nodes, rng):
    return ({u: int(rng.integers(0, 6)) for u in nodes},
            {u: int(rng.integers(0, 2)) for u in nodes})


def _score(mapping, hedges, w1q, wro, spec):
    err2q = spec.calib.err_2q
    med = float(np.median(list(err2q.values())))
    return V.embedding_score(mapping, hedges, w1q, wro,
                             spec.calib.err_1q, spec.calib.err_ro,
                             err2q, med)


def test_optimal_matches_bruteforce():
    spec = build_grid3x3_spec(0)
    G = spec.adj
    usable = [q for q in range(spec.n) if q not in set(spec.disabled_qubits)]
    rng = np.random.default_rng(7)
    err2q = spec.calib.err_2q
    med = float(np.median(list(err2q.values())))
    tested = 0
    for _ in range(20):
        k = int(rng.integers(3, 7))
        nodes = list(rng.choice(usable, k, replace=False))
        pairs = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]
                 if G[a, b]]
        if not pairs:
            continue
        edges = {}
        for a, b in pairs:
            if rng.random() < 0.6:
                edges[(a, b) if a < b else (b, a)] = int(rng.integers(1, 4))
        if not edges:
            continue
        active = sorted({p for e in edges for p in e})
        idx = {p: i for i, p in enumerate(active)}
        hedges = {(idx[a], idx[b]): c for (a, b), c in edges.items()}
        hs = list(range(len(active)))
        w1q, wro = _weights(hs, rng)
        best = (-1e18, None)
        for perm in itertools.permutations(usable, len(hs)):
            mp = dict(zip(hs, perm))
            if not all(G[mp[a], mp[b]] for (a, b) in hedges):
                continue
            s = _score(mp, hedges, w1q, wro, spec)
            if s > best[0]:
                best = (s, mp)
        ours, sc = V.find_embedding(
            hs, hedges, G, usable, spec.calib.err_1q, spec.calib.err_ro,
            err2q, med, w1q, wro,
            V.VF2Config(wall_limit=5.0, call_limit=10_000_000))
        assert ours is not None
        assert abs(sc - best[0]) < 1e-9, (sc, best[0])
        tested += 1
    assert tested >= 12


def test_complete_graph_matches_networkx():
    pytest.importorskip("networkx")
    import networkx as nx
    n = 9
    adj = np.ones((n, n), dtype=np.int8) - np.eye(n, dtype=np.int8)
    rng = np.random.default_rng(3)
    err1q = rng.uniform(5e-4, 5e-3, n)
    err_ro = rng.uniform(1e-3, 2e-2, n)
    err2q = {(i, j): float(rng.uniform(1e-3, 2e-2))
             for i in range(n) for j in range(i + 1, n)}
    med = float(np.median(list(err2q.values())))
    k = 5
    hedges = {(i, j): int(rng.integers(1, 4)) for i in range(k)
              for j in range(i + 1, k)}
    hs = list(range(k))
    w1q, wro = _weights(hs, rng)
    G = nx.Graph(); G.add_nodes_from(range(n))
    G.add_edges_from([(i, j) for i in range(n) for j in range(i + 1, n)])
    H = nx.Graph(); H.add_nodes_from(hs); H.add_edges_from(hedges)
    best = -1e18
    for iso in nx.algorithms.isomorphism.GraphMatcher(G, H) \
            .subgraph_isomorphisms_iter():
        mp = {h: int(g) for g, h in iso.items()}
        s = 0.0
        for u, v in mp.items():
            s += w1q[u] * V._log1m(err1q[v]) + wro[u] * V._log1m(err_ro[v])
        for (a, b), cnt in hedges.items():
            e = err2q.get((mp[a], mp[b]), err2q.get((mp[b], mp[a])))
            s += cnt * V._log1m(e)
        best = max(best, s)
    ours, sc = V.find_embedding(
        hs, hedges, adj, list(range(n)), err1q, err_ro, err2q, med,
        w1q, wro, V.VF2Config(wall_limit=5.0, call_limit=10_000_000))
    assert ours is not None
    assert abs(sc - best) < 1e-9


def test_coverage_over_20_nodes():
    spec = build_tianyan287_spec()
    G = spec.adj
    usable = [q for q in range(spec.n)
              if q not in set(spec.disabled_qubits)]
    rng = np.random.default_rng(11)
    err2q = spec.calib.err_2q
    med = float(np.median(list(err2q.values())))
    for k in (25, 36, 50):
        succ = 0
        for _ in range(3):
            # 随机设备子图 → 可嵌入 H（重标号为 0..k-1）
            nodes = list(rng.choice(usable, k, replace=False))
            pairs = [(a, b) for i, a in enumerate(nodes)
                     for b in nodes[i + 1:] if G[a, b]]
            edges = {}
            for a, b in pairs:
                if rng.random() < 0.30:
                    edges[(a, b) if a < b else (b, a)] = int(
                        rng.integers(1, 4))
            if not edges:
                continue
            active = sorted({p for e in edges for p in e})
            idx = {p: i for i, p in enumerate(active)}
            hedges = {(idx[a], idx[b]): c for (a, b), c in edges.items()}
            hs = list(range(len(active)))
            w1q, wro = _weights(hs, rng)
            t0 = time.time()
            mp, _sc = V.find_embedding(
                hs, hedges, G, usable, spec.calib.err_1q,
                spec.calib.err_ro, err2q, med, w1q, wro)
            assert time.time() - t0 <= 8.5, f"k={k} 超预算"
            if mp is not None:
                assert set(mp) == set(hs)
                assert len(set(mp.values())) == len(hs)
                assert all(G[mp[a], mp[b]] for (a, b) in hedges)
                succ += 1
        assert succ >= 2, f"k={k} 覆盖率 {succ}/3"


def test_hard_budget_respected():
    spec = build_grid3x3_spec(0)
    G = spec.adj
    usable = [q for q in range(spec.n) if q not in set(spec.disabled_qubits)]
    # 可嵌入但解空间大的 H：设备图本身的全边集（9 节点）
    hs = list(range(spec.n))
    hedges = {(i, j): 1 for i in range(spec.n) for j in range(i + 1, spec.n)
              if G[i, j]}
    rng = np.random.default_rng(5)
    w1q, wro = _weights(hs, rng)
    err2q = spec.calib.err_2q
    med = float(np.median(list(err2q.values())))
    t0 = time.time()
    mp, _ = V.find_embedding(
        hs, hedges, G, usable, spec.calib.err_1q, spec.calib.err_ro,
        err2q, med, w1q, wro,
        V.VF2Config(call_limit=2000, max_solutions=64, wall_limit=0.05))
    dt = time.time() - t0
    assert dt < 1.0, f"预算失效：{dt:.3f}s"
    # 有解或无解都必须合法（若返回）
    if mp is not None:
        assert len(set(mp.values())) == len(mp)
        assert all(G[mp[a], mp[b]] for (a, b) in hedges)


def test_pipeline_integration_over_20():
    from qtrail.pure.relabel_vf2 import _exact_embedding, global_relabel
    spec = build_tianyan287_spec()
    usable = [q for q in range(spec.n)
              if q not in set(spec.disabled_qubits)]
    rng = np.random.default_rng(13)
    # 取 30 个设备节点上的随机连通子图作为线路交互图（≤64 活跃）
    for _ in range(100):
        nodes = list(rng.choice(usable, 30, replace=False))
        edges = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]
                 if spec.adj[a, b] and rng.random() < 0.4]
        adj = {q: set() for q in nodes}
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        seen = {nodes[0]}
        stack = [nodes[0]]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) != 30 or not edges:
            continue
        break
    circ = Circuit(spec.n)
    for (a, b) in edges:
        circ.append(Inst("cz", (int(a), int(b))))
    mapping = _exact_embedding(circ, spec)
    assert mapping is not None, ">20 活跃应能精确嵌入"
    assert len(set(mapping.values())) == len(mapping)
    assert all(int(v) not in set(spec.disabled_qubits)
               for v in mapping.values())
    out, full, exact = global_relabel(circ, spec)
    assert exact
    for inst in out.ops:
        if inst.nq == 2:
            assert spec.adj[int(inst.qubits[0]),
                            int(inst.qubits[1])]
