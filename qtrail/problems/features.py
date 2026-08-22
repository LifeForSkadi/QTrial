"""6-dim program node features (CO-MAP Appendix A).

1. single-gate density          mu_s = #1Q gates on q / total gates
2. entanglement density (ctrl)  mu_c = #2Q gates where q is control / total 2Q gates
3. entanglement density (tgt)   mu_t = #2Q gates where q is target / total 2Q gates
4. influence score              fraction of qubits reachable <= 4 hops (undirected)
5. pagerank centrality          weighted PageRank (alpha=0.85), min-max scaled
6. quantum causal cone          forward reachability via control->target edges / n
"""
from __future__ import annotations

import networkx as nx
import numpy as np

R_HOPS = 4


def compute_node_features(n: int, adj: np.ndarray,
                          single_count: np.ndarray, control_count: np.ndarray,
                          target_count: np.ndarray, n_1q: int, n_2q: int) -> np.ndarray:
    total = max(n_1q + n_2q, 1)

    mu_s = single_count / total
    mu_c = control_count / max(n_2q, 1)
    mu_t = target_count / max(n_2q, 1)

    # influence: undirected reachability within R_HOPS
    reach = (adj > 0).astype(np.float32)
    acc = reach.copy()
    cur = reach.copy()
    for _ in range(R_HOPS - 1):
        cur = (cur @ reach > 0).astype(np.float32)
        acc += cur
    influence = np.clip(acc.sum(axis=1) / max(n - 1, 1), 0.0, 1.0)

    # pagerank on the weighted (undirected) program graph
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                g.add_edge(i, j, weight=float(adj[i, j]))
    if g.number_of_edges() > 0:
        pr = nx.pagerank(g, alpha=0.85, weight="weight", max_iter=100)
    else:
        pr = {i: 1.0 / n for i in range(n)}
    pr_arr = np.array([pr[i] for i in range(n)], dtype=np.float32)
    lo, hi = float(pr_arr.min()), float(pr_arr.max())
    pr_arr = (pr_arr - lo) / (hi - lo + 1e-9)

    # causal cone: forward reachability through control->target edges
    dir_adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if control_count[i] > 0 and adj[i, j] > 0 and i != j:
                # edge i->j counts towards cone of i only if i acted as control
                # somewhere (approximation of directed influence)
                dir_adj[i, j] = 1.0
    cone = np.zeros(n, dtype=np.float32)
    if n_2q > 0:
        cur = dir_adj.copy()
        acc = cur.copy()
        for _ in range(R_HOPS - 1):
            cur = (cur @ dir_adj > 0).astype(np.float32)
            acc += cur
        cone = np.clip(acc.sum(axis=1) / max(n - 1, 1), 0.0, 1.0)

    return np.stack([mu_s, mu_c, mu_t, influence, pr_arr, cone], axis=1).astype(np.float32)
