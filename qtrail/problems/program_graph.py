"""Program graph: logical-qubit interaction graph derived from a circuit.

Pure numpy — no qiskit dependency. The qiskit -> ops extraction lives in
utils/qasm_io.py; this module consumes a generic list of (name, qubits) ops.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qtrail.problems.features import compute_node_features


@dataclass
class ProgramGraph:
    n: int                              # logical qubit count
    adj: np.ndarray                     # [n, n] float32 symmetric, w_ij = #2Q gates
    node_features: np.ndarray | None    # [n, 6] float32 (None -> one-hot mode)
    logical_order: np.ndarray           # [n] int64 decode order
    circuit_id: str = "circuit"
    ops_meta: dict = field(default_factory=dict)  # extra stats for logging

    @property
    def total_weight(self) -> float:
        return float(self.adj.sum() / 2.0)

    def cost(self, pi: np.ndarray, dist: np.ndarray, dist_mult: float = 2.0) -> float:
        """CO-MAP terminal cost for a full mapping pi (logical -> physical).

        cost = dist_mult * sum_{i<j} w_ij * dist[pi_i, pi_j]
        """
        idx = pi.astype(np.int64)
        d = dist[np.ix_(idx, idx)]
        return float((self.adj * d).sum() / 2.0 * dist_mult)


def build_program_graph(n: int, ops: list, circuit_id: str = "circuit",
                        order: np.ndarray | None = None,
                        compute_feats: bool = True,
                        temporal_alpha: float = 0.0) -> ProgramGraph:
    """Build a ProgramGraph from a list of ops.

    Args:
        n: number of logical qubits.
        ops: list of (name, (q0, q1)) tuples for 2Q gates and (name, (q0,))
            for 1Q gates (other gates ignored, e.g. barriers/measures).
        order: decode order; default = qubit appearance order 0..n-1.
        temporal_alpha: 时序感知加权（QTrail 优化）：边权 += alpha × 该交互对
            跨越的层数（贪心分层）。强调跨时间层的持续耦合（QFT 式链式交互），
            使布局更重视贯穿整条线路的交互结构。
    """
    adj = np.zeros((n, n), dtype=np.float32)
    control_count = np.zeros(n, dtype=np.float32)
    target_count = np.zeros(n, dtype=np.float32)
    single_count = np.zeros(n, dtype=np.float32)
    n_2q = 0
    n_1q = 0
    # 时序分层（贪心）：layer(q) = 比特 q 上次所在层 + 1
    last_layer = np.zeros(n, dtype=np.int64)
    pair_layers: dict = {}  # (i,j) -> set of layers
    cur_layer = 0
    for name, qs in ops:
        if len(qs) == 1:
            single_count[qs[0]] += 1
            n_1q += 1
            cur_layer = max(cur_layer, int(last_layer[qs[0]]) + 1)
            last_layer[qs[0]] = cur_layer
        elif len(qs) == 2:
            c, t = int(qs[0]), int(qs[1])
            adj[c, t] += 1
            adj[t, c] += 1
            control_count[c] += 1
            target_count[t] += 1
            n_2q += 1
            cur_layer = max(int(last_layer[c]), int(last_layer[t])) + 1
            last_layer[c] = last_layer[t] = cur_layer
            if temporal_alpha > 0:
                key = (min(c, t), max(c, t))
                pair_layers.setdefault(key, set()).add(cur_layer)
        # >2Q gates must be decomposed before calling this function

    if temporal_alpha > 0 and pair_layers:
        for (i, j), layers in pair_layers.items():
            adj[i, j] += temporal_alpha * len(layers)
            adj[j, i] += temporal_alpha * len(layers)

    feats = None
    if compute_feats:
        feats = compute_node_features(n, adj, single_count, control_count,
                                      target_count, n_1q, n_2q)

    if order is None:
        order = np.arange(n, dtype=np.int64)

    return ProgramGraph(
        n=n,
        adj=adj,
        node_features=feats,
        logical_order=order,
        circuit_id=circuit_id,
        ops_meta={"n_1q": n_1q, "n_2q": n_2q},
    )


def random_program_graph(n: int, p: float = 0.3, rng: np.random.Generator | None = None,
                         weighted: bool = False, circuit_id: str = "random") -> ProgramGraph:
    """Erdos-Renyi program graph (CO-MAP training distribution, p=0.3).

    With weighted=True edges get a random interaction count 1..5, mimicking
    circuits with repeated interactions.
    """
    rng = rng or np.random.default_rng()
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w = int(rng.integers(1, 6)) if weighted else 1
                adj[i, j] = adj[j, i] = w
    return ProgramGraph(
        n=n,
        adj=adj,
        node_features=None,  # one-hot / embedding mode for random graphs
        logical_order=np.arange(n, dtype=np.int64),
        circuit_id=circuit_id,
        ops_meta={"n_1q": 0, "n_2q": int(adj.sum() / 2)},
    )
