"""Decoding strategies: greedy / sampling / multistart (single-instance API)."""
from __future__ import annotations

import numpy as np
import torch

from qtrail.config import DecodeConfig
from qtrail.devices.spec import DeviceSpec
from qtrail.envs import terminal_cost_np
from qtrail.models import QAPolicy
from qtrail.problems import ProgramGraph, collate_instances

_cost = lambda pi, adj, dist: terminal_cost_np(pi, adj, dist, dist_mult=2.0)


def decode_layout(policy: QAPolicy, graph: ProgramGraph, spec: DeviceSpec,
                  noise_lambda: float = 0.0, mode: str = "greedy",
                  order: np.ndarray | None = None, dev: str = "cpu",
                  batch_size: int = 32) -> np.ndarray:
    """Decode a layout pi [n] (logical -> physical) for one instance.

    Args:
        mode: greedy | sample.
        order: optional decode order override (perturbed orders for diversity).
    """
    g = graph
    if order is not None:
        g = ProgramGraph(n=graph.n, adj=graph.adj, node_features=graph.node_features,
                         logical_order=order, circuit_id=graph.circuit_id,
                         ops_meta=graph.ops_meta)
    batch = collate_instances([g], spec, noise_lambda=noise_lambda).to(dev)
    policy.eval()
    with torch.no_grad():
        _, _, pi = policy(batch, mode="greedy" if mode == "greedy" else "sample")
    return pi[0, :graph.n].cpu().numpy().astype(np.int64)


def decode_layouts_batch(policy: QAPolicy, graphs: list[ProgramGraph],
                         spec: DeviceSpec, noise_lambda: float = 0.0,
                         mode: str = "greedy", dev: str = "cpu",
                         batch_size: int = 32) -> list[np.ndarray]:
    """Decode layouts for many instances (batched for speed)."""
    outs = []
    for start in range(0, len(graphs), batch_size):
        chunk = graphs[start:start + batch_size]
        batch = collate_instances(chunk, spec, noise_lambda=noise_lambda).to(dev)
        policy.eval()
        with torch.no_grad():
            _, _, pi = policy(batch, mode=mode)
        for b, g in enumerate(chunk):
            outs.append(pi[b, :g.n].cpu().numpy().astype(np.int64))
    return outs


def multistart_decode(policy: QAPolicy, graph: ProgramGraph, spec: DeviceSpec,
                      cfg: DecodeConfig, noise_lambda: float = 0.0,
                      dev: str = "cpu", rng: np.random.Generator | None = None,
                      k: int | None = None) -> list[np.ndarray]:
    """Generate up to k diverse candidate layouts (QTrail multi-start).

    Composition: `samples` sampling decodes + `shuffled_greedy` greedy decodes
    with shuffled decode orders + one plain greedy. Deduplicated by cost,
    top-k by (cost) returned.
    """
    rng = rng or np.random.default_rng(0)
    k = k or cfg.multistart_k
    cands: list[np.ndarray] = []
    for _ in range(cfg.samples):
        cands.append(decode_layout(policy, graph, spec, noise_lambda,
                                   mode="sample", dev=dev))
    for _ in range(cfg.shuffled_greedy):
        order = rng.permutation(graph.n)
        cands.append(decode_layout(policy, graph, spec, noise_lambda,
                                   mode="greedy", order=order, dev=dev))
    cands.append(decode_layout(policy, graph, spec, noise_lambda,
                               mode="greedy", dev=dev))

    dist_eff = spec.distance_matrix(noise_lambda)
    scored = [(c, float(_cost(c, graph.adj, dist_eff))) for c in cands]
    scored.sort(key=lambda t: t[1])
    unique: list[np.ndarray] = []
    for c, cost in scored:
        if all(not np.array_equal(c, u) for u in unique):
            unique.append(c)
        if len(unique) >= k:
            break
    return unique
