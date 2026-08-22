"""Batched problem representation fed to the policy network (pure torch)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from qtrail.devices.spec import DeviceSpec
from qtrail.problems.program_graph import ProgramGraph


@dataclass
class Batch:
    """A batch of mapping instances padded to n_max, plus a shared device.

    Program side (padded with zero rows / masked):
      p_idx  [B, n_max] int64     logical node indices (embedding mode)
      p_feat [B, n_max, 6] float  node features (zeros on padding; may be unused)
      p_adj  [B, n_max, n_max]    interaction weights
      p_mask [B, n_max] bool      1 = real node
      order  [B, n_max] int64     decode order (padded entries == n_max)
      p_n    [B] int64            real program size per instance

    Device side (shared, full size):
      d_feat [N, 7] / d_idx [N]   calib features or one-hot indices
      d_adj  [N, N] float         coupling graph (binary)
      d_mask [N] bool             usable physical qubits
      dist   [N, N] float         effective distance (topo + lambda_n * noise)
    """
    p_idx: torch.Tensor
    p_feat: torch.Tensor
    p_adj: torch.Tensor
    p_mask: torch.Tensor
    order: torch.Tensor
    p_n: torch.Tensor
    has_feat: torch.Tensor        # [B] bool: instance carries 6-dim node features
    d_feat: torch.Tensor
    d_idx: torch.Tensor
    d_adj: torch.Tensor
    d_mask: torch.Tensor
    dist: torch.Tensor
    w_sum: torch.Tensor           # [B] total interaction weight per instance
    n_max: int
    device_n: int

    @property
    def batch_size(self) -> int:
        return self.p_idx.shape[0]

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            p_idx=self.p_idx.to(device), p_feat=self.p_feat.to(device),
            p_adj=self.p_adj.to(device), p_mask=self.p_mask.to(device),
            order=self.order.to(device), p_n=self.p_n.to(device),
            has_feat=self.has_feat.to(device),
            d_feat=self.d_feat.to(device), d_idx=self.d_idx.to(device),
            d_adj=self.d_adj.to(device), d_mask=self.d_mask.to(device),
            dist=self.dist.to(device), w_sum=self.w_sum.to(device),
            n_max=self.n_max, device_n=self.device_n,
        )


def collate_instances(instances: list[ProgramGraph], spec: DeviceSpec,
                      noise_lambda: float = 0.0, dtype=torch.float32) -> Batch:
    """Pad a list of ProgramGraphs into one Batch for the given DeviceSpec."""
    B = len(instances)
    n_max = max(g.n for g in instances)
    N = spec.n

    p_idx = torch.zeros(B, n_max, dtype=torch.int64)
    p_feat = torch.zeros(B, n_max, 6, dtype=dtype)
    p_adj = torch.zeros(B, n_max, n_max, dtype=dtype)
    p_mask = torch.zeros(B, n_max, dtype=torch.bool)
    order = torch.full((B, n_max), n_max, dtype=torch.int64)
    p_n = torch.zeros(B, dtype=torch.int64)
    has_feat = torch.zeros(B, dtype=torch.bool)
    w_sum = torch.zeros(B, dtype=dtype)

    for b, g in enumerate(instances):
        n = g.n
        p_n[b] = n
        p_mask[b, :n] = True
        p_idx[b, :n] = torch.arange(n)
        p_adj[b, :n, :n] = torch.from_numpy(g.adj)
        if g.node_features is not None:
            p_feat[b, :n] = torch.from_numpy(g.node_features)
            has_feat[b] = True
        order[b, :n] = torch.from_numpy(g.logical_order)
        w_sum[b] = float(g.adj.sum() / 2.0)

    d_feat = torch.from_numpy(spec.node_features.copy()).to(dtype)
    d_idx = torch.arange(N)
    d_adj = torch.from_numpy(spec.adj.astype(np.float32)).to(dtype)
    d_mask = torch.ones(N, dtype=torch.bool)
    for q in spec.disabled_qubits:
        d_mask[q] = False
    dist = torch.from_numpy(spec.distance_matrix(noise_lambda).astype(np.float32)).to(dtype)

    return Batch(p_idx=p_idx, p_feat=p_feat, p_adj=p_adj, p_mask=p_mask,
                 order=order, p_n=p_n, has_feat=has_feat,
                 d_feat=d_feat, d_idx=d_idx,
                 d_adj=d_adj, d_mask=d_mask, dist=dist, w_sum=w_sum,
                 n_max=n_max, device_n=N)
