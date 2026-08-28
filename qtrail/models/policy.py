"""QAPolicy: program encoder + device encoder + pointer decoder assembly.

Implements the CO-MAP sequential decoding loop (one logical qubit per step,
fixed order), greedy / sampling action selection, and terminal sparse reward.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qtrail.config import Config, GraphConfig, ModelConfig, RewardConfig
from qtrail.envs.qap_env import terminal_reward
from qtrail.models.decoder import PointerDecoder
from qtrail.models.gat import GATEncoder
from qtrail.models.graph_transformer import GraphTransformerEncoder
from qtrail.models.layers import init_linear
from qtrail.problems.instance import Batch

_NEG = -1e9


class QAPolicy(nn.Module):
    def __init__(self, model: ModelConfig, device_n: int,
                 max_program_n: int | None = None,
                 reward_cfg: RewardConfig | None = None,
                 graph_cfg: GraphConfig | None = None):
        super().__init__()
        self.model_cfg = model
        self.d = model.d
        self.device_n = device_n
        self.max_n = max_program_n or device_n
        r = reward_cfg or RewardConfig()
        self.reward_dist_mult = r.dist_mult
        self.reward_normalize = r.normalize
        self.reward_depth_lambda = r.depth_lambda
        self.reward_compactness_lambda = r.compactness_lambda
        self.graph_cfg = graph_cfg or GraphConfig()

        # ---- input projections (both paths always exist; per-instance select)
        self.prog_lin = nn.Linear(6, model.d)
        self.prog_emb = nn.Embedding(self.max_n + 1, model.d)
        if model.device_features == "calib":
            self.dev_proj = nn.Linear(7, model.d)
        else:
            self.dev_proj = nn.Embedding(device_n, model.d)

        # ---- encoders (shared architecture family via config)
        # LAUREL 残差（arXiv:2411.07501）：旧 checkpoint 的 model_cfg 无该字段，
        # 用 getattr 兜底 → 默认 "none" → 行为不变。
        laurel = getattr(model, "laurel", "none")
        laurel_rank = getattr(model, "laurel_rank", 4)
        enc_kwargs = dict(d=model.d, layers=model.gat_layers, heads=model.gat_heads,
                          dropout=model.gat_dropout, norm=model.norm,
                          laurel=laurel, laurel_rank=laurel_rank)
        if model.encoder == "gat":
            self.prog_encoder = GATEncoder(**enc_kwargs)
            self.dev_encoder = GATEncoder(**enc_kwargs)
        elif model.encoder == "gt":
            gt_kwargs = dict(d=model.d, layers=model.gt_layers, heads=model.gt_heads,
                             ff=model.gt_ff, dropout=model.gt_dropout,
                             dist_bias=model.gt_dist_bias, tau=model.gt_tau,
                             laurel=laurel, laurel_rank=laurel_rank)
            self.prog_encoder = GraphTransformerEncoder(**gt_kwargs, bias_from_adj=True)
            self.dev_encoder = GraphTransformerEncoder(**gt_kwargs, bias_from_adj=False)
        else:
            raise ValueError(f"unknown encoder: {model.encoder}")

        self.decoder = PointerDecoder(d=model.d, heads=model.decoder_heads,
                                      clamp=model.clamp, context=model.context,
                                      rich_context=model.rich_context)
        self.apply(init_linear)

    # ------------------------------------------------------------ encoding
    def encode_program(self, batch: Batch) -> torch.Tensor:
        """Per-instance input path: 6-dim features where available (circuit
        graphs), learned index embedding otherwise (random graphs; paper's
        one-hot fallback)."""
        B, n_max, _ = batch.p_feat.shape
        if self.model_cfg.program_features == "onehot":
            x = self.prog_emb(batch.p_idx.clamp(0, self.max_n))
        else:
            x = torch.zeros(B, n_max, self.d, device=batch.p_idx.device)
            n_feat = int(batch.has_feat.sum())
            if n_feat > 0:
                x[batch.has_feat] = self.prog_lin(batch.p_feat[batch.has_feat])
            x[~batch.has_feat] = self.prog_emb(
                batch.p_idx[~batch.has_feat].clamp(0, self.max_n))
        return self.prog_encoder(x, batch.p_adj, batch.p_mask)

    def encode_device(self, batch: Batch) -> torch.Tensor:
        if self.model_cfg.device_features == "calib":
            x = self.dev_proj(batch.d_feat)                       # [N, d]
        else:
            x = self.dev_proj(batch.d_idx)
        x = x.unsqueeze(0)                                        # [1, N, d]
        mask = batch.d_mask.unsqueeze(0)
        adj = batch.d_adj.unsqueeze(0)
        return self.dev_encoder(x, adj, mask).squeeze(0)          # [N, d]

    # -------------------------------------------------------------- rollout
    def forward(self, batch: Batch, mode: str = "sample",
                return_pi: bool = True):
        """One rollout over the batch.

        Returns (logp [B], reward [B], pi [B, n_max]) for mode='sample';
        for mode='greedy' logp is a dummy zero tensor.
        """
        B, n_max = batch.p_idx.shape
        N = batch.device_n  # 多拓扑训练：设备规模随 batch 变化
        dev = batch.p_idx.device

        p_emb = self.encode_program(batch)                        # [B, n, d]
        d_emb = self.encode_device(batch)                         # [N, d]
        d_emb_b = d_emb.unsqueeze(0).expand(B, N, -1)             # [B, N, d]

        assigned = torch.zeros(B, N, dtype=torch.bool, device=dev)
        assigned |= ~batch.d_mask.unsqueeze(0)                    # unusable qubits pre-masked
        logical_assigned = torch.zeros(B, n_max, dtype=torch.bool, device=dev)
        pi = torch.zeros(B, n_max, dtype=torch.int64, device=dev)
        total_logp = torch.zeros(B, device=dev)
        total_entropy = torch.zeros(B, device=dev)
        prev_emb: torch.Tensor | None = None
        stack_sum = torch.zeros(B, self.d, device=dev)

        for t in range(n_max):
            done = t >= batch.p_n                                  # [B]
            cur_idx = batch.order[:, t].clamp(0, n_max - 1)        # [B]

            h_cur = p_emb.gather(1, cur_idx.view(B, 1, 1).expand(B, 1, self.d)).squeeze(1)
            h_prev = h_cur if prev_emb is None else prev_emb
            stack_sum = stack_sum + h_cur
            stack_mean = stack_sum / (t + 1)

            ctx = self.decoder.build_context(h_cur, h_prev, stack_mean)

            dist_ctx = None
            if self.model_cfg.rich_context:
                dist_ctx = self._dist_context(batch, cur_idx, logical_assigned, pi, done)

            s = self.decoder.scores(ctx, d_emb_b, dist_ctx)        # [B, M, N]

            if mode == "sample":
                a, h, logp_t, ent_t = self.decoder.sample(s, assigned, done)
                total_logp = total_logp + logp_t
                total_entropy = total_entropy + ent_t
            else:
                a, h = self.decoder.greedy(s, assigned, done)

            # write pi[b, logical] = a for non-done instances
            pi_new = pi.scatter(1, cur_idx.unsqueeze(1), a.unsqueeze(1))
            pi = torch.where(done.unsqueeze(1), pi, pi_new)
            logical_assigned = logical_assigned.scatter(
                1, cur_idx.unsqueeze(1), (~done).unsqueeze(1))
            assigned = assigned.scatter(1, a.unsqueeze(1),
                                        torch.ones(B, 1, dtype=torch.bool, device=dev))
            prev_emb = h_cur

        reward = terminal_reward(pi, batch.p_adj, batch.dist, batch.w_sum,
                                 dist_mult=self.reward_dist_mult,
                                 normalize=self.reward_normalize,
                                 depth_lambda=self.reward_depth_lambda,
                                 compactness_lambda=self.reward_compactness_lambda)
        self.last_entropy = total_entropy / batch.p_n.clamp(min=1)
        return total_logp, reward, pi

    # ------------------------------------------------------------ helpers
    def _dist_context(self, batch: Batch, cur_idx: torch.Tensor,
                      logical_assigned: torch.Tensor, pi: torch.Tensor,
                      done: torch.Tensor) -> torch.Tensor:
        """dist_ctx[b, j] = mean over placed neighbors l of cur:
        w(cur, l) * dist[pi(l), j]. Zero for finished episodes."""
        B, n_max = batch.p_idx.shape
        N = batch.device_n
        w_cur = batch.p_adj.gather(1, cur_idx.view(B, 1, 1).expand(B, 1, n_max)).squeeze(1)
        pi_c = pi.clamp(0, N - 1)
        dist_b = batch.dist if batch.dist.dim() == 3 else \
            batch.dist.unsqueeze(0).expand(B, N, N)
        d_sel = dist_b.gather(1, pi_c.unsqueeze(2).expand(B, n_max, N))  # [B, n, N]
        contrib = (w_cur.unsqueeze(2) * logical_assigned.unsqueeze(2) * d_sel)
        cnt = logical_assigned.sum(1, keepdim=True).clamp(min=1)             # [B, 1]
        ctx = contrib.sum(1) / cnt                                          # [B, N]
        return ctx.masked_fill(done.unsqueeze(1), 0.0)

    # ---------------------------------------------------------- checkpoint
    def save_checkpoint(self, path, *, epoch: int, val_cost: float,
                        optimizer=None, rng=None, extra: dict | None = None) -> None:
        import torch as _t
        ckpt = {
            "model": self.state_dict(),
            "epoch": epoch,
            "val_cost": val_cost,
            "model_cfg": self.model_cfg,
            "reward_cfg": {"dist_mult": self.reward_dist_mult,
                           "normalize": self.reward_normalize,
                           "depth_lambda": self.reward_depth_lambda,
                           "compactness_lambda": self.reward_compactness_lambda},
            "graph_cfg": self.graph_cfg.__dict__ if hasattr(self, "graph_cfg") else None,
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        if rng is not None:
            ckpt["rng"] = rng
        if extra:
            ckpt.update(extra)
        _t.save(ckpt, path)

    @classmethod
    def load_checkpoint(cls, path, device_n: int, map_location=None,
                        max_program_n: int | None = None):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model_cfg = ckpt.get("model_cfg")
        if model_cfg is None:
            model_cfg = ModelConfig()  # legacy checkpoints
        reward_cfg = None
        if ckpt.get("reward_cfg"):
            reward_cfg = RewardConfig(**ckpt["reward_cfg"])
        graph_cfg = None
        if ckpt.get("graph_cfg"):
            graph_cfg = GraphConfig(**ckpt["graph_cfg"])
        if max_program_n is None and "model" in ckpt:
            # 从检查点推断 embedding 表尺寸（多拓扑检查点可达 485）
            emb = ckpt["model"].get("prog_emb.weight")
            if emb is not None:
                max_program_n = emb.shape[0] - 1
        policy = cls(model_cfg, device_n, max_program_n=max_program_n,
                     reward_cfg=reward_cfg, graph_cfg=graph_cfg)
        policy.load_state_dict(ckpt["model"])
        if isinstance(map_location, (str, torch.device)):
            policy = policy.to(map_location)
        return policy, ckpt
