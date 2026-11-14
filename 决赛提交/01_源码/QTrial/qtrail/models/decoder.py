"""Pointer-attention decoder (CO-MAP reproduction).

Context variants (paper Sec. 4.2.2):
  concat_project   c = W [h_cur; h_prev]            (default, best for greedy)
  project_concat   c = W [W1 h_cur; W2 h_prev]
  stack_project    c = W mean({h_0..h_t})           (worst in paper)

Attention scores per head m for physical node j:
  s[m, j] = C * tanh( q_m(c) . K_j / sqrt(d) ),  C = 10 (paper Eq. 11)

QTrail extension (rich_context): a value-aware distance context adds
  s[m, j] += w_m * dist_ctx[j] + b_m
where dist_ctx[j] = sum over assigned neighbors q' of the current logical
qubit: w(q, q') * dist[pi(q'), j] — the partial CO cost is injected into
attention so the decoder sees its own progress.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qtrail.models.layers import init_linear


class PointerDecoder(nn.Module):
    def __init__(self, d: int = 128, heads: int = 16, clamp: float = 10.0,
                 context: str = "concat_project", rich_context: bool = False):
        super().__init__()
        self.heads = heads
        self.clamp = clamp
        self.context = context
        self.rich_context = rich_context

        if context == "concat_project":
            self.ctx_proj = nn.Linear(2 * d, d)
        elif context == "project_concat":
            self.cur_proj = nn.Linear(d, d // 2)
            self.prev_proj = nn.Linear(d, d // 2)
            self.ctx_proj = nn.Linear(d, d)
        elif context == "stack_project":
            self.ctx_proj = nn.Linear(d, d)
        else:
            raise ValueError(f"unknown context variant: {context}")

        # per-head query projection; shared key projection (paper Eq. 10)
        self.wq = nn.Linear(d, heads * d)
        self.wk = nn.Linear(d, d)

        # rich context: per-head affine transform of the distance context scalar
        if rich_context:
            self.dist_w = nn.Parameter(torch.zeros(heads))
            self.dist_b = nn.Parameter(torch.zeros(heads))
        self.apply(init_linear)

    def build_context(self, h_cur: torch.Tensor, h_prev: torch.Tensor,
                      h_stack_mean: torch.Tensor | None = None) -> torch.Tensor:
        if self.context == "concat_project":
            return self.ctx_proj(torch.cat([h_cur, h_prev], dim=-1))
        if self.context == "project_concat":
            return self.ctx_proj(torch.cat([self.cur_proj(h_cur),
                                            self.prev_proj(h_prev)], dim=-1))
        # stack_project
        return self.ctx_proj(h_stack_mean)

    def scores(self, ctx: torch.Tensor, d_emb: torch.Tensor,
               dist_ctx: torch.Tensor | None = None) -> torch.Tensor:
        """Pointer attention scores [B, M, N] (before masking).

        ctx: [B, d]; d_emb: [B, N, d]; dist_ctx: [B, N] optional.
        """
        B, N, _ = d_emb.shape
        q = self.wq(ctx).view(B, self.heads, -1)          # [B, M, d]
        k = self.wk(d_emb)                                # [B, N, d]
        s = torch.einsum("bmd,bnd->bmn", q, k) / (k.shape[-1] ** 0.5)
        s = self.clamp * torch.tanh(s)
        if dist_ctx is not None:
            s = s + self.dist_w.view(1, -1, 1) * dist_ctx.unsqueeze(1) \
                + self.dist_b.view(1, -1, 1)
        return s

    # ------------------------------------------------------- action selection
    @staticmethod
    def masked_logits(scores: torch.Tensor, assigned: torch.Tensor,
                      done: torch.Tensor) -> torch.Tensor:
        """Apply -1e9 for occupied physical qubits and finished episodes.

        Uses -1e9 (not -inf) so log_softmax stays finite for sampling.

        scores: [B, M, N]; assigned: [B, N] bool; done: [B] bool.
        """
        logits = scores.clone()
        logits = logits.masked_fill(assigned.unsqueeze(1), -1e9)
        logits = logits.masked_fill(done.view(-1, 1, 1), -1e9)
        return logits

    def greedy(self, scores: torch.Tensor, assigned: torch.Tensor,
               done: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """argmax over (head, physical qubit); returns (action [B], head [B])."""
        logits = self.masked_logits(scores, assigned, done)
        flat = logits.view(logits.shape[0], -1)                    # [B, M*N]
        idx = flat.argmax(dim=-1)
        head = idx // logits.shape[-1]
        action = idx % logits.shape[-1]
        # finished episodes: pick a safe dummy action (logp zeroed elsewhere)
        action = action.clamp(min=0)
        return action, head

    def sample(self, scores: torch.Tensor, assigned: torch.Tensor,
               done: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a head uniformly, then an action from that head's softmax.

        Returns (action [B], head [B], logp [B], entropy [B]) where logp
        excludes the constant head-choice term and is 0 for finished episodes;
        entropy is the sampled head's categorical entropy (0 when done).
        """
        B, M, N = scores.shape
        logits = self.masked_logits(scores, assigned, done)
        head = torch.randint(0, M, (B,), device=scores.device)
        logp_head = F.log_softmax(logits, dim=-1)[
            torch.arange(B, device=scores.device), head]         # [B, N]
        dist = torch.distributions.Categorical(logits=logp_head)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        logp = logp.masked_fill(done, 0.0)
        entropy = entropy.masked_fill(done, 0.0)
        return action, head, logp, entropy

    def logp_of(self, scores: torch.Tensor, assigned: torch.Tensor,
                actions: torch.Tensor, heads: torch.Tensor,
                done: torch.Tensor) -> torch.Tensor:
        """Log-probability of given (action, head) pairs (for off-policy use)."""
        logits = self.masked_logits(scores, assigned, done)
        B = logits.shape[0]
        logp_head = F.log_softmax(logits, dim=-1)[
            torch.arange(B, device=scores.device), heads]
        logp = logp_head.gather(1, actions.unsqueeze(1)).squeeze(1)
        return logp.masked_fill(done, 0.0)
