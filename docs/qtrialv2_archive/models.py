"""HSRL-Map 路由策略：GAT 编码程序图 + 交换边评分器（纯 PyTorch）。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = max(out_dim // heads, 1)
        self.lin = nn.Linear(in_dim, heads * self.head_dim, bias=False)
        self.att_l = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_r = nn.Parameter(torch.empty(heads, self.head_dim))
        self.out_lin = nn.Linear(heads * self.head_dim, out_dim)
        self.dropout = dropout
        self.residual = in_dim == out_dim
        if not self.residual:
            self.res_lin = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        nn.init.xavier_uniform_(self.out_lin.weight)
        nn.init.zeros_(self.out_lin.bias)
        if not self.residual:
            nn.init.xavier_uniform_(self.res_lin.weight)

    def forward(self, x, adj):
        B, n, _ = x.shape
        H, dh = self.heads, self.head_dim
        Wh = self.lin(x).view(B, n, H, dh)
        e = (Wh * self.att_l).sum(-1)
        logits = F.leaky_relu(e.unsqueeze(2) + e.unsqueeze(1), 0.2)
        mask = (adj > 0).float() + torch.eye(n, device=x.device).unsqueeze(0)
        logits = logits.masked_fill(mask.unsqueeze(-1) <= 0, -1e9)
        alpha = F.softmax(logits, dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.einsum("bijh,bjhf->bihf", alpha, Wh).reshape(B, n, H * dh)
        out = self.out_lin(out)
        res = x if self.residual else self.res_lin(x)
        return F.relu(out + res)


class GATEncoder(nn.Module):
    def __init__(self, d=128, layers=4, heads=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [GATLayer(d, d, heads=heads) for _ in range(layers)])

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x


class LayoutPolicy(nn.Module):
    """布局策略：GAT 编码程序图+设备图 → 指针解码 → 初始布局。

    稀疏终局奖励 = 路由层从该布局出发的真实成果（层级闭环，创新③）。
    """

    def __init__(self, d=128, max_n=64, device_n=36, heads=8):
        super().__init__()
        self.d = d
        self.max_n = max_n
        self.device_n = device_n
        self.node_proj = nn.Linear(2, d)          # [剩余交互度, 1Q 数]
        self.dev_proj = nn.Linear(2, d)           # [度, 中心性代理=1]
        self.gnn = GATEncoder(d=d, layers=4, heads=heads)
        self.dev_gnn = GATEncoder(d=d, layers=4, heads=heads)
        self.wq = nn.Linear(d, heads * d)
        self.wk = nn.Linear(d, d)
        self.heads = heads
        self.clamp = 10.0

    def encode_program(self, adj, feats):
        return self.gnn(self.node_proj(feats), adj)

    def encode_device(self, d_adj, d_feats):
        x = self.dev_gnn(self.dev_proj(d_feats), d_adj)
        return x  # [1, N, d]

    def forward(self, adj, feats, d_adj, d_feats, mode="sample"):
        """单实例解码：返回 (pi [n], logp 张量)。"""
        B = 1
        n = adj.shape[1]
        h = self.encode_program(adj, feats)          # [1, n, d]
        d_emb = self.encode_device(d_adj, d_feats)   # [1, N, d]
        assigned = torch.zeros(1, self.device_n, dtype=torch.bool,
                               device=adj.device)
        pi = torch.zeros(n, dtype=torch.int64, device=adj.device)
        logp_sum = torch.zeros(1, device=adj.device)
        for t in range(n):
            ctx = h[0, t]                            # 顺序解码
            q = self.wq(ctx).view(1, self.heads, -1)  # [1, M, d]
            k = self.wk(d_emb[0])                     # [N, d]
            s = torch.einsum("bmd,nd->bmn", q, k.unsqueeze(0)) / (self.d ** 0.5)
            s = self.clamp * torch.tanh(s)            # [1, M, N]
            s = s.masked_fill(assigned.unsqueeze(1), -1e9)
            if mode == "greedy":
                idx = s.view(-1).argmax()
                logp = torch.zeros(1, device=adj.device)
            else:
                head = torch.randint(0, self.heads, (1,), device=adj.device)
                dist = torch.distributions.Categorical(logits=s[0, head[0]])
                idx = dist.sample()
                logp = dist.log_prob(idx)
            a = int(idx % self.device_n)
            pi[t] = a
            assigned[0, a] = True
            logp_sum = logp_sum + logp
        return pi, logp_sum


class RoutingPolicy(nn.Module):
    """深度感知路由策略（稀疏终局奖励训练）。

    状态：程序图（剩余交互权重）+ 前沿门的距离 + 候选交换的局部特征
    动作：候选 SWAP 边（掩码 softmax）
    """

    def __init__(self, d=128, max_n=64, heads=8):
        super().__init__()
        self.d = d
        self.max_n = max_n
        # 逻辑比特节点特征（迭代 2：5 维全眼前瞻特征）-> d
        self.node_proj = nn.Linear(5, d)
        self.gnn = GATEncoder(d=d, layers=4, heads=heads)
        self.empty_emb = nn.Parameter(torch.zeros(d))  # 空位置学习嵌入
        # 交换边评分：MLP([h_u, h_v, 6 维机制特征])
        # 机制特征 = [benefit, pot_delta, decay, dist, ext_benefit, ext_pot_delta]
        # —— 学习式评分函数（创新①）：机制复刻 + 权重由 RL 学习
        self.score_mlp = nn.Sequential(
            nn.Linear(2 * d + 6, d), nn.ReLU(),
            nn.Linear(d, 1),
        )

    def encode_qubits(self, rem_adj, feats):
        """rem_adj: [B, n, n] 剩余交互权重; feats: [B, n, 5] 全眼前瞻特征。"""
        return self.gnn(self.node_proj(feats), rem_adj)

    def score_candidates(self, h, cands_info):
        """cands_info: list per batch of (u, v, benefit, pot_delta, decay, dist)
        返回 logits [B, C_max]（填充 -inf 掩码）。"""
        B = len(cands_info)
        C = max(len(c) for c in cands_info)
        logits = torch.full((B, C), -float("inf"), device=h.device)
        for b, cands in enumerate(cands_info):
            if not cands:
                continue
            feats = torch.tensor(
                [[c[2], c[3], c[4], c[5], c[6], c[7]] for c in cands],
                dtype=torch.float32, device=h.device)
            hu = h[b, [c[0] for c in cands]]
            hv = h[b, [c[1] for c in cands]]
            inp = torch.cat([hu, hv, feats], dim=-1)
            logits[b, :len(cands)] = self.score_mlp(inp).squeeze(-1)
        return logits

    # -------------------------------------------------------- rollout
    def forward(self, envs, rem_adjs, feats, mode="sample"):
        """一批环境上的一步决策。

        envs: list[RoutingEnv]（同批共享决策）;
        rem_adjs/feats: 每环境的程序图状态（由 trainer 提供）。
        返回 (actions [(a,b)], logp [B])。
        """
        B = len(envs)
        h = self.encode_qubits(rem_adjs, feats)
        n_log = h.shape[1]
        # 扩展一个"空位置"节点（无逻辑比特的物理位置）
        h = torch.cat([h, self.empty_emb.view(1, 1, -1).expand(B, 1, -1)],
                      dim=1)
        cands_info = []
        for env in envs:
            cands = env.candidates()
            info = []
            inv = {p: q for q, p in env.pos.items()}
            for (a, b) in cands:
                u = inv.get(a, n_log)   # 空位置 -> n_log（扩展节点）
                v = inv.get(b, n_log)
                benefit = env.swap_benefit(a, b)
                pot_delta = env.swap_pot_delta(a, b)
                decay = env.swap_decay(a, b)
                ext_b, ext_p = env.swap_ext_features(a, b)
                d = int(env.dist[a, b])
                info.append((u, v, float(benefit), float(pot_delta),
                             float(decay), float(d),
                             float(ext_b), float(ext_p)))
            cands_info.append(info)
        logits = self.score_candidates(h, cands_info)  # [B, C]
        if mode == "greedy":
            idx = logits.argmax(-1)
            logp = torch.zeros(B, device=logits.device)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample()
            logp = dist.log_prob(idx)
        actions = []
        for b, env in enumerate(envs):
            cands = env.candidates()
            i = int(idx[b].item())
            i = min(i, len(cands) - 1) if cands else 0
            actions.append(cands[i] if cands else (0, 0))
        return actions, logp
