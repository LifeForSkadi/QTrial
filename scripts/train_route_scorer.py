# -*- coding: utf-8 -*-
"""学习式路由评分器训练（2026-11-10）。

设计：只学"候选交换边的评分函数"（RouteScorer MLP），SABRE 骨架保留。
训练 = ① 回归暖启动（模仿手写线性评分，几百步）→ ② REINFORCE +
自批评基线微调（采样 vs 贪心基线，终局奖励 = -swap 数）。
线路池 = 自生成 native287 + queko287（与评测集零重叠，按规模分桶采样）。

验收（贪心评估，平凡布局）：诊断集 swap 总数低于前瞻基线
（gf2^16 1371 + gf2^32 6477 + dtc_100 4488 = 10336），不达标弃用。
输出：checkpoints/rl_route_scorer_{best,last}.pt + logs/。
"""
import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.rl_score import RouteScorer
from qtrail.pure.router import sabre_route

POOLS = [Path("data/tianyan287/queko287"),
         Path("data/tianyan287/native287")]
DIAG = [("data/benchpress/qasm/feynman/gf2^16_mult.qasm", 1371),
        ("data/benchpress/qasm/feynman/gf2^32_mult.qasm", 6477),
        ("data/benchpress/qasm/dtc/dtc_100_cx_12345.qasm", 4488)]
DIAG_FILES = {Path(p).name for p, _ in DIAG}
BP = Path("data/benchpress/qasm")  # 难路由线路补充池（排除诊断集 3 条）
CKPT_DIR = Path("checkpoints")
LOG = Path("logs/train_route_scorer.jsonl")
MAX_SWAPS_TRAIN = 30000
MAX_SWAPS_EVAL = 100000


class Collector(RouteScorer):
    """暖启动数据收集器：只记录 (F, lin)，不实际评分。"""

    def __init__(self, pairs):
        super().__init__()
        self.pairs = pairs

    def forward(self, F, lin=None):
        self.pairs.append((F.numpy().copy(), lin.numpy().copy()))
        return lin  # 手写评分直接作为选择依据（行为与无 scorer 一致）


def build_circuit_pool():
    """按规模分桶的线路清单 [(path, size)]（生成池 + benchpress 难路由
    补充，排除诊断集 3 条）。"""
    buckets = {20: [], 40: [], 60: [], 80: [], 105: []}
    for pool in POOLS:
        m = pool / "manifest.jsonl"
        if not m.exists():
            continue
        for line in m.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            size = int(r.get("size", 0))
            p = pool / "qasm" / r["file"]
            if p.exists():
                buckets.setdefault(size, []).append(p)
    for p in sorted(BP.rglob("*.qasm")):
        if p.name in DIAG_FILES:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        try:
            circ = parse_qasm(text)
        except Exception:
            continue
        if 10 <= circ.n <= 100:
            # 按最近规模桶归类（生成池桶键粒度）
            key = min((k for k in buckets if circ.n <= k), default=105)
            buckets[key].append(p)
    return buckets


def sample_circuit(buckets, rng):
    # 规模权重：小线路快、状态多样；benchpress 难线路与生成池均衡
    sizes = [20, 20, 40, 40, 60, 60, 80, 80, 105]
    while True:
        s = rng.choice(sizes)
        lst = buckets.get(s)
        if lst:
            return rng.choice(lst), s


def run_episode(scorer, spec, circ, seed, sample, temp=1.0,
                stats=None, max_swaps=MAX_SWAPS_TRAIN):
    routed, swaps, _ = sabre_route(
        circ, spec, {i: i for i in range(circ.n)}, seed=seed,
        scorer=scorer, scorer_sample=sample, scorer_temp=temp,
        scorer_stats=stats, max_swaps=max_swaps)
    return swaps


def diag_eval(scorer, spec):
    """诊断集贪心评估：返回 (各条 swap, 总数)。"""
    res = []
    for path, baseline in DIAG:
        circ = parse_qasm(Path(path).read_text(encoding="utf-8"))
        _, swaps, _ = sabre_route(circ, spec, {i: i for i in range(circ.n)},
                                  seed=42, scorer=scorer, scorer_sample=False)
        res.append((Path(path).name, swaps, baseline))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--warm", type=int, default=600,
                    help="暖启动回归步数（模仿手写评分）")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--ent-beta", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = args.dev

    spec = build_tianyan287_spec(load_device_config())
    buckets = build_circuit_pool()
    total = {s: len(v) for s, v in buckets.items()}
    print(f"[pool] {total}", flush=True)
    rng = random.Random(args.seed)

    scorer = RouteScorer().to(dev)
    if args.resume:
        scorer.load_state_dict(torch.load(args.resume, map_location=dev))
        print(f"[resume] {args.resume}")
    baseline = deepcopy(scorer)
    baseline.eval()

    # ---------------- ① 回归暖启动：模仿手写线性评分
    pairs = []
    collector = Collector(pairs)
    n_collect = 0
    print("[warm] collecting (F, lin) pairs...", flush=True)
    t0 = time.time()
    while len(pairs) < 20000 and time.time() - t0 < 600:
        p, s = sample_circuit(buckets, rng)
        try:
            circ = parse_qasm(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_episode(collector, spec, circ, seed=0, sample=False)
        n_collect += 1
    print(f"[warm] {len(pairs)} pairs from {n_collect} circuits "
          f"({time.time()-t0:.0f}s)", flush=True)
    F_all = torch.from_numpy(np.concatenate([f for f, _ in pairs])).to(dev)
    T_all = torch.from_numpy(np.concatenate([t for _, t in pairs])).to(dev)
    # 目标标准化（z-score）+ 特征标准化：尺度无关的暖启动
    t_mean = T_all.mean()
    t_std = T_all.std().clamp(min=1e-6)
    T_all = (T_all - t_mean) / t_std
    f_mean = F_all.mean(dim=0, keepdim=True)
    f_std = F_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    F_all = (F_all - f_mean) / f_std
    opt = torch.optim.Adam(scorer.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()
    scorer.train()
    idx = np.random.permutation(len(pairs))
    for step in range(args.warm):
        b = idx[(step * 512) % len(idx):(step * 512) % len(idx) + 512]
        out = scorer(F_all[b])
        loss = lossf(out, T_all[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"  warm {step}/{args.warm} loss={float(loss.detach()):.3f}",
                  flush=True)
    print("[warm] done", flush=True)
    baseline.load_state_dict(scorer.state_dict())

    # ---------------- ② REINFORCE 微调
    opt = torch.optim.Adam(scorer.parameters(), lr=args.lr)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_sum = None
    best_swaps_log = {}
    for epoch in range(1, args.epochs + 1):
        scorer.train()
        grads = []
        total_loss = 0.0
        n_ep = 0
        sw_s_list, sw_b_list = [], []
        for _ in range(args.episodes):
            p, s = sample_circuit(buckets, rng)
            try:
                circ = parse_qasm(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            seed = random.randint(0, 10 ** 9)
            stats_s = {"logp": 0.0, "entropy": 0.0}
            sw_s = run_episode(scorer, spec, circ, seed, sample=True,
                               temp=args.temp, stats=stats_s)
            if sw_s >= MAX_SWAPS_TRAIN:
                continue  # 触顶样本无意义
            if "logp_t" not in stats_s:
                continue  # 零交换决策线路（无需路由）：无梯度信号
            with torch.no_grad():
                sw_b = run_episode(baseline, spec, circ, seed, sample=False)
            adv = float(sw_b - sw_s)
            loss = -(adv * stats_s["logp_t"]) \
                - args.ent_beta * stats_s["ent_t"]
            loss.backward()
            grads.append(1)
            total_loss += float(loss.detach())
            n_ep += 1
            sw_s_list.append(sw_s)
            sw_b_list.append(sw_b)
            if len(grads) >= 8:
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
                opt.step()
                opt.zero_grad()
                grads = []
        if grads:
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
            opt.step()
            opt.zero_grad()
        baseline.load_state_dict(scorer.state_dict())

        # 评估（每 5 epoch 或首末）
        res = None
        if epoch % 5 == 0 or epoch in (1, args.epochs):
            scorer.eval()
            with torch.no_grad():
                res = diag_eval(scorer, spec)
            scorer.train()
            total_sw = sum(r[1] for r in res)
            if best_sum is None or total_sw < best_sum:
                best_sum = total_sw
                torch.save(scorer.state_dict(),
                           CKPT_DIR / "rl_route_scorer_best.pt")
                best_swaps_log = {r[0]: (r[1], r[2]) for r in res}
            torch.save(scorer.state_dict(), CKPT_DIR / "rl_route_scorer_last.pt")
        row = {
            "epoch": epoch,
            "loss": round(total_loss / max(n_ep, 1), 3),
            "sw_s_mean": round(float(np.mean(sw_s_list)), 1) if sw_s_list else None,
            "sw_b_mean": round(float(np.mean(sw_b_list)), 1) if sw_b_list else None,
            "diag": None if res is None else
            {r[0]: r[1] for r in res},
            "best_sum": best_sum,
            "wall_s": None,
        }
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"epoch {epoch:3d}/{args.epochs} loss={row['loss']} "
              f"sw_s={row['sw_s_mean']} sw_b={row['sw_b_mean']} "
              f"diag={row['diag']} best={best_sum}",
              flush=True)
    print(f"[done] best diag sum = {best_sum} (基线 10336) "
          f"{'达标' if best_sum is not None and best_sum < 10336 else '未达标'}")
    print(f"[best] {best_swaps_log}")


if __name__ == "__main__":
    main()
