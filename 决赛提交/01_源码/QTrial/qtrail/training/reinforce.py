"""REINFORCE training with a self-critic greedy rollout baseline.

Paper: REINFORCE + "greedy rollout" baseline (Kool et al.), batch 512,
Adam lr 3e-4, sparse terminal reward. The paper's validation-set mean
baseline is available as cfg.train.baseline='val_mean' for ablations.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from qtrail.config import CHECKPOINTS_DIR, LOGS_DIR, Config
from qtrail.devices.spec import DeviceSpec
from qtrail.models import QAPolicy
from qtrail.problems import collate_instances
from qtrail.training.datasets import EpisodeSource, bucket_batches


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(policy: QAPolicy, source: EpisodeSource,
             noise_lambda: float, n_val: int, dev) -> float:
    """Mean normalized greedy cost on the validation set (多设备)."""
    policy.eval()
    episodes = source.val_episodes()[:n_val]
    total = 0.0
    for idxs, batch_spec in bucket_batches(episodes, 128):
        graphs = [episodes[i][0] for i in idxs]
        batch = collate_instances(graphs, batch_spec,
                                  noise_lambda=noise_lambda).to(dev)
        _, reward, _ = policy(batch, mode="greedy")
        total += float((-reward).sum())
    policy.train()
    return total / max(len(episodes), 1)


def train(cfg: Config, spec, *, out_dir: Path | str | None = None,
          resume: str | None = None, dev: str | None = None,
          epochs_override: int | None = None) -> dict:
    """Run the training loop; returns final stats dict.

    spec: 单个 DeviceSpec，或 DeviceSpec 列表（多拓扑混合训练）。
    """
    device_pool = list(spec) if isinstance(spec, (list, tuple)) else [spec]
    max_n = max(s.n for s in device_pool)

    out_dir = Path(out_dir) if out_dir else CHECKPOINTS_DIR
    logs_dir = LOGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    tcfg = cfg.train
    set_seed(tcfg.seed)
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    noise_lambda = cfg.device.noise.lambda_n if cfg.reward.mode in ("noise", "combined") else 0.0

    # ---- model
    if resume:
        policy, ckpt = QAPolicy.load_checkpoint(resume, device_n=max_n,
                                                map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[train] resumed from {resume} at epoch {start_epoch}")
    else:
        policy = QAPolicy(cfg.model, max_n, reward_cfg=cfg.reward,
                          graph_cfg=cfg.graph).to(device)
        start_epoch = 1
    baseline = deepcopy(policy)
    baseline.eval()

    n_params = sum(p.numel() for p in policy.parameters())
    names = ",".join(s.name for s in device_pool)
    print(f"[train] encoder={cfg.model.encoder} devices={names} dev={dev} "
          f"params={n_params/1e6:.2f}M")

    optimizer = torch.optim.Adam(policy.parameters(), lr=tcfg.lr, weight_decay=1e-6)
    if resume and "optimizer" in (ckpt if resume else {}):
        optimizer.load_state_dict(ckpt["optimizer"])

    total_epochs = epochs_override or tcfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=tcfg.lr_min)

    # ---- episode source (circuit pool optional)
    pool = None
    pool_path = Path(__file__).resolve().parent.parent.parent / tcfg.pool
    if pool_path.exists() and tcfg.dataset_mix.get("circuit", 0) > 0:
        from qtrail.utils.bench import load_graph_pool
        pool = load_graph_pool(pool_path)
        print(f"[train] circuit pool: {len(pool)} graphs")
    source = EpisodeSource(tcfg, device_pool, graph_pool=pool)

    base_name = device_pool[0].name if len(device_pool) == 1 else \
        f"multi{len(device_pool)}topo"
    name = f"{base_name}_{cfg.model.encoder}_{cfg.reward.mode}_{cfg.model.device_features}"
    if cfg.reward.depth_lambda > 0:
        name += f"_dep{cfg.reward.depth_lambda}"
    if cfg.reward.compactness_lambda > 0:
        name += f"_c{cfg.reward.compactness_lambda}"
    if cfg.graph.temporal_alpha > 0:
        name += f"_t{cfg.graph.temporal_alpha}"
    if cfg.reward.oracle_mode == "routing":
        name += f"_oracle{cfg.reward.oracle_max_n}"
        if cfg.reward.oracle_noise_lambda > 0:
            name += f"_n{cfg.reward.oracle_noise_lambda}"
    log_path = logs_dir / f"train_{name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    best_val = float("inf")
    patience = 0
    stats = {"epochs": 0, "best_val": None}

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        episodes = source.sample_episodes(tcfg.episodes_per_epoch)
        total_loss = 0.0
        n_batches = 0
        grad_norm = 0.0

        policy.train()
        cm_cache = {}
        use_oracle = cfg.reward.oracle_mode == "routing"
        for idxs, batch_spec in bucket_batches(episodes, tcfg.batch_size):
            graphs = [episodes[i][0] for i in idxs]
            max_n = max((g.n for g in graphs), default=1)
            # 大 N 显存保护：批大小按 (128/n_max)^2 缩放
            b_eff = max(8, int(tcfg.batch_size * (128.0 / max_n) ** 2))
            b_eff = min(b_eff, tcfg.batch_size)
            for s in range(0, len(graphs), b_eff):
                sub = graphs[s:s + b_eff]
                batch = collate_instances(sub, batch_spec,
                                          noise_lambda=noise_lambda).to(device)
                logp, reward_s, pi_s = policy(batch, mode="sample")

                if use_oracle:
                    from qtrail.training.oracle import oracle_rewards
                    cm = cm_cache.get(id(batch_spec))
                    if cm is None:
                        from qtrail.pipeline.routing import coupling_map_from_spec
                        cm = coupling_map_from_spec(batch_spec)
                        cm_cache[id(batch_spec)] = cm
                    reward_s = oracle_rewards(
                        sub, pi_s, cm, seed=tcfg.seed,
                        oracle_max_n=cfg.reward.oracle_max_n,
                        cap=cfg.reward.oracle_weight_cap,
                        static_rewards=reward_s,
                        noise_lambda=cfg.reward.oracle_noise_lambda,
                        dist=batch_spec.dist, noise_dist=batch_spec.noise_dist)

                with torch.no_grad():
                    _, reward_b, pi_b = baseline(batch, mode="greedy")
                    if use_oracle:
                        reward_b = oracle_rewards(
                            sub, pi_b, cm, seed=tcfg.seed,
                            oracle_max_n=cfg.reward.oracle_max_n,
                            cap=cfg.reward.oracle_weight_cap,
                            static_rewards=reward_b,
                            noise_lambda=cfg.reward.oracle_noise_lambda,
                            dist=batch_spec.dist, noise_dist=batch_spec.noise_dist)

                adv = reward_s - reward_b.detach()
                loss = -(adv * logp).mean()
                if tcfg.entropy_beta > 0:
                    ent = policy.last_entropy
                    loss = loss - tcfg.entropy_beta * ent

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                                           tcfg.grad_clip).item()
                optimizer.step()
                total_loss += float(loss.detach())
                n_batches += 1

        scheduler.step()
        # hard-copy baseline (self-critic)
        baseline.load_state_dict(policy.state_dict())

        val_cost = evaluate(policy, source, noise_lambda, tcfg.val_episodes,
                            device) if epoch % tcfg.log_every == 0 else None
        wall = time.time() - t0

        if val_cost is not None:
            improved = val_cost < best_val - 1e-6
            if improved:
                best_val = val_cost
                patience = 0
                policy.save_checkpoint(out_dir / f"{name}_best.pt", epoch=epoch,
                                       val_cost=val_cost, optimizer=optimizer)
            else:
                patience += 1
            stats["best_val"] = best_val

        if epoch % tcfg.save_every == 0:
            policy.save_checkpoint(out_dir / f"{name}_last.pt", epoch=epoch,
                                   val_cost=val_cost if val_cost is not None else 0.0,
                                   optimizer=optimizer)

        lr = optimizer.param_groups[0]["lr"]
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch, "loss": total_loss / max(n_batches, 1),
                "val_cost": val_cost, "lr": lr, "grad_norm": grad_norm,
                "wall_s": round(wall, 2),
            }) + "\n")
        print(f"epoch {epoch:4d}/{total_epochs} | loss {total_loss/max(n_batches,1):8.3f} | "
              f"val {val_cost if val_cost is None else round(val_cost,3)} | "
              f"lr {lr:.2e} | grad {grad_norm:6.2f} | {wall:5.1f}s")

        if patience >= tcfg.early_stop_patience:
            print(f"[train] early stop at epoch {epoch} (patience {patience})")
            break

    stats["epochs"] = epoch
    stats["best_val"] = best_val if best_val < float("inf") else None
    stats["best_path"] = str(out_dir / f"{name}_best.pt")
    stats["log_path"] = str(log_path)
    print(f"[train] done: {stats}")
    return stats
