"""多任务微调（用户确认方案）：扩展池 + 混合目标 + 分组优势标准化 + 设备轮换。

每批：70% 后处理保真度实例（≤50 比特真实线路，reward = target_post opt=1
保真度对数）+ 30% 静态代价实例（51-433 比特大图，保住大 N 静态先验、
防灾难性遗忘）——两组各自标准化优势后合并。

设备轮换：每 epoch 在天衍-287 / 6×6 / heavy-hex-115 / Sycamore-53 间轮换
（与 multi7topo 训练同机制，避免设备特异性过拟合）。

从 c0.05 检查点起步；新检查点独立命名 postfid_mt_best.pt（原权重不动）。
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.devices.architectures import (build_grid_family_spec,
                                          build_heavyhex_spec,
                                          build_sycamore53_spec)
from qtrail.models import QAPolicy
from qtrail.pipeline.metrics import estimate_fidelity
from qtrail.pipeline.routing import (coupling_map_from_spec,
                                     decompose_to_platform)
from qtrail.pipeline.target_post import route_target_post
from qtrail.problems import ProgramGraph, build_program_graph
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import (extract_ops, load_qasm2, sanitize_qasm,
                                  strip_measurements)

BASE_CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OUT_CKPT = ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_"
            "postfid_mt_best.pt")
LOG_FLOOR = math.log(1e-30)


def post_reward(qc, pi, spec, cm, seed, opt=1):
    layout = {i: int(pi[i]) for i in range(qc.num_qubits)}
    try:
        routed = route_target_post(qc, spec, layout, seed=seed,
                                   optimization_level=opt)
        routed = decompose_to_platform(routed, cm, optimization_level=1,
                                       seed=seed)
        fid = estimate_fidelity(routed, spec.calib)
    except Exception:
        return LOG_FLOOR
    return math.log(max(float(fid), 1e-30))


def load_pools():
    """post 池（≤50 比特真实线路）+ static 池（51-433 大图，纯静态代价）。"""
    from qiskit import qasm2
    post = []   # (name, QuantumCircuit)
    static = []  # ProgramGraph
    seen = set()

    v4 = pickle.load(open("data/mqtbench/graph_pool_v4.pkl", "rb"))
    for g in v4:
        if g.n > 50:
            static.append(g)
            continue
        meta = getattr(g, "ops_meta", None)
        qasm = meta.get("qasm") if meta else None
        if not qasm or g.circuit_id in seen:
            continue
        seen.add(g.circuit_id)
        try:
            qc = qasm2.loads(sanitize_qasm(qasm))
            qc, _ = strip_measurements(qc)
            if 0 < qc.num_qubits <= 50:
                post.append((g.circuit_id, qc))
        except Exception:
            continue

    for p in iter_queko_files("BIGD"):
        if p.stem in seen:
            continue
        seen.add(p.stem)
        qc = load_qasm2(p)
        qc, _ = strip_measurements(qc)
        post.append((p.stem, qc))

    for p in Path("data/QASMBench").rglob("*.qasm"):
        if p.stem in seen:
            continue
        try:
            qc = qasm2.loads(sanitize_qasm(p.read_text(encoding="utf-8")))
            qc, _ = strip_measurements(qc)
            if not (0 < qc.num_qubits <= 50):
                continue
            if any(len(i.qubits) > 2 for i in qc.data
                   if i.operation.name not in ("barrier", "measure")):
                continue  # 多比特门线路（syndrome 等）跳过
            seen.add(p.stem)
            post.append((p.stem, qc))
        except Exception:
            continue

    # 大图池去重 + 上限（保证 epoch 采样速度）
    static = static[:600]
    return post, static


def main():
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec0 = build_tianyan287_spec(dev_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, ckpt = QAPolicy.load_checkpoint(BASE_CKPT, device_n=spec0.n,
                                            map_location=device)
    policy.train()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    post_pool, static_pool = load_pools()
    rng = np.random.default_rng(0)
    n_post_val = max(4, len(post_pool) // 10)
    val_post = post_pool[-n_post_val:]
    train_post = post_pool[:-n_post_val]
    print(f"post train {len(train_post)} / val {len(val_post)} | "
          f"static {len(static_pool)}", flush=True)

    import copy as _copy
    baseline = _copy.deepcopy(policy)
    baseline.eval()

    topologies = [
        build_tianyan287_spec(dev_cfg),
        build_grid_family_spec(6, 6, seed=0),
        build_heavyhex_spec(distance=7, seed=0),
        build_sycamore53_spec(seed=0),
    ]
    noise_lambda = (cfg.device.noise.lambda_n
                    if cfg.reward.mode in ("noise", "combined") else 0.0)

    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    best_val = float("-inf")
    patience_left = 15

    from qtrail.search.decoding import collate_instances

    for epoch in range(1, 61):
        t0 = time.time()
        spec = topologies[(epoch - 1) % len(topologies)]
        cm = coupling_map_from_spec(spec)
        n_post = 64
        n_static = 24
        p_idx = rng.integers(0, len(train_post), n_post)
        s_idx = rng.integers(0, len(static_pool), n_static)

        losses = []
        # ---- 后处理保真度组（真实路由）
        post_graphs, post_qcs = [], []
        for i in p_idx:
            name, qc = train_post[i]
            ops = extract_ops(qc)
            g = build_program_graph(qc.num_qubits, ops, circuit_id=name,
                                    temporal_alpha=cfg.graph.temporal_alpha)
            post_graphs.append(g)
            post_qcs.append(qc)
        batch_p = collate_instances(post_graphs, spec,
                                    noise_lambda=noise_lambda).to(device)
        logp_p, _, pi_p = policy(batch_p, mode="sample")
        rp_s = [post_reward(qc, pi_p[b, :g.n], spec, cm, seed=epoch)
                for b, (qc, g) in enumerate(zip(post_qcs, post_graphs))]
        with torch.no_grad():
            _, _, pi_pb = baseline(batch_p, mode="greedy")
        rp_b = [post_reward(qc, pi_pb[b, :g.n], spec, cm, seed=epoch)
                for b, (qc, g) in enumerate(zip(post_qcs, post_graphs))]
        adv_p = (torch.tensor(rp_s, device=device)
                 - torch.tensor(rp_b, device=device))
        adv_p = (adv_p - adv_p.mean()) / (adv_p.std() + 1e-8)

        # ---- 静态代价组（大 N 图，无路由；用策略内置终局奖励）
        static_graphs = [static_pool[i] for i in s_idx]
        max_n = max(g.n for g in static_graphs)
        b_eff = max(8, int(n_static * (128.0 / max_n) ** 2))
        b_eff = min(b_eff, n_static)
        adv_s_parts, logp_s_parts = [], []
        for s in range(0, n_static, b_eff):
            sub = static_graphs[s:s + b_eff]
            batch_s = collate_instances(sub, spec,
                                        noise_lambda=noise_lambda).to(device)
            logp_s, reward_s, _ = policy(batch_s, mode="sample")
            with torch.no_grad():
                _, reward_b, _ = baseline(batch_s, mode="greedy")
            adv_s = reward_s - reward_b.detach()
            adv_s = (adv_s - adv_s.mean()) / (adv_s.std() + 1e-8)
            adv_s_parts.append(adv_s)
            logp_s_parts.append(logp_s)
        adv_s = torch.cat(adv_s_parts)
        logp_s = torch.cat(logp_s_parts)

        loss = -(torch.cat([adv_p, adv_s]) * torch.cat([logp_p, logp_s])).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        baseline.load_state_dict(policy.state_dict())
        losses.append(float(loss))

        if epoch % 3 == 0:
            with torch.no_grad():
                vals = []
                for name, qc in val_post:
                    ops = extract_ops(qc)
                    g = build_program_graph(
                        qc.num_qubits, ops, circuit_id=name,
                        temporal_alpha=cfg.graph.temporal_alpha)
                    batch = collate_instances([g], spec0,
                                              noise_lambda=noise_lambda).to(device)
                    _, _, pi_b = policy(batch, mode="greedy")
                    vals.append(post_reward(qc, pi_b[0, :g.n], spec0,
                                            coupling_map_from_spec(spec0),
                                            seed=0))
                vmean = float(np.mean(vals))
            if vmean > best_val + 1e-6:
                best_val = vmean
                patience_left = 15
                policy.save_checkpoint(OUT_CKPT, epoch=epoch, val_cost=-vmean,
                                       optimizer=opt, rng=None)
            else:
                patience_left -= 3
            print(f"epoch {epoch:3d} | loss {np.mean(losses):7.3f} | "
                  f"val {vmean:7.2f} (best {best_val:7.2f}) | "
                  f"patience {patience_left} | {time.time() - t0:.0f}s",
                  flush=True)
            if patience_left <= 0:
                print(f"[postfid_mt] early stop at epoch {epoch} "
                      f"(best val {best_val:.2f})", flush=True)
                break
        else:
            print(f"epoch {epoch:3d} | loss {np.mean(losses):7.3f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"[postfid_mt] done, best val {best_val:.2f} -> {OUT_CKPT}")


if __name__ == "__main__":
    main()
