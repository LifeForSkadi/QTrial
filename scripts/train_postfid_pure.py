"""postfid_pure 微调：终局奖励 = pure 管线（自研路由器+后处理+分解）真实
保真度对数——训练闭环与交付管线完全对齐，训练环节也零 qiskit 奖励计算。

REINFORCE + 自批评贪心基线；优势标准化；早停（patience 15）。
从 c0.05 检查点微调；新检查点独立命名 postfid_pure_best.pt。
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.metrics import estimate_fidelity as pure_fidelity
from qtrail.pure.post import decompose_to_platform, post_route
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route
from qtrail.problems import build_program_graph
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import (extract_ops, sanitize_qasm,
                                  strip_measurements)

BASE_CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OUT_CKPT = ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_"
            "postfid_pure_best.pt")
CACHE = Path("data/mqtbench/stratified")
LOG_FLOOR = math.log(1e-30)


def pure_reward(pure_c, pi, spec, seed):
    """布局 → 自研路由器 → 后处理栈 → 平台基分解 → 保真度对数奖励。"""
    layout = {i: int(pi[i]) for i in range(pure_c.n)}
    try:
        routed, swaps, final = sabre_route(pure_c, spec, layout, seed=seed)
        routed, _ = post_route(routed, final)
        final_qc = decompose_to_platform(routed)
        fid = pure_fidelity(final_qc, spec.calib)
    except Exception:
        return LOG_FLOOR
    return math.log(max(float(fid), 1e-30))


def load_pool(temporal_alpha):
    """一次构建：name / qiskit 线路（图构建用）/ pure Circuit（奖励用）。
    返回 [(name, qc, pure_circ)]。"""
    import re
    from qiskit import qasm2
    post = []
    seen = set()
    for p in sorted(CACHE.glob("*.qasm")):
        size = int(p.stem.rsplit("_", 1)[1])
        if size > 25 or p.stem == "qwalk_25":
            continue
        seen.add(p.stem)
        text = sanitize_qasm(p.read_text(encoding="utf-8"))
        qc = qasm2.loads(text)
        qc, _ = strip_measurements(qc)
        post.append((p.stem, qc, parse_qasm(text)))
    for p in iter_queko_files("BIGD"):
        if not (".0D1_.1D2_" in p.name or ".3D1_.4D2_" in p.name):
            continue
        if p.stem in seen:
            continue
        seen.add(p.stem)
        text = sanitize_qasm(p.read_text(encoding="utf-8"))
        qc = qasm2.loads(text)
        qc, _ = strip_measurements(qc)
        post.append((p.stem, qc, parse_qasm(text)))
    return post


def main():
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, ckpt = QAPolicy.load_checkpoint(BASE_CKPT, device_n=spec.n,
                                            map_location=device)
    policy.train()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    pool = load_pool(cfg.graph.temporal_alpha)
    rng = np.random.default_rng(0)
    n_val = max(4, len(pool) // 10)
    val_pool = pool[-n_val:]
    train_pool = pool[:-n_val]
    print(f"train {len(train_pool)} / val {len(val_pool)}", flush=True)

    import copy as _copy
    baseline = _copy.deepcopy(policy)
    baseline.eval()

    noise_lambda = (cfg.device.noise.lambda_n
                    if cfg.reward.mode in ("noise", "combined") else 0.0)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    best_val = float("-inf")
    patience_left = 15

    from qtrail.search.decoding import collate_instances

    def graphs_for(items):
        return [build_program_graph(qc.num_qubits, extract_ops(qc),
                                    circuit_id=name,
                                    temporal_alpha=cfg.graph.temporal_alpha)
                for name, qc, _ in items]

    for epoch in range(1, 61):
        t0 = time.time()
        idxs = rng.integers(0, len(train_pool), 64)
        losses = []
        for start in range(0, 64, 32):
            batch_idx = idxs[start:start + 32]
            items = [train_pool[i] for i in batch_idx]
            graphs = graphs_for(items)
            batch = collate_instances(graphs, spec,
                                      noise_lambda=noise_lambda).to(device)
            logp, _, pi_s = policy(batch, mode="sample")
            r_s = [pure_reward(pure_c, pi_s[b, :g.n], spec, epoch)
                   for b, ((_, _, pure_c), g) in enumerate(zip(items, graphs))]
            with torch.no_grad():
                _, _, pi_b = baseline(batch, mode="greedy")
            r_b = [pure_reward(pure_c, pi_b[b, :g.n], spec, epoch)
                   for b, ((_, _, pure_c), g) in enumerate(zip(items, graphs))]
            adv = torch.tensor(r_s, device=device) - torch.tensor(r_b,
                                                                  device=device)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            loss = -(adv * logp).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        baseline.load_state_dict(policy.state_dict())

        if epoch % 3 == 0:
            with torch.no_grad():
                vals = []
                for _, qc, pure_c in val_pool:
                    g = build_program_graph(
                        qc.num_qubits, extract_ops(qc), circuit_id="v",
                        temporal_alpha=cfg.graph.temporal_alpha)
                    batch = collate_instances([g], spec,
                                              noise_lambda=noise_lambda).to(device)
                    _, _, pi_b = policy(batch, mode="greedy")
                    vals.append(pure_reward(pure_c, pi_b[0, :g.n], spec, 0))
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
                print(f"[postfid_pure] early stop at epoch {epoch} "
                      f"(best val {best_val:.2f})", flush=True)
                break
        else:
            print(f"epoch {epoch:3d} | loss {np.mean(losses):7.3f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"[postfid_pure] done, best val {best_val:.2f} -> {OUT_CKPT}")


if __name__ == "__main__":
    main()
