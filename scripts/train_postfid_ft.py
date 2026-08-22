"""奖励重训闭环（微调）：终局奖励 = 后处理管线（target_post opt=1）的真实
保真度对数。

从当前最优检查点（c0.05）微调——保留噪声规避先验，防止重训后效果下降；
新检查点独立命名（绝不覆盖原权重）；验收门禁 = 简单 A/B 胜出后才采用。

REINFORCE + 自批评贪心基线；优势标准化；早停（patience 15）。
训练集：MQTBench ≤25 比特 + QUEKO dense/sparse（共 ~60 条，采样替代）。
"""
from __future__ import annotations

import json
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
from qtrail.pipeline.metrics import estimate_fidelity
from qtrail.pipeline.routing import (coupling_map_from_spec,
                                     decompose_to_platform)
from qtrail.pipeline.target_post import route_target_post
from qtrail.problems import build_program_graph
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import (extract_ops, load_qasm2, sanitize_qasm,
                                  strip_measurements)

BASE_CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OUT_CKPT = ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_"
            "postfid_ft_best.pt")
CACHE = Path("data/mqtbench/stratified")
LOG_FLOOR = math.log(1e-30)


def post_reward(qc, pi, spec, cm, seed, opt=1):
    """布局 → target_post 管线（opt 级）→ 物理分解 → 保真度对数奖励。"""
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


def load_circuits():
    """训练/验证线路池（≤25 比特，共 ~60 条）。"""
    from qiskit import qasm2
    out = []
    for p in sorted(CACHE.glob("*.qasm")):
        size = int(p.stem.rsplit("_", 1)[1])
        if size <= 25 and p.stem != "qwalk_25":
            out.append((p.stem, qasm2.loads(
                sanitize_qasm(p.read_text(encoding="utf-8")))))
    for p in iter_queko_files("BIGD"):
        if ".0D1_.1D2_" in p.name or ".3D1_.4D2_" in p.name:
            out.append((p.stem, load_qasm2(p)))
    return out


def main():
    import torch as t

    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    cm = coupling_map_from_spec(spec)

    device = "cuda" if t.cuda.is_available() else "cpu"
    policy, ckpt = QAPolicy.load_checkpoint(BASE_CKPT, device_n=spec.n,
                                            map_location=device)
    policy.train()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg  # 图表示与训练一致

    pool = load_circuits()
    rng = np.random.default_rng(0)
    n_train = int(len(pool) * 0.85)
    train_pool = pool[:n_train]
    val_pool = pool[n_train:]
    print(f"train {len(train_pool)} / val {len(val_pool)} circuits", flush=True)

    import copy as _copy
    baseline = _copy.deepcopy(policy)  # 自批评基线（深拷贝，保持全部配置）
    baseline.eval()

    opt = t.optim.Adam(policy.parameters(), lr=1e-4)  # 微调：小学习率
    best_val = float("-inf")
    patience_left = 15
    noise_lambda = (cfg.device.noise.lambda_n
                    if cfg.reward.mode in ("noise", "combined") else 0.0)

    log_lines = []
    for epoch in range(1, 81):
        t0 = time.time()
        idxs = rng.integers(0, len(train_pool), 64)
        losses = []
        for batch_start in range(0, 64, 32):
            batch_idx = idxs[batch_start:batch_start + 32]
            graphs = []
            qcs = []
            for i in batch_idx:
                name, qc = train_pool[i]
                qc_c, _ = strip_measurements(qc)
                ops = extract_ops(qc_c)
                g = build_program_graph(qc_c.num_qubits, ops, circuit_id=name,
                                        temporal_alpha=cfg.graph.temporal_alpha)
                graphs.append(g)
                qcs.append(qc_c)
            from qtrail.search.decoding import collate_instances
            batch = collate_instances(graphs, spec,
                                      noise_lambda=noise_lambda).to(device)
            logp, _, pi_s = policy(batch, mode="sample")
            # 采样轨迹的后处理奖励
            r_s = []
            for b, (qc, g) in enumerate(zip(qcs, graphs)):
                r_s.append(post_reward(qc, pi_s[b, :g.n], spec, cm, seed=epoch))
            with t.no_grad():
                _, _, pi_b = baseline(batch, mode="greedy")
            r_b = []
            for b, (qc, g) in enumerate(zip(qcs, graphs)):
                r_b.append(post_reward(qc, pi_b[b, :g.n], spec, cm, seed=epoch))
            adv = t.tensor(r_s, device=device) - t.tensor(r_b, device=device)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # 优势标准化
            loss = -(adv * logp).mean()
            opt.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        baseline.load_state_dict(policy.state_dict())

        if epoch % 3 == 0:
            with t.no_grad():
                vals = []
                for name, qc in val_pool:
                    qc_c, _ = strip_measurements(qc)
                    ops = extract_ops(qc_c)
                    g = build_program_graph(
                        qc_c.num_qubits, ops, circuit_id=name,
                        temporal_alpha=cfg.graph.temporal_alpha)
                    from qtrail.search.decoding import collate_instances as ci
                    batch = ci([g], spec, noise_lambda=noise_lambda).to(device)
                    _, _, pi_b = policy(batch, mode="greedy")
                    vals.append(post_reward(qc_c, pi_b[0, :g.n], spec, cm,
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
            log_lines.append({"epoch": epoch, "loss": float(np.mean(losses)),
                              "val": vmean, "best": best_val})
            if patience_left <= 0:
                print(f"[postfid_ft] early stop at epoch {epoch} "
                      f"(best val {best_val:.2f})", flush=True)
                break
        else:
            print(f"epoch {epoch:3d} | loss {np.mean(losses):7.3f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
    with open("logs/postfid_ft.jsonl", "a", encoding="utf-8") as f:
        for line in log_lines:
            f.write(json.dumps(line) + "\n")
    print(f"[postfid_ft] done, best val {best_val:.2f} -> {OUT_CKPT}")


if __name__ == "__main__":
    main()
