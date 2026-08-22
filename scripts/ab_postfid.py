"""奖励重训闭环验收门禁：新模型（postfid_ft）vs 当前模型（c0.05）。

两模型都走完整交付管线（target_post opt=3 + 8 种子 × top-3，fidelity 规则），
35 条线路（MQTBench ≤10 15 条 + ≤25 10 条 + QUEKO dense 10 条）逐线路对比。
胜出判据：新模型组均值保真度更高 且 逐线路胜场 > 负场。
用法: python scripts/ab_postfid.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2
import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.mapper import Mapper
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm, strip_measurements, load_qasm2

CACHE = Path("data/mqtbench/stratified")
CUR = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
NEW = ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_"
       "postfid_ft_best.pt")


def circuits():
    out = []
    for p in sorted(CACHE.glob("*.qasm")):
        size = int(p.stem.rsplit("_", 1)[1])
        if size <= 25 and p.stem != "qwalk_25":
            out.append((p.stem, qasm2.loads(
                sanitize_qasm(p.read_text(encoding="utf-8")))))
    for p in iter_queko_files("BIGD"):
        if ".3D1_.4D2_" in p.name:
            out.append((p.stem, load_qasm2(p)))
    return out


def run_model(ckpt_path, spec, cfg, circuits_all):
    policy, ckpt = QAPolicy.load_checkpoint(ckpt_path, device_n=spec.n,
                                            map_location="cuda")
    policy.eval()
    my_cfg = Config()
    my_cfg.device = cfg.device
    my_cfg.model = ckpt.get("model_cfg", cfg.model)
    my_cfg.graph = policy.graph_cfg
    rows = {}
    for name, qc in circuits_all:
        qc_c, _ = strip_measurements(qc)
        mapper = Mapper(spec, policy=policy, cfg=my_cfg, dev="cuda", seed=42,
                        selection_rule="fidelity", target_post=True,
                        target_post_opt=3, target_post_seeds=8,
                        target_post_top_per_seed=3)
        try:
            res = mapper.map_circuit(qc_c, circuit_id=name)
            rows[name] = {"swaps": res.swap_count,
                          "depth": res.metrics["depth"],
                          "fidelity": res.metrics["est_fidelity"]}
        except Exception as e:
            rows[name] = {"error": f"{type(e).__name__}: {str(e)[:50]}"}
    return rows


def main():
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    circuits_all = circuits()
    print(f"A/B: {len(circuits_all)} circuits", flush=True)

    t0 = time.time()
    cur = run_model(CUR, spec, cfg, circuits_all)
    print(f"current model done ({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time()
    new = run_model(NEW, spec, cfg, circuits_all)
    print(f"new model done ({time.time() - t0:.0f}s)", flush=True)

    wins = losses = ties = 0
    d_cur, d_new = [], []
    for name in cur:
        if "fidelity" not in cur[name] or "fidelity" not in new[name]:
            continue
        fc, fn = cur[name]["fidelity"], new[name]["fidelity"]
        d_cur.append(fc)
        d_new.append(fn)
        if fn > fc * (1 + 1e-9):
            wins += 1
        elif fc > fn * (1 + 1e-9):
            losses += 1
        else:
            ties += 1
    m_cur = sum(d_cur) / len(d_cur)
    m_new = sum(d_new) / len(d_new)
    verdict = (m_new > m_cur and wins > losses)
    with open("tables/ab_postfid.json", "w", encoding="utf-8") as f:
        json.dump({"mean_current": m_cur, "mean_new": m_new,
                   "wins": wins, "losses": losses, "ties": ties,
                   "verdict": "adopt" if verdict else "reject"},
                  f, ensure_ascii=False, indent=2)
    print(f"\n== A/B verdict ==")
    print(f"mean fidelity: current {m_cur:.4f} vs new {m_new:.4f} "
          f"({(m_new / m_cur - 1) * 100:+.2f}%)")
    print(f"per-circuit: new wins {wins} / losses {losses} / ties {ties}")
    print(f"verdict: {'ADOPT (run full stratified)' if verdict else 'REJECT (keep current)'}")


if __name__ == "__main__":
    main()
