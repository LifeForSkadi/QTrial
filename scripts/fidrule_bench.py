"""QTrial 保真度决胜规则对比 + mean2q 机制证据补跑（无 MCTS，快速）。

对每条分层线路：
  - QTrial+SABRE fidelity 规则（保真度优先竞技）
  - QTrial+SABRE swap 规则（SWAP 优先 + 保真度决胜）
  - 盲目 O1 / pytket 的 mean_2q_err（热区规避的机制证据列）
输出: tables/noise_aware/{bucket}_fidrule.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2, transpile
import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.baselines import sabre_transpile
from qtrail.pipeline.external import tket_compile
from qtrail.pipeline.mapper import Mapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pipeline.routing import decompose_to_platform
from qtrail.utils.qasm_io import sanitize_qasm, strip_measurements

CACHE = Path("data/mqtbench/stratified")
OUT = Path("tables/noise_aware")
CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"


def main():
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    calib = spec.calib
    policy, ckpt = QAPolicy.load_checkpoint(
        CKPT, device_n=spec.n,
        map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    cm = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
          if spec.adj[i, j]]

    for p in sorted(CACHE.glob("*.qasm")):
        name = p.stem
        size = int(name.rsplit("_", 1)[1])
        if size > 50:
            continue
        bucket = "small" if size <= 10 else "medium"
        qc = qasm2.loads(sanitize_qasm(p.read_text(encoding="utf-8")))
        qc, _ = strip_measurements(qc)
        row = {"circuit": name, "n": size}

        # QTrial fidelity 规则 + swap 规则
        for rule in ("fidelity", "swap"):
            t0 = time.time()
            try:
                mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda",
                                seed=42, selection_rule=rule,
                                routing_method="sabre")
                res = mapper.map_circuit(qc, circuit_id=name)
                row[f"os_{rule}_swaps"] = res.swap_count
                row[f"os_{rule}_depth"] = res.metrics["depth"]
                row[f"os_{rule}_twoq_depth"] = res.metrics["twoq_depth"]
                row[f"os_{rule}_fidelity"] = res.metrics["est_fidelity"]
                row[f"os_{rule}_mean2q"] = res.metrics.get("mean_2q_err")
                row[f"os_{rule}_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row[f"os_{rule}_error"] = f"{type(e).__name__}: {str(e)[:60]}"

        # 盲目 O1 的 mean2q（机制证据）
        try:
            blind = transpile(qc, coupling_map=cm,
                              basis_gates=["rz", "sx", "x", "cz"],
                              optimization_level=1, seed_transpiler=42)
            m = compute_metrics(blind, blind.count_ops().get("swap", 0), calib)
            row["blind_o1_mean2q"] = m.get("mean_2q_err")
        except Exception:
            pass

        # pytket 的 mean2q
        try:
            expanded, sc = tket_compile(qc, cm, spec.n)
            expanded = decompose_to_platform(expanded,
                                             Mapper(spec).cm,
                                             optimization_level=1, seed=42)
            m = compute_metrics(expanded, sc, calib)
            row["tket_mean2q"] = m.get("mean_2q_err")
        except Exception:
            pass

        with open(OUT / f"{bucket}_fidrule.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            f.flush()
        print(f"  {name:30s} fid-rule {row.get('os_fidelity_swaps', '-')}/"
              f"{row.get('os_fidelity_fidelity', '-')} | "
              f"swap-rule {row.get('os_swap_swaps', '-')}/"
              f"{row.get('os_swap_fidelity', '-')}", flush=True)
    print("[fidrule] done")


if __name__ == "__main__":
    main()
