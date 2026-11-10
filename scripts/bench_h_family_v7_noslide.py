# -*- coding: utf-8 -*-
"""变体：禁用 _preslide_1q 的达标三线路（检验 preslide 是否过度聚合）。
与 bench_h_family_v7.py 同口径，仅 monkeypatch 解碎片为 no-op。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import qtrail.pure.post_v2 as pv2
pv2._preslide_1q = lambda ops, max_rounds=8: 0  # 禁用解碎片

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
RL_O3 = Path("tables/compare_models/rl_o3_587_rows.jsonl")
CASES = [("bp_square-heisenberg_square_heisenberg_N16",
          "data/benchpress/qasm/square-heisenberg/square_heisenberg_N16.qasm"),
         ("bp_square-heisenberg_square_heisenberg_N25",
          "data/benchpress/qasm/square-heisenberg/square_heisenberg_N25.qasm"),
         ("bp_square-heisenberg_square_heisenberg_N36",
          "data/benchpress/qasm/square-heisenberg/square_heisenberg_N36.qasm")]


def main():
    rl_o3 = {}
    if RL_O3.exists():
        for line in RL_O3.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rl_o3[r["circuit"]] = r.get("rl_o3_fidelity")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    for name, path in CASES:
        circ = parse_qasm(Path(path).read_text(encoding="utf-8"))
        torch.manual_seed(42)
        mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                            selection_rule="fidelity", use_post=True,
                            pipeline="v5", routing_seeds=16)
        t0 = time.time()
        res = mapper.map_circuit(circ, circuit_id=name)
        m = res["metrics"]
        f = m["est_fidelity"]
        base = rl_o3.get(name)
        ratio = (f / base) if base else float("nan")
        print(f'{name}: v7={f:.4e} RL+O3={base} 比值={ratio:.3f}x '
              f'swaps={res["swap_count"]} twoq={m["twoq_count"]} '
              f'depth={m["depth"]} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == "__main__":
    main()
