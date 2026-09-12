"""布局贡献对比：本方案布局 + qiskit SABRE 路由 vs pytket / Qiskit O1 / O3。

对指定测试集逐线路在**同一进程**内跑四方法（MQTBench 随机参数线路
每次加载不同，必须单进程对照）：
  1. 本方案 = RL 布局（指定权重）+ qiskit SABRE 路由（支撑版 Mapper）
  2. Qiskit O1 / O3 = SABRE 编译管线（同指标口径）
  3. pytket = DefaultMappingPass 默认映射管线（隐式置换物化 + 平台基
     分解，与 final_report_bench 同口径）

用法：
  python scripts/bench_layout_contribution.py --bench benchpress --limit 30
  python scripts/bench_layout_contribution.py --bench mqtbench_n15
  python scripts/bench_layout_contribution.py --bench queko_BIGD --limit 30
输出：tables/compare_models/layout_contribution_<bench>_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.external import tket_compile
from qtrail.pipeline.mapper import Mapper as QiskitMapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pipeline.routing import decompose_to_platform
from qtrail.cli.eval_cli import evaluate_benchmark, load_benchmark_circuits
from qtrail.utils.qasm_io import strip_measurements

DEFAULT_CKPT = ("checkpoints/demo_laurel/"
                "tianyan-287_gat_combined_calib_dep0.1_t0.5_best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="benchpress")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spec = build_tianyan287_spec()
    cm_qiskit = QiskitMapper(spec).cm
    cm_edges = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
                if spec.adj[i, j]]
    calib = spec.calib

    circuits = load_benchmark_circuits(
        args.bench, args.limit if args.limit else None)
    print(f"bench={args.bench} circuits={len(circuits)}")

    # 本方案（RL 布局 + qiskit SABRE 路由）+ qiskit 基线
    policy, ckpt = QAPolicy.load_checkpoint(args.checkpoint, device_n=spec.n,
                                            map_location=args.dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    mapper = QiskitMapper(spec, policy=policy, cfg=cfg, dev=args.dev,
                          seed=args.seed)
    rows = evaluate_benchmark(circuits, mapper,
                              baselines=("qiskit-o1", "qiskit-o3"))

    # tket 基线（同进程）
    for r, (name, qc) in zip(rows, circuits):
        qc_clean, _ = strip_measurements(qc)
        t0 = time.time()
        try:
            expanded, sc = tket_compile(qc_clean, cm_edges, spec.n)
            expanded = decompose_to_platform(expanded, cm_qiskit,
                                             optimization_level=1, seed=42)
            m = compute_metrics(expanded, sc, calib)
            r["tket_swaps"] = sc
            r["tket_twoq"] = m["twoq_count"]
            r["tket_depth"] = m["depth"]
            r["tket_fidelity"] = m["est_fidelity"]
            r["tket_mean_2q_err"] = m.get("mean_2q_err")
            r["tket_wall"] = round(time.time() - t0, 2)
        except Exception as e:
            r["tket_error"] = f"{type(e).__name__}: {str(e)[:60]}"
        fid = r.get("fidelity")
        print(f"  {name[:52]:52s} "
              f"rl_fid={fid:.4f}" if isinstance(fid, float) else
              f"  {name[:52]:52s} rl_error={r.get('error')}",
              flush=True)

    out = (Path("tables/compare_models")
           / f"layout_contribution_{args.bench}_rows.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
