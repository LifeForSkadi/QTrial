"""布局贡献对比：本方案布局 + qiskit SABRE 路由 vs pytket / 盲目与噪声感知 O1/O3。

对指定测试集逐线路在**同一进程**内跑六方法（MQTBench 随机参数线路
每次加载不同，必须单进程对照）：
  1. 本方案 = RL 布局（指定权重）+ qiskit SABRE 路由（支撑版 Mapper）
  2. 盲目 O1 / O3 = SABRE 编译管线（同指标口径）
  3. 噪声感知 O1 / O3 = 校准 Target 注入噪声模型的 qiskit 编译管线
  4. pytket = DefaultMappingPass 默认映射管线（隐式置换物化 + 平台基
     分解，与 final_report_bench 同口径）

逐条落盘 + 断点续跑：每条线路六方法全部完成后立即写盘；重跑时自动
跳过已含 aw-o3 字段的线路。

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
from qtrail.pipeline.baselines import aware_transpile, build_target
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
    print(f"bench={args.bench} circuits={len(circuits)}", flush=True)

    # 断点续跑：已含 aw-o3 字段的线路直接复用
    out = (Path("tables/compare_models")
           / f"layout_contribution_{args.bench}_rows.jsonl")
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["circuit"]] = r

    def aw_done(r):
        return "aw-o3_fidelity" in r or "aw-o3_error" in r

    def write_rows(path, rs):
        with open(path, "w", encoding="utf-8") as f:
            for row in rs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
    target = build_target(spec)

    rows = []
    for name, qc in circuits:
        if name in done and aw_done(done[name]):
            rows.append(done[name])
            print(f"  [skip] {name}", flush=True)
            write_rows(out, rows)
            continue
        r = evaluate_benchmark([(name, qc)], mapper,
                               baselines=("qiskit-o1", "qiskit-o3"))[0]

        qc_clean, _ = strip_measurements(qc)

        # tket 基线（同进程）
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

        # 噪声感知 O1/O3 基线（同进程，校准 Target 注入噪声模型，
        # 口径同 final_report_bench 的 aw_o1/aw_o3）
        for opt, pref in ((1, "aw-o1"), (3, "aw-o3")):
            t0 = time.time()
            try:
                routed = aware_transpile(qc_clean, target,
                                         optimization_level=opt, seed=42)
                sc = routed.count_ops().get("swap", 0)
                m = compute_metrics(routed, sc, calib)
                r[f"{pref}_swaps"] = sc
                r[f"{pref}_twoq"] = m["twoq_count"]
                r[f"{pref}_depth"] = m["depth"]
                r[f"{pref}_fidelity"] = m["est_fidelity"]
                r[f"{pref}_mean_2q_err"] = m.get("mean_2q_err")
                r[f"{pref}_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                r[f"{pref}_error"] = f"{type(e).__name__}: {str(e)[:60]}"

        rows.append(r)
        write_rows(out, rows)
        fid = r.get("fidelity")
        print(f"  {name[:52]:52s} "
              f"rl_fid={fid:.4f} aw_o1={r.get('aw-o1_fidelity')} "
              f"aw_o3={r.get('aw-o3_fidelity')}" if isinstance(fid, float)
              else f"  {name[:52]:52s} rl_error={r.get('error')}",
              flush=True)

    print(f"saved -> {out} ({len(rows)} 行)")


if __name__ == "__main__":
    main()
