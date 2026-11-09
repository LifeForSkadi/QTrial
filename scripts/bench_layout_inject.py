"""布局注入对照：本方案 RL 布局注入 qiskit 噪声感知 O3 管线 vs 原生噪声感知 O3。

对指定测试集逐线路在**同一进程**内对照（MQTBench 随机参数线路每次加载
不同，注入组与原生组必须共享同一电路实例）：
  1. 注入组 = RL 布局（指定权重，多起点 + 局部搜索）作为 initial_layout
     注入 qiskit 噪声感知 O3 编译管线（校准 Target）
  2. 原生组 = qiskit 噪声感知 O3（同一 Target，自选布局）
两组指标均以 compute_metrics 同口径计算（校准数据一致、种子一致）。

--screen：注入组改用 O3 路由试探筛选模式（候选池 + SABRE 预筛排序 +
O3 试探 + 时间红线兜底，budget/per-candidate 可调），原生组不变。

逐条落盘 + 断点续跑：每条线路两组完成后立即写盘；重跑时自动跳过已含
inj_fidelity 的线路。

用法：
  python scripts/bench_layout_inject.py --bench benchpress --limit 30
  python scripts/bench_layout_inject.py --bench mqtbench_n15
  python scripts/bench_layout_inject.py --bench queko_BIGD --limit 30
  python scripts/bench_layout_inject.py --bench benchpress --limit 30 --screen
输出：tables/compare_models/layout_inject_<bench>_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import transpile

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.baselines import build_target
from qtrail.pipeline.mapper import Mapper as QiskitMapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.cli.eval_cli import load_benchmark_circuits
from qtrail.utils.qasm_io import PLATFORM_BASIS, strip_measurements

DEFAULT_CKPT = ("checkpoints/demo_laurel/"
                "tianyan-287_gat_combined_calib_dep0.1_t0.5_best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="benchpress")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--screen", action="store_true",
                    help="注入组改用 O3 路由试探筛选模式（候选池+SABRE 预筛"
                         "+O3 试探+时间红线兜底）")
    ap.add_argument("--budget", type=float, default=600.0,
                    help="O3 筛选总时间预算（秒，默认 600）")
    ap.add_argument("--per-candidate", type=float, default=60.0,
                    help="O3 筛选单候选上限（秒，默认 60）")
    ap.add_argument("--no-native", action="store_true",
                    help="跳过原生感知 O3 组（固定文件测试集已由 "
                         "layout_contribution 的 aw-o3 覆盖，避免重复计算）")
    args = ap.parse_args()

    spec = build_tianyan287_spec()
    calib = spec.calib
    target = build_target(spec)

    circuits = load_benchmark_circuits(
        args.bench, args.limit if args.limit else None)
    print(f"bench={args.bench} circuits={len(circuits)}", flush=True)

    out = (Path("tables/compare_models")
           / f"layout_inject_{args.bench}_rows.jsonl")
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["circuit"]] = r

    policy, ckpt = QAPolicy.load_checkpoint(args.checkpoint, device_n=spec.n,
                                            map_location=args.dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    mapper = QiskitMapper(spec, policy=policy, cfg=cfg, dev=args.dev,
                          seed=args.seed)
    screen_mapper = None
    if args.screen:
        screen_mapper = QiskitMapper(
            spec, policy=policy, cfg=cfg, dev=args.dev, seed=args.seed,
            target_post=True, target_post_opt=3, target_post_score_opt=3,
            selection_rule="fidelity", o3_screening=True,
            o3_budget=args.budget, o3_per_candidate=args.per_candidate)

    def write_rows(path, rs):
        with open(path, "w", encoding="utf-8") as f:
            for row in rs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = []
    for name, qc in circuits:
        if name in done and ("inj_fidelity" in done[name]
                             or "inj_error" in done[name]):
            rows.append(done[name])
            print(f"  [skip] {name}", flush=True)
            write_rows(out, rows)
            continue
        qc_clean, _ = strip_measurements(qc)
        row = {"circuit": name, "n_qubits": qc_clean.num_qubits}

        if args.screen:
            # 注入组 = O3 路由试探筛选模式（mapper 全管线：候选池 +
            # SABRE 预筛排序 + O3 试探 + 时间红线兜底），原生组同前
            t0 = time.time()
            try:
                res = screen_mapper.map_circuit(qc_clean, circuit_id=name,
                                                optimization_level=1)
                row["inj_swaps"] = res.swap_count
                row["inj_twoq"] = res.metrics.get("twoq_count")
                row["inj_depth"] = res.metrics.get("depth")
                row["inj_fidelity"] = res.metrics.get("est_fidelity")
                row["inj_mean_2q_err"] = res.metrics.get("mean_2q_err")
                row["inj_wall"] = round(time.time() - t0, 2)
                row["inj_method"] = res.method
                row["inj_screen"] = res.screening
            except Exception as e:
                row["inj_error"] = f"{type(e).__name__}: {str(e)[:60]}"
        else:
            # RL 布局（多起点 + 局部搜索）
            init = None
            try:
                res = mapper.map_circuit(qc_clean, circuit_id=name,
                                         optimization_level=1)
                init = {qc_clean.qubits[i]: int(res.layout[i])
                        for i in range(qc_clean.num_qubits)}
            except Exception as e:
                row["rl_error"] = f"{type(e).__name__}: {str(e)[:60]}"

            # 注入组：RL 布局 → 噪声感知 O3 管线
            t0 = time.time()
            if init is None:
                row["inj_error"] = row.get("rl_error", "rl layout failed")
            else:
                try:
                    routed = transpile(qc_clean, target=target,
                                       basis_gates=PLATFORM_BASIS,
                                       optimization_level=3,
                                       initial_layout=init,
                                       seed_transpiler=42)
                    m = compute_metrics(routed,
                                        routed.count_ops().get("swap", 0),
                                        calib)
                    row["inj_swaps"] = m["swap_count"]
                    row["inj_twoq"] = m["twoq_count"]
                    row["inj_depth"] = m["depth"]
                    row["inj_fidelity"] = m["est_fidelity"]
                    row["inj_mean_2q_err"] = m.get("mean_2q_err")
                    row["inj_wall"] = round(time.time() - t0, 2)
                except Exception as e:
                    row["inj_error"] = f"{type(e).__name__}: {str(e)[:60]}"

        # 原生组：噪声感知 O3（自选布局，同种子）；--no-native 时跳过
        # （固定文件测试集上等价于 layout_contribution 的 aw-o3）
        if not args.no_native:
            t0 = time.time()
            try:
                routed = transpile(qc_clean, target=target,
                                   basis_gates=PLATFORM_BASIS,
                                   optimization_level=3, seed_transpiler=42)
                m = compute_metrics(routed, routed.count_ops().get("swap", 0),
                                    calib)
                row["native_swaps"] = m["swap_count"]
                row["native_twoq"] = m["twoq_count"]
                row["native_depth"] = m["depth"]
                row["native_fidelity"] = m["est_fidelity"]
                row["native_mean_2q_err"] = m.get("mean_2q_err")
                row["native_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["native_error"] = f"{type(e).__name__}: {str(e)[:60]}"

        rows.append(row)
        write_rows(out, rows)
        print(f"  {name[:52]:52s} inj={row.get('inj_fidelity')} "
              f"native={row.get('native_fidelity')}", flush=True)

    print(f"saved -> {out} ({len(rows)} 行)")


if __name__ == "__main__":
    main()
