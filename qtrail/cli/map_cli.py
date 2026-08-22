"""Judge-facing mapping CLI: python -m qtrail.map input.qasm [options]"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from qtrail.config import CHECKPOINTS_DIR, Config, load_device_config
from qtrail.pipeline.mapper import Mapper

log = logging.getLogger("qtrail")


# 推荐权重优先级：v2（深度感知+时序图，最终模型）> v1 噪声感知 > 其余
_PREFERRED = [
    "tianyan-287_gat_combined_calib_dep0.1_t0.5_best.pt",
    "tianyan-287_gat_combined_calib_best.pt",
    "tianyan-287_gat_topology_onehot_best.pt",
    "tianyan-287_gt_combined_calib_best.pt",
]


def find_checkpoint(name: str, explicit: str | None = None) -> str | None:
    if explicit:
        p = Path(explicit)
        return str(p) if p.exists() else None
    for cand in (Path(name), CHECKPOINTS_DIR / name):
        if cand.exists():
            return str(cand)
    for pref in _PREFERRED:
        p = CHECKPOINTS_DIR / pref
        if p.exists():
            return str(p)
    # scan checkpoints dir for a device-matching best checkpoint
    if CHECKPOINTS_DIR.exists():
        matches = sorted(CHECKPOINTS_DIR.glob(f"{name.split('_')[0]}_*_best.pt"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return str(matches[0])
    return None


def load_policy(checkpoint: str | None, spec, dev: str):
    """Load a policy from checkpoint (returns None if unavailable)."""
    if not checkpoint:
        return None, None
    from qtrail.models import QAPolicy
    policy, ckpt = QAPolicy.load_checkpoint(checkpoint, device_n=spec.n,
                                            map_location=dev)
    policy.eval()
    return policy, ckpt


def run(args) -> dict:
    t0 = time.time()
    cfg = Config()
    dev_cfg = load_device_config()
    cfg.device = dev_cfg

    # ---- device spec (live platform config when requested)
    spec = None
    if args.calibration == "live":
        from qtrail.devices.adapter import download_tianyan287_spec
        spec = download_tianyan287_spec(machine=args.device)
    if spec is None:
        if args.device == "tianyan-287":
            from qtrail.devices import build_tianyan287_spec
            spec = build_tianyan287_spec(dev_cfg)
        elif args.device == "grid-8x8":
            from qtrail.devices import build_grid8x8_spec
            spec = build_grid8x8_spec()
        elif args.device == "grid-3x3":
            from qtrail.devices import build_grid3x3_spec
            spec = build_grid3x3_spec()
        else:
            raise ValueError(f"unknown device: {args.device}")
        if args.calibration == "live":
            log.warning("live calibration unavailable; using synthetic spec")

    # ---- policy
    dev = args.dev or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    ckpt_path = find_checkpoint(_PREFERRED[0], args.checkpoint)
    policy, ckpt = load_policy(ckpt_path, spec, dev)
    if policy is not None:
        cfg.model = ckpt.get("model_cfg", cfg.model)
        cfg.graph = policy.graph_cfg  # 评测图表示与训练一致
        log.info("policy loaded: %s (epoch %s)", ckpt_path, ckpt.get("epoch"))
    else:
        log.warning("no checkpoint found; heuristic ladder will be used")

    # ---- circuit
    from qiskit import QuantumCircuit
    from qtrail.utils.qasm_io import (append_measurements, load_qasm2,
                                      strip_measurements)
    try:
        qc = load_qasm2(args.input)
    except Exception as e:
        print(f"错误：无法解析 OpenQASM 2.0 文件 {args.input}\n  {type(e).__name__}: {e}")
        raise SystemExit(1)
    qc_clean, removed = strip_measurements(qc)
    if qc_clean.num_qubits == 0:
        raise ValueError("circuit has no qubits")

    # ---- map
    # 研究管线（默认）：RL 多起点布局 + 路由感知择优 + SabreSwap 仅作路由器
    # --ensemble：混合竞技工程增强（纳入 SABRE 布局/O3/pytket 候选）
    cfg.decode.multistart_k = args.starts
    cfg.postprocess.enabled = args.postprocess
    mapper = Mapper(spec, policy=policy, cfg=cfg, dev=dev, seed=args.seed,
                    use_tket=args.ensemble, use_o3=args.ensemble,
                    include_sabre=args.ensemble,
                    selection_rule=args.rule,
                    routing_method=args.routing,
                    cqlib_objective=args.cqlib_objective,
                    cqlib_timeout=args.cqlib_timeout,
                    target_post=args.target_post,
                    target_post_opt=args.target_post_opt,
                    target_post_seeds=args.target_post_seeds,
                    target_post_top_per_seed=(args.target_post_top or None))
    try:
        result = mapper.map_circuit(qc_clean, circuit_id=Path(args.input).stem,
                                    optimization_level=args.opt,
                                    has_measurements=bool(removed))
    except ValueError as e:
        print(f"错误：{e}")
        raise SystemExit(1)

    # ---- outputs
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    final_qc = append_measurements(result.routed_qc, removed, result.final_layout,
                                   original=qc)

    metrics = dict(result.metrics)
    metrics["method"] = result.method
    metrics["wall_s"] = round(time.time() - t0, 3)
    metrics["warnings"] = result.warnings
    metrics["layout"] = {str(k): v for k, v in result.layout.items()}
    metrics["final_layout"] = ({str(k): v for k, v in result.final_layout.items()}
                               if result.final_layout is not None else None)
    metrics["checkpoint"] = ckpt_path

    if "qasm" in args.format:
        from qtrail.utils.qasm_io import write_qasm2
        qasm_path = out_dir / f"{stem}_mapped.qasm"
        write_qasm2(final_qc, qasm_path)
        metrics["qasm_path"] = str(qasm_path)
    if "qcis" in args.format:
        from qtrail.utils.qcis import write_qcis
        qcis_path = out_dir / f"{stem}.qcis"
        write_qcis(final_qc, qcis_path)
        metrics["qcis_path"] = str(qcis_path)

    # ---- baselines (comparative)
    if args.baseline:
        from qtrail.pipeline.baselines import (sabre_swap_count,
                                               sabre_transpile)
        from qtrail.pipeline.metrics import compute_metrics
        for b in args.baseline.split(","):
            b = b.strip()
            if b.startswith("qiskit-o"):
                o = int(b.split("-o")[-1])
                try:
                    sc, routed = sabre_swap_count(qc_clean, mapper.cm,
                                                  optimization_level=o,
                                                  seed=args.seed)
                    final_b = sabre_transpile(qc_clean, mapper.cm,
                                              optimization_level=o,
                                              seed=args.seed)
                    m = compute_metrics(final_b, sc, spec.calib)
                    metrics[f"baseline_{b}"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                for k, v in m.items()}
                except Exception as e:
                    metrics[f"baseline_{b}"] = {"error": str(e)}

    json_path = out_dir / f"{stem}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=float)
    metrics["metrics_path"] = str(json_path)

    # ---- console summary
    print(f"\n== QTrial mapping result: {args.input} ({spec.name}) ==")
    print(f"  method          : {result.method}")
    print(f"  SWAP count      : {result.swap_count}")
    print(f"  2Q gate count   : {result.metrics.get('twoq_count')}")
    print(f"  depth / 2Q depth: {result.metrics.get('depth')} / {result.metrics.get('twoq_depth')}")
    print(f"  est. fidelity   : {result.metrics.get('est_fidelity', 0):.4f}")
    for k, v in metrics.items():
        if k.startswith("baseline_"):
            if "error" in v:
                print(f"  {k}: ERROR {v['error']}")
            else:
                print(f"  {k}: swaps={v.get('swap_count')} 2q={v.get('twoq_count')} "
                      f"depth={v.get('depth')} fid={v.get('est_fidelity', 0):.4f}")
    print(f"  wall time       : {metrics['wall_s']}s")
    for w in result.warnings:
        print(f"  warning         : {w}")
    print(f"  outputs         : {out_dir}/")

    if args.submit:
        from qtrail.submit import submit_qcis
        qcis_path = out_dir / f"{stem}.qcis"
        submit_qcis(qcis_path, machine=args.device)

    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description="QTrial quantum circuit mapper")
    ap.add_argument("input", help="OpenQASM 2.0 circuit file")
    ap.add_argument("--device", default="tianyan-287",
                    choices=["tianyan-287", "grid-8x8", "grid-3x3"])
    ap.add_argument("--output", "-o", default="out")
    ap.add_argument("--format", default="qasm,qcis,json")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--calibration", default="synthetic",
                    choices=["synthetic", "live", "none"])
    ap.add_argument("--decode", default="multistart",
                    choices=["multistart", "greedy", "sample"])
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--postprocess", type=lambda s: s.lower() in ("on", "true", "1"),
                    default=True)
    ap.add_argument("--opt", type=int, default=1, help="final decomposition level")
    ap.add_argument("--baseline", default="qiskit-o1,qiskit-o3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--fast", action="store_true",
                    help="跳过路由感知择优的候选路由评估（最快模式，仅 RL 贪心+LS）")
    ap.add_argument("--ensemble", action="store_true",
                    help="混合竞技工程增强：纳入 SABRE 布局/O3/pytket 候选（研究管线默认关闭）")
    ap.add_argument("--routing", default="sabre",
                    choices=["sabre", "lexi", "cqlib"],
                    help="路由器：sabre（默认）| lexi（自研深度感知）| "
                         "cqlib（平台原生 MCTS，注入 RL 布局）")
    ap.add_argument("--cqlib-objective", default="depth",
                    choices=["size", "depth"],
                    help="cqlib 路由目标（仅 --routing cqlib 生效）")
    ap.add_argument("--cqlib-timeout", type=float, default=300.0,
                    help="cqlib 路由超时保护（秒）")
    ap.add_argument("--target-post", action="store_true",
                    help="Target 后处理管线：RL 布局 + qiskit O1 预设"
                         "（噪声感知 Target 驱动，路由后重标记/酉综合）")
    ap.add_argument("--target-post-opt", type=int, default=1,
                    choices=[1, 2, 3],
                    help="target-post 优化级别（3 = 更激进的酉综合，更强但更慢）")
    ap.add_argument("--target-post-seeds", type=int, default=8,
                    help="target-post 多种子候选数（多试验取最优）")
    ap.add_argument("--target-post-top", type=int, default=3,
                    help="每种子静态代价 top-K 入池（0=全量）")
    ap.add_argument("--rule", default="swap", choices=["swap", "fidelity", "depth"],
                    help="混合竞技决胜规则（默认 swap 优先）")
    ap.add_argument("--submit", action="store_true",
                    help="submit QCIS to the Tianyan platform (needs login key)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    run(args)
