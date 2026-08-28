"""qiskit-free 映射 CLI：python -m qtrail.pure_cli input.qasm [options]

零 qiskit 依赖的竞赛交付入口：自研解析 → RL 布局 → 自研 SABRE 路由 →
自研后处理 → CZ 基输出（QASM/QCIS/JSON）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser(description="QTrial 纯自研映射管线（零 qiskit）")
    ap.add_argument("input", help="OpenQASM 2.0 线路文件")
    ap.add_argument("--device", default="tianyan-287")
    ap.add_argument("--output", "-o", default="out")
    ap.add_argument("--rule", default="fidelity",
                    choices=["swap", "fidelity", "depth"],
                    help="候选决胜规则（默认 fidelity，与最终评测同口径）")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--fidelity-checkpoint", default=None,
                    help="可选：QuEst 图 Transformer 保真度预测器权重（无则用乘积模型）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--no-post", action="store_true", help="禁用后处理栈")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    assert "qiskit" not in sys.modules, "qiskit leaked into pure CLI"

    from qtrail.config import Config, load_device_config
    from qtrail.pure.qasm import load_qasm_file
    from qtrail.pure.mapper import PureMapper
    from qtrail.pure.export import to_qasm, to_qcis

    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    if args.device != "tianyan-287":
        raise ValueError(f"unknown device: {args.device} (pure 路径支持 tianyan-287)")
    from qtrail.devices import build_tianyan287_spec
    spec = build_tianyan287_spec(dev_cfg)

    import torch
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = None
    ckpt_path = args.checkpoint or (
        "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt")
    if Path(ckpt_path).exists():
        from qtrail.models import QAPolicy
        policy, ck = QAPolicy.load_checkpoint(ckpt_path, device_n=spec.n,
                                              map_location=dev)
        policy.eval()
        cfg.model = ck.get("model_cfg", cfg.model)
        cfg.graph = policy.graph_cfg

    fidelity_predictor = None
    if args.fidelity_checkpoint and Path(args.fidelity_checkpoint).exists():
        from qtrail.models import FidelityPredictor
        fidelity_predictor, _ = FidelityPredictor.load_checkpoint(
            args.fidelity_checkpoint, map_location=dev)
        fidelity_predictor.eval()

    circ = load_qasm_file(args.input)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=args.seed,
                        selection_rule=args.rule, use_post=not args.no_post,
                        fidelity_predictor=fidelity_predictor)
    res = mapper.map_circuit(circ, circuit_id=Path(args.input).stem)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    qasm_path = out_dir / f"{stem}_mapped.qasm"
    qasm_path.write_text(to_qasm(res["routed_circuit"]), encoding="utf-8")
    qcis_path = out_dir / f"{stem}.qcis"
    qcis_path.write_text(to_qcis(res["routed_circuit"]), encoding="utf-8")

    metrics = dict(res["metrics"])
    metrics["method"] = res["method"]
    metrics["wall_s"] = round(time.time() - t0, 3)
    metrics["layout"] = {str(k): v for k, v in res["layout"].items()}
    metrics["final_layout"] = {str(k): v
                               for k, v in res["final_layout"].items()}
    metrics["checkpoint"] = ckpt_path
    metrics["qasm_path"] = str(qasm_path)
    metrics["qcis_path"] = str(qcis_path)
    json_path = out_dir / f"{stem}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=float)

    assert "qiskit" not in sys.modules, "qiskit leaked during mapping"
    print(f"== QTrial pure mapping: {args.input} ({spec.name}) ==")
    print(f"  method      : {res['method']}")
    print(f"  swap_count  : {res['swap_count']}")
    print(f"  depth       : {res['metrics']['depth']} (2Q: {res['metrics']['twoq_depth']})")
    print(f"  est_fidelity: {res['metrics']['est_fidelity']:.6f}")
    print(f"  wall        : {metrics['wall_s']:.2f}s")
    print(f"  outputs     : {qasm_path}, {qcis_path}, {json_path}")
    print(f"  qiskit-free : confirmed (zero qiskit imports)")


if __name__ == "__main__":
    main()
