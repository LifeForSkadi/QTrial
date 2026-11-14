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
    ap.add_argument("--device", default="tianyan-287",
                    choices=["tianyan-287", "tianyan-287-real"],
                    help="tianyan-287 = 评测口径（188 边矩形网格+合成校准）；"
                         "tianyan-287-real = 真机拓扑（168 可用边+4 禁用比特+"
                         "平台校准快照）")
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
    ap.add_argument("--pipeline", default="v4", choices=["v4", "v5"],
                    help="管线版本：v4 = 原管线（默认，行为不变）；"
                         "v5 = 新管线（1Q 游程优化 + 2Q 块重合成，测试中）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    assert "qiskit" not in sys.modules, "qiskit leaked into pure CLI"

    from qtrail.config import Config, load_device_config
    from qtrail.pure.qasm import load_qasm_file
    from qtrail.pure.mapper import PureMapper
    from qtrail.pure.export import to_qasm, to_qcis

    if args.device == "tianyan-287":
        dev_cfg = load_device_config()
    elif args.device == "tianyan-287-real":
        # 真机拓扑（2026-11-09 API 核验固化）：168 可用边 + 4 禁用比特
        # + 平台真实校准快照（configs/device_tianyan287_real*.{yaml,json}）
        dev_cfg = load_device_config("configs/device_tianyan287_real.yaml")
    else:
        raise ValueError(f"unknown device: {args.device} "
                         "(pure 路径支持 tianyan-287 / tianyan-287-real)")
    cfg = Config()
    cfg.device = dev_cfg
    from qtrail.devices import build_tianyan287_spec
    if args.device == "tianyan-287":
        spec = build_tianyan287_spec(dev_cfg)
    else:
        from qtrail.devices import build_tianyan287_real_spec
        spec = build_tianyan287_real_spec(dev_cfg)

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
                        fidelity_predictor=fidelity_predictor,
                        pipeline=args.pipeline)
    res = mapper.map_circuit(circ, circuit_id=Path(args.input).stem)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    qasm_path = out_dir / f"{stem}_mapped.qasm"
    qasm_path.write_text(to_qasm(res["routed_circuit"]), encoding="utf-8")
    qcis_path = out_dir / f"{stem}.qcis"
    qcis_path.write_text(to_qcis(res["routed_circuit"],
                                 labels=spec.qubit_labels),
                         encoding="utf-8")

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

    # 后端披露（2026-11-12 审查 P1-2；2026-11-13 第 3 项后默认全自研）：
    # A2 2Q 块合成默认走自研 Weyl/Vatan-Williams 闭式实现（weyl.py）；
    # qiskit 仅在显式 QTRAIL_A2_BACKEND=qiskit 时作为对照后端参与。
    from qtrail.pure import synthesis as _syn
    a2_backend = _syn.A2_BACKEND
    leaked = "qiskit" in sys.modules
    assert not leaked or a2_backend == "qiskit", \
        "qiskit leaked into pure CLI outside the A2 synthesizer backend"
    print(f"== QTrial pure mapping: {args.input} ({spec.name}) ==")
    print(f"  method      : {res['method']}")
    print(f"  swap_count  : {res['swap_count']}")
    print(f"  depth       : {res['metrics']['depth']} (2Q: {res['metrics']['twoq_depth']})")
    print(f"  est_fidelity: {res['metrics']['est_fidelity']:.6f}")
    print(f"  wall        : {metrics['wall_s']:.2f}s")
    print(f"  outputs     : {qasm_path}, {qcis_path}, {json_path}")
    if a2_backend == "qiskit":
        print("  A2 backend  : qiskit Weyl/KAK 闭式分解器（显式启用的对照后端）")
    elif a2_backend == "torch":
        print("  A2 backend  : 自研 torch 合成回退（零 qiskit 依赖）")
    elif a2_backend == "vw":
        print("  A2 backend  : 自研 Weyl/Vatan-Williams 闭式合成"
              "（零 qiskit 依赖）")
    else:
        print("  A2 backend  : 未使用（v4 管线或线路无 2Q 块）")


if __name__ == "__main__":
    main()
