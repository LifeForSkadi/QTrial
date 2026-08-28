"""对比多个模型（checkpoint）在测试集上的映射性能 + 生产基线。

一次性跑多个权重，输出并排对比表（SWAP / 深度 / 2Q / 保真度 / 静态代价
/ 相对 qiskit 基线），数据逐线路落盘 jsonl。

用法：
  python scripts/compare_models.py \
      --bench queko_BIGD --limit 30 \
      --model "LAUREL(89ep)=checkpoints/demo_laurel/tianyan-287_gat_combined_calib_dep0.1_t0.5_best.pt" \
      --model "源项目(400ep)=checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt" \
      --baselines qiskit-o1,qiskit-o3 --dev cpu

每个 --model 的格式为「标签=权重路径」。baselines 默认为 qiskit-o1,qiskit-o3。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Windows 控制台默认 GBK，强制 UTF-8 避免中文乱码
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import Config, load_device_config
from qtrail.pipeline.mapper import Mapper
from qtrail.cli.eval_cli import (evaluate_benchmark, load_benchmark_circuits,
                                 summarize)


def build_spec(device):
    from qtrail.devices import (build_grid3x3_spec, build_grid8x8_spec,
                                build_tianyan287_spec)
    dev_cfg = load_device_config()
    if device == "tianyan-287":
        return build_tianyan287_spec(dev_cfg)
    if device == "grid-8x8":
        return build_grid8x8_spec()
    return build_grid3x3_spec()


def load_policy(checkpoint, spec, dev):
    from qtrail.models import QAPolicy
    policy, ckpt = QAPolicy.load_checkpoint(checkpoint, device_n=spec.n,
                                            map_location=dev)
    policy.eval()
    return policy, ckpt


def run_one(checkpoint, spec, circuits, baselines, dev, seed):
    """对单个权重跑完整评测，返回 (summary, rows)。"""
    policy, ckpt = load_policy(checkpoint, spec, dev)
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    mapper = Mapper(spec, policy=policy, cfg=cfg, dev=dev, seed=seed)
    rows = evaluate_benchmark(circuits, mapper, baselines=baselines)
    return summarize(rows, baselines=baselines), rows


def main():
    ap = argparse.ArgumentParser(description="对比多个模型在测试集上的映射性能")
    ap.add_argument("--bench", default="queko_BIGD",
                    choices=["queko_BIGD", "queko_BNTF", "queko_BSS", "mqtbench_n15"])
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--device", default="tianyan-287")
    ap.add_argument("--model", action="append", required=True,
                    help="标签=权重路径（可重复多次）")
    ap.add_argument("--baselines", default="qiskit-o1,qiskit-o3")
    ap.add_argument("--dev", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tables/compare_models")
    args = ap.parse_args()

    dev = args.dev or "cpu"
    baselines = tuple(args.baselines.split(","))
    spec = build_spec(args.device)
    circuits = load_benchmark_circuits(args.bench, args.limit)

    models = []
    for m in args.model:
        if "=" not in m:
            print(f"[skip] --model 需「标签=路径」格式：{m}", file=sys.stderr)
            continue
        label, path = m.split("=", 1)
        if not Path(path).exists():
            print(f"[skip] 权重不存在：{label} -> {path}", file=sys.stderr)
            continue
        models.append((label, path))

    if not models:
        print("没有可用模型，退出。", file=sys.stderr)
        return 1

    print(f"== 测试集 {args.bench}（{len(circuits)} 条） @ {spec.name}，baselines={args.baselines} ==\n")
    results = {}
    for label, path in models:
        print(f"[eval] {label} ...", flush=True)
        summary, rows = run_one(path, spec, circuits, baselines, dev, args.seed)
        results[label] = summary
        # 逐线路数据落盘
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        import json
        safe = label.replace("/", "_").replace(" ", "_")
        with open(out_dir / f"{safe}_rows.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")

    # ---- 并排对比表
    def g(s, key):
        v = s.get(key)
        return v[0] if isinstance(v, tuple) and v[0] is not None else float("nan")

    def g_med(s, key):
        v = s.get(key)
        return v[1] if isinstance(v, tuple) and v[1] is not None else float("nan")

    header = f"{'指标':<16}" + "".join(f"{label:<20}" for label, _ in models)
    print("\n" + header)
    print("-" * len(header))

    def row(name, fn):
        vals = [fn(s) for s in results.values()]
        line = f"{name:<16}" + "".join(f"{v:<20.4f}" for v in vals)
        print(line)

    row("SWAP 均值", lambda s: g(s, "swaps"))
    row("SWAP 中位", lambda s: g_med(s, "swaps"))
    row("深度均值", lambda s: g(s, "depth"))
    row("2Q 门均值", lambda s: g(s, "twoq"))
    row("估计保真度", lambda s: g(s, "fidelity"))
    row("静态代价", lambda s: g(s, "static_cost"))
    for b in baselines:
        row(f"{b} SWAP", lambda s, b=b: g(s, f"{b}_swaps"))

    # 相对生产基线的劣势
    print("\n-- 相对生产基线（负 = 比基线多 SWAP）--")
    for label, s in results.items():
        our = g(s, "swaps")
        parts = []
        for b in baselines:
            bb = g(s, f"{b}_swaps")
            if bb:
                parts.append(f"{b} {100*(1 - our/bb):+.1f}%")
        print(f"  {label:<20} {'  '.join(parts)}")

    print(f"\n逐线路数据 -> {Path(args.out)}/")


if __name__ == "__main__":
    main()
