"""Run all three training configs in sequence (GAT topo -> GAT noise -> GT noise).

Usage: python scripts/run_training.py [--dev cuda] [--epochs N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.training.reinforce import train

CONFIGS = [
    ("configs/train_gat.yaml", "CO-MAP 复现基线 (GAT + one-hot + 拓扑奖励)"),
    ("configs/train_gat_noise.yaml", "噪声感知 (GAT + 校准特征 + 混合奖励)"),
    ("configs/train_gt_noise.yaml", "QTrail 完整版 (GraphTransformer + 噪声)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    dev_cfg = load_device_config()
    spec = build_tianyan287_spec(dev_cfg)

    results = []
    for path, desc in CONFIGS:
        print(f"\n{'=' * 70}\n== {desc}\n== {path}\n{'=' * 70}", flush=True)
        cfg = Config.load(path)
        cfg.device = dev_cfg
        try:
            stats = train(cfg, spec, out_dir=args.out, dev=args.dev,
                          epochs_override=args.epochs)
            results.append({"config": path, **stats})
        except Exception as e:
            print(f"[run_training] FAILED {path}: {e}", flush=True)
            results.append({"config": path, "error": str(e)})

    print("\n===== SUMMARY =====")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
