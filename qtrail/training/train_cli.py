"""Training entry point: python -m qtrail.train --config configs/train_gat.yaml"""
from __future__ import annotations

import argparse

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.training.reinforce import train


def main(argv=None):
    ap = argparse.ArgumentParser(description="QTrial policy training")
    ap.add_argument("--config", required=True, help="training config yaml")
    ap.add_argument("--device-config", default=None, help="device config yaml "
                    "(default: configs/device_tianyan287.yaml)")
    ap.add_argument("--out", default=None, help="checkpoint output dir")
    ap.add_argument("--resume", default=None, help="resume from checkpoint")
    ap.add_argument("--dev", default=None, choices=["cpu", "cuda"], help="device")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs")
    ap.add_argument("--device-name", default="tianyan-287", choices=[
        "tianyan-287", "grid-8x8", "grid-3x3"], help="which device spec to build")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    if args.device_config:
        dev_cfg = load_device_config(args.device_config)
    else:
        dev_cfg = load_device_config()

    if cfg.train.device_pool:
        # 多拓扑混合训练：按名单构建设备池
        pool = [_build_device(name, dev_cfg) for name in cfg.train.device_pool]
        train(cfg, pool, out_dir=args.out, resume=args.resume, dev=args.dev,
              epochs_override=args.epochs)
        return

    if args.device_name == "tianyan-287":
        spec = build_tianyan287_spec(dev_cfg)
    elif args.device_name == "grid-8x8":
        from qtrail.devices import build_grid8x8_spec
        spec = build_grid8x8_spec(seed=dev_cfg.calibration_seed)
    else:
        from qtrail.devices import build_grid3x3_spec
        spec = build_grid3x3_spec(seed=dev_cfg.calibration_seed)
    cfg.device = dev_cfg

    train(cfg, spec, out_dir=args.out, resume=args.resume, dev=args.dev,
          epochs_override=args.epochs)


def _build_device(name: str, dev_cfg):
    from qtrail.devices import (build_tianyan287_spec, build_grid8x8_spec,
                                build_grid3x3_spec, build_sycamore53_spec,
                                build_heavyhex_spec, build_grid_family_spec)
    if name == "tianyan-287":
        return build_tianyan287_spec(dev_cfg)
    if name == "grid-8x8":
        return build_grid8x8_spec(seed=dev_cfg.calibration_seed)
    if name == "grid-3x3":
        return build_grid3x3_spec(seed=dev_cfg.calibration_seed)
    if name == "sycamore-53":
        return build_sycamore53_spec()
    if name == "heavy-hex-115":
        return build_heavyhex_spec(7)
    if name.startswith("grid-"):
        rows, cols = name[5:].split("x")
        return build_grid_family_spec(int(rows), int(cols),
                                      seed=dev_cfg.calibration_seed)
    raise ValueError(f"unknown device preset: {name}")


if __name__ == "__main__":
    main()
