# -*- coding: utf-8 -*-
"""预路由消化 A/B（2026-11-13 QFT 归因）。

同一设备/校准/检查点/种子，比较 PureMapper(pre_cancel=False/True)：
  - 1Q / 2Q / 读出门数、深度、2Q 深度、乘积 log 保真度、墙钟。
预注册判据（研究阶段，先于结果写定）：
  P1 语义：A/B 产物在紧凑子空间状态矢量上等价（本脚本不测，由
     tests/test_pre_cancel_semantics.py 覆盖）；
  P2 QFT 族：2Q 门数下降 ≥40%，logf 不劣化，零非法；
  P3 其他族：logf 不劣化超过 2%，否则默认保持关闭 / 按族开关；
  P4 墙钟：单条不超过基线 1.5×。

用法：
  python scripts/ab_pre_cancel.py --circuits qft_N055,qft_N080 --out rows.jsonl
  python scripts/ab_pre_cancel.py --family qft --limit 6 --out rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.metrics import (_product_log_fidelity, circuit_depth,
                                 twoq_depth)
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"


def build_index():
    idx = {}
    for p in Path("data/benchpress/qasm").rglob("*.qasm"):
        idx[f"bp_{p.parent.name}_{p.stem}"] = ("bp", p)
    for p in Path("data/mqtbench/stratified").glob("*.qasm"):
        idx[p.stem] = ("mqt", p)
    for p in iter_queko_files("BIGD"):
        idx[p.stem] = ("queko", p)
    return idx


def load_text(idx, name):
    kind, p = idx[name]
    text = p.read_text(encoding="utf-8", errors="replace")
    return text if kind == "bp" else sanitize_qasm(text)


def measure(circ, spec):
    n1 = sum(1 for i in circ.ops if i.nq == 1)
    n2 = sum(1 for i in circ.ops if i.nq == 2)
    return {"1q": n1, "2q": n2, "ro": len(circ.measures),
            "depth": circuit_depth(circ), "2qdepth": twoq_depth(circ),
            "logf": round(_product_log_fidelity(circ, spec.calib), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default=None, help="逗号分隔线路名")
    ap.add_argument("--family", default=None, help="名称包含该子串")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = build_tianyan287_spec(load_device_config())
    policy, _ = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                         map_location="cpu")
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    idx = build_index()

    names = ([n.strip() for n in args.circuits.split(",") if n.strip()]
             if args.circuits else sorted(idx))
    if args.family:
        names = [n for n in names if args.family in n]
    if args.limit:
        names = names[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for name in names:
        if name not in idx:
            print(f"skip unknown {name}", flush=True)
            continue
        text = load_text(idx, name)
        c = parse_qasm(text)
        row = {"circuit": name, "n": c.n}
        for arm in (False, True):
            torch.manual_seed(args.seed)
            mapper = PureMapper(spec, policy=policy, cfg=cfg, dev="cpu",
                                seed=args.seed, selection_rule="fidelity",
                                use_post=True, pipeline="v5",
                                routing_seeds=args.seeds, pre_cancel=arm)
            t0 = time.time()
            try:
                res = mapper.map_circuit(c, circuit_id=name)
                arm_row = measure(res["routed_circuit"], spec)
                arm_row["swaps"] = res["swap_count"]
                arm_row["wall"] = round(time.time() - t0, 2)
            except Exception as e:  # noqa: BLE001
                arm_row = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            row["pre_cancel" if arm else "base"] = arm_row
        b = row.get("base", {})
        p = row.get("pre_cancel", {})
        if "2q" in b and "2q" in p:
            row["d2q"] = b["2q"] - p["2q"]
            row["ratio_2q"] = p["2q"] / max(b["2q"], 1)
            row["dlogf"] = round(p["logf"] - b["logf"], 4)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name:<30} base 2q={b.get('2q')} logf={b.get('logf')} | "
              f"pre 2q={p.get('2q')} logf={p.get('logf')} | "
              f"ratio={row.get('ratio_2q')} dlogf={row.get('dlogf')}",
              flush=True)


if __name__ == "__main__":
    main()
