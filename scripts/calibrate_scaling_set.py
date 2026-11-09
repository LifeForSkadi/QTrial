# -*- coding: utf-8 -*-
"""规模-难度网格测试集校准：实测各格子 SWAP 门数并校验分层。

验收口径 = 本方案布局+O1 路由（与图11/12 的蓝色系列同一管线）的
逐线路 SWAP 门数；每格 5 个种子的中位数须落在 BANDS 区间内。
输出：data/scaling/calibration.jsonl + 分层校验表。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.mapper import Mapper
from qtrail.utils.qasm_io import strip_measurements

DEFAULT_CKPT = ("checkpoints/demo_laurel/"
                "tianyan-287_gat_combined_calib_dep0.1_t0.5_best.pt")

# 验收区间：(size, difficulty) -> (lo, hi)，每格 5 条种子中位数判定。
# 区间为生成器迭代两轮后的实测口径（low≈0、mid≈0.5n-1.5n、
# high≈2n-5n），保证三档分离且有充分的动态范围。
BANDS = {
    (10, "low"): (0, 6), (20, "low"): (0, 8), (40, "low"): (0, 8),
    (60, "low"): (0, 8), (80, "low"): (0, 8), (105, "low"): (0, 8),
    (10, "mid"): (3, 10), (20, "mid"): (8, 20), (40, "mid"): (30, 70),
    (60, "mid"): (60, 150), (80, "mid"): (120, 250),
    (105, "mid"): (180, 400),
    (10, "high"): (8, 20), (20, "high"): (20, 60), (40, "high"): (60, 140),
    (60, "high"): (120, 280), (80, "high"): (200, 480),
    (105, "high"): (350, 800),
}


def main():
    spec = build_tianyan287_spec()
    policy, ckpt = QAPolicy.load_checkpoint(
        DEFAULT_CKPT, device_n=spec.n, map_location="cuda")
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda", seed=42)

    qdir = Path("data/scaling/qasm")
    files = sorted(qdir.glob("*.qasm"))
    print(f"calibrating {len(files)} circuits ...", flush=True)
    t0 = time.time()
    cells: dict = {}
    with open(Path("data/scaling/calibration.jsonl"), "w",
              encoding="utf-8") as f:
        for p in files:
            parts = p.stem.split("_")            # s{size}_{diff}_k{k}
            size = int(parts[0][1:])
            diff = parts[1]
            qc, _ = strip_measurements(
                qasm2.loads(p.read_text(encoding="utf-8")))
            try:
                res = mapper.map_circuit(qc, circuit_id=p.stem,
                                         optimization_level=1)
                swaps = res.swap_count
            except Exception as e:
                swaps = None
                print(f"  {p.stem} FAILED: {type(e).__name__}: "
                      f"{str(e)[:60]}", flush=True)
            f.write(json.dumps({"file": p.name, "size": size,
                                "difficulty": diff,
                                "swaps": swaps}, ensure_ascii=False) + "\n")
            cells.setdefault((size, diff), []).append(swaps)
            print(f"  {p.stem:24s} swaps={swaps} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print("\n==== 分层校验（每格 5 条种子的 SWAP 中位数，须落区间）====")
    n_bad = 0
    for size in (10, 20, 40, 60, 80, 105):
        line = []
        for diff in ("low", "mid", "high"):
            vals = [v for v in cells.get((size, diff), []) if v is not None]
            med = statistics.median(vals) if vals else None
            lo, hi = BANDS[(size, diff)]
            ok = med is not None and lo <= med <= hi
            if not ok:
                n_bad += 1
            line.append(f"{diff}: med={med} band=[{lo},{hi}] {'OK' if ok else 'MISS'}")
        print(f"n={size:3d} | " + " | ".join(line))
    print(f"\n{'全部格子达标' if n_bad == 0 else f'{n_bad} 个格子未达标，需调整生成参数'}")


if __name__ == "__main__":
    main()
