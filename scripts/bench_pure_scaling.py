# -*- coding: utf-8 -*-
"""自研方案（QTrial 纯管线，零 qiskit）在规模-难度网格测试集上的运行。

交付管线 = 交付检查点 + 合成 287 规格 + PureMapper（fidelity 规则、
置换重写后处理开启），逐线路记录墙钟/交换数/深度/估计保真度。
供图11/图12 的运行时间折线对比加入"自研方案"系列。

逐条落盘 + 断点续跑（已含 wall_s 的线路跳过）。
用法：python scripts/bench_pure_scaling.py
输出：tables/compare_models/layout_pure_scaling_rows.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
RULE = "fidelity"


def main():
    spec = build_tianyan287_spec()
    policy, ck = QAPolicy.load_checkpoint(
        CKPT, device_n=spec.n, map_location="cuda")
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    qdir = Path("data/scaling/qasm")
    files = sorted(qdir.glob("*.qasm"))
    out = Path("tables/compare_models/layout_pure_scaling_rows.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["circuit"]] = r

    def write_rows(rows):
        with open(out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = []
    t0 = time.time()
    for p in files:
        name = f"scaling/{p.stem}"
        if name in done and "wall_s" in done[name]:
            rows.append(done[name])
            print(f"  [skip] {p.stem}", flush=True)
            write_rows(rows)
            continue
        text = p.read_text(encoding="utf-8")
        circ = parse_qasm(text)
        row = {"circuit": name, "n_qubits": circ.n}
        try:
            mapper = PureMapper(spec, policy=policy, cfg=cfg, dev="cuda",
                                seed=42, selection_rule=RULE, use_post=True)
            t1 = time.time()
            res = mapper.map_circuit(circ, circuit_id=name)
            row["wall_s"] = round(time.time() - t1, 3)
            row["swaps"] = res["swap_count"]
            row["depth"] = res["metrics"]["depth"]
            row["twoq"] = res["metrics"].get("twoq_count")
            row["fidelity"] = res["metrics"]["est_fidelity"]
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:60]}"
        rows.append(row)
        write_rows(rows)
        print(f"  {p.stem:24s} wall={row.get('wall_s')} "
              f"swaps={row.get('swaps')} fid={row.get('fidelity')} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"saved -> {out} ({len(rows)} 行)")


if __name__ == "__main__":
    main()
