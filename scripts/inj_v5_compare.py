# -*- coding: utf-8 -*-
"""实验 1：v5 纯管线 vs RL 布局+感知 O3（inj 口径）在 layout_inject 抽样集上。

线路集 = tables/compare_models/layout_inject_{benchpress,mqtbench_n15,
queko_BIGD}_rows.jsonl 的 84 条（inj 字段为已测的"本方案布局+感知O3"）。
本脚本跑 v5（c0.05 权重、seed=42、fidelity 规则、合成口径）逐线路对比。
输出：tables/compare_models/inj_v5_rows.jsonl + inj_v5_summary.md
断点续跑。
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import load_qasm_file

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
SRC_FILES = ["layout_inject_benchpress_rows.jsonl",
             "layout_inject_mqtbench_n15_rows.jsonl",
             "layout_inject_queko_BIGD_rows.jsonl"]
OUT = Path("tables/compare_models/inj_v5_rows.jsonl")
SUMMARY = Path("tables/compare_models/inj_v5_summary.md")


def find_file(name):
    if name.startswith("benchpress/"):
        parts = name.split("/")
        p = Path("data/benchpress/qasm") / parts[1] / f"{parts[2]}.qasm"
        return p if p.exists() else None
    if name.startswith("mqtbench/"):
        p = Path("data/mqtbench/stratified") / f"{name.split('/')[1]}.qasm"
        return p if p.exists() else None
    if name.startswith("queko_BIGD/"):
        p = Path("data/Queko/BIGD") / f"{name.split('/')[1]}.qasm"
        return p if p.exists() else None
    return None


def trimmed(vals):
    vals = sorted(vals)
    k = max(1, int(len(vals) * 0.1))
    return (statistics.mean(vals[k:len(vals) - k])
            if 2 * k < len(vals) else statistics.mean(vals))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dev={dev}", flush=True)
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    cases = []
    for fn in SRC_FILES:
        p = Path("tables/compare_models") / fn
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cases.append((r["circuit"], r))
    print(f"线路 {len(cases)} 条", flush=True)

    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["circuit"]] = r

    rows = list(done.values())
    for i, (name, inj) in enumerate(cases):
        if name in done:
            continue
        src = find_file(name)
        row = {"circuit": name, "n": inj.get("n_qubits")}
        if src is None:
            row["v5_error"] = "file missing"
        else:
            try:
                torch.manual_seed(42)
                circ = load_qasm_file(src)
                mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev,
                                    seed=42, selection_rule="fidelity",
                                    pipeline="v5")
                t0 = time.time()
                res = mapper.map_circuit(circ, circuit_id=name)
                row["v5_fidelity"] = res["metrics"]["est_fidelity"]
                row["v5_depth"] = res["metrics"]["depth"]
                row["v5_swaps"] = res["swap_count"]
                row["v5_twoq"] = res["metrics"]["twoq_count"]
                row["v5_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["v5_error"] = f"{type(e).__name__}: {str(e)[:60]}"
        row["inj_fidelity"] = inj.get("inj_fidelity")
        rows.append(row)
        with open(OUT, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(cases)} ({name})", flush=True)
    print(f"完成 -> {OUT}", flush=True)

    # 汇总
    ok = [r for r in rows if "v5_fidelity" in r and r.get("inj_fidelity")]
    lines = ["# 实验 1：v5 纯管线 vs RL 布局+感知 O3（inj）\n",
             "线路集：layout_inject 三基准抽样（84 条）；v5 = c0.05 权重 + "
             "fidelity 规则 + 合成口径。\n"]
    for group in ("benchpress", "mqtbench", "queko"):
        sub = [r for r in ok if r["circuit"].startswith(group)]
        if not sub:
            continue
        v5 = [r["v5_fidelity"] for r in sub]
        ij = [r["inj_fidelity"] for r in sub]
        wins = sum(1 for r in sub if r["v5_fidelity"] > r["inj_fidelity"])
        lines.append(f"\n## {group}（{len(sub)} 条）\n")
        lines.append("| 口径 | 保真度（均/截/中） |\n|---|---|\n")
        lines.append(f"| v5 | {statistics.mean(v5):.4f}/{trimmed(v5):.4f}/"
                     f"{statistics.median(v5):.4f} |\n")
        lines.append(f"| inj（RL布局+感知O3） | {statistics.mean(ij):.4f}/"
                     f"{trimmed(ij):.4f}/{statistics.median(ij):.4f} |\n")
        lines.append(f"| 比值 v5/inj | "
                     f"{statistics.mean(v5)/statistics.mean(ij):.2f}x |\n")
        lines.append(f"| 逐条胜场 | v5 赢 {wins}/{len(sub)} |\n")
    v5all = [r["v5_fidelity"] for r in ok]
    ijall = [r["inj_fidelity"] for r in ok]
    lines.append(f"\n## 总体（{len(ok)} 条）\n")
    lines.append(f"- v5 {statistics.mean(v5all):.4f} vs inj "
                 f"{statistics.mean(ijall):.4f} -> "
                 f"{statistics.mean(v5all)/statistics.mean(ijall):.2f}x；"
                 f"胜场 {sum(1 for r in ok if r['v5_fidelity'] > r['inj_fidelity'])}/{len(ok)}\n")
    SUMMARY.write_text("".join(lines), encoding="utf-8")
    print(f"汇总 -> {SUMMARY}")


if __name__ == "__main__":
    main()
