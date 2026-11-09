# -*- coding: utf-8 -*-
"""最终分层报告 v5：QTrial v5 管线（B+A2+C1+L3+非劣化开关）全量重测。

与 v3 同线路集（Benchpress ≤105 + MQTBench stratified + QUEKO 子集），
同 c0.05 权重、seed=42、fidelity 规则。基线（感知/盲目 O1/O3、pytket）
不重跑——对比时读旧表（tables/final_report/）。

输出：tables/final_report_v5/（{bucket}_fidelity_rows.jsonl + summary.md，
含 v3 修复后口径与感知 O3 对照列）。断点续跑。
"""
from __future__ import annotations

import json
import re
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
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OUT = Path("tables/final_report_v5")
V3 = Path("tables/final_report_v3")
OLD = Path("tables/final_report")
CACHE = Path("data/mqtbench/stratified")
BP = Path("data/benchpress/qasm")
RULE = "fidelity"


def load_circuits():
    from qiskit import qasm2  # noqa: F401（与 v3 完全同口径）
    buckets = {"small": [], "medium": [], "large": []}
    skipped = 0
    for p in BP.rglob("*.qasm"):
        text = p.read_text(encoding="utf-8", errors="replace")
        try:
            circ = parse_qasm(text)
            if circ.n == 0:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        if circ.n > 105:
            skipped += 1
            continue
        b = "small" if circ.n <= 10 else ("medium" if circ.n <= 50 else "large")
        buckets[b].append((f"bp_{p.parent.name}_{p.stem}", text))
    for p in sorted(CACHE.glob("*.qasm")):
        name = p.stem
        size = int(name.rsplit("_", 1)[1])
        if size > 105:
            continue
        b = "small" if size <= 10 else ("medium" if size <= 50 else "large")
        buckets[b].append((name, sanitize_qasm(p.read_text(encoding="utf-8"))))
    for p in iter_queko_files("BIGD"):
        if re.search(r"\.(3|4)D1_\.(4|5)D2_", p.name) \
                or re.search(r"\.0D1_\.1D2_", p.name):
            buckets["medium"].append(
                (p.stem, sanitize_qasm(p.read_text(encoding="utf-8"))))
    return buckets, skipped


def trimmed(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * 0.1))
    return (statistics.mean(vals[k:len(vals) - k])
            if 2 * k < len(vals) else statistics.mean(vals))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="small,medium,large",
                    help="只跑指定层（并行分片用：如 --buckets large）")
    args = ap.parse_args()
    buckets_sel = [b for b in args.buckets.split(",") if b]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dev={dev} buckets={buckets_sel}", flush=True)
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    buckets, skipped = load_circuits()
    total = sum(len(v) for v in buckets.values())
    print(f"circuits: small {len(buckets['small'])} / medium "
          f"{len(buckets['medium'])} / large {len(buckets['large'])} "
          f"(total {total}, skipped {skipped})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for bucket in buckets_sel:
        rows = []
        out_file = OUT / f"{bucket}_{RULE}_rows.jsonl"
        done_names = set()
        if out_file.exists():
            for line in out_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done_names.add(r["circuit"])
                    rows.append(r)
        n_done = len(done_names)
        for i, (name, ptext) in enumerate(buckets[bucket]):
            if name in done_names:
                continue
            t0 = time.time()
            row = {"circuit": name, "n": parse_qasm(ptext).n}
            try:
                torch.manual_seed(42)  # 布局确定性（与 v3 同口径）
                mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev,
                                    seed=42, selection_rule=RULE, use_post=True,
                                    pipeline="v5")
                res = mapper.map_circuit(parse_qasm(ptext), circuit_id=name)
                row["qt_swaps"] = res["swap_count"]
                row["qt_depth"] = res["metrics"]["depth"]
                row["qt_twoq_depth"] = res["metrics"]["twoq_depth"]
                row["qt_fidelity"] = res["metrics"]["est_fidelity"]
                row["qt_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["qt_error"] = f"{type(e).__name__}: {str(e)[:60]}"
            rows.append(row)
            with open(out_file, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i + 1 + n_done) % 5 == 0:
                print(f"  [{bucket}] {i + 1 + n_done}/{len(buckets[bucket])} "
                      f"({name})", flush=True)
        print(f"[{bucket}] 完成 {len(buckets[bucket])} 条 -> {out_file}",
              flush=True)

    # ---- 汇总（v5 vs v3 修复后 vs 感知 O3）
    titles = {"small": "≤10 比特", "medium": "11-50 比特",
              "large": "51-105 比特"}
    v3 = {}
    old = {}
    for bucket in ("small", "medium", "large"):
        for src, store in ((V3, v3), (OLD, old)):
            p = src / f"{bucket}_{RULE}_rows.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        store[r["circuit"]] = r
    lines = ["# QTrial v5 管线全量汇总（2026-11-09）\n",
             "v5 = B（1Q 游程）+ A2（2Q 块重合成）+ C1（噪声感知重标记）+ "
             "L3（学习式重标记）+ 非劣化开关；基线不重跑，感知 O3 读旧表。\n"]
    for bucket in ("small", "medium", "large"):
        if bucket not in buckets_sel and not (OUT / f"{bucket}_{RULE}_rows.jsonl").exists():
            continue
        p = OUT / f"{bucket}_{RULE}_rows.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in
                p.read_text(encoding="utf-8").splitlines() if l.strip()]
        ok = [r for r in rows if "qt_fidelity" in r]
        n_err = len(rows) - len(ok)

        def stat(key):
            vals = [r[key] for r in ok if key in r]
            if not vals:
                return None
            return (statistics.mean(vals), trimmed(vals),
                    statistics.median(vals))

        def fmt(t):
            return (f"{t[0]:.4f}/{t[1]:.4f}/{t[2]:.4f}"
                    if t and t[0] is not None else "-")

        v3_ok = [v3[r["circuit"]] for r in ok
                 if r["circuit"] in v3 and "qt_fidelity" in v3[r["circuit"]]]
        aw_ok = [old[r["circuit"]] for r in ok
                 if r["circuit"] in old and "aw_o3_fidelity" in old[r["circuit"]]]

        def stat2(rows_, key):
            vals = [r[key] for r in rows_]
            if not vals:
                return None
            return (statistics.mean(vals), trimmed(vals),
                    statistics.median(vals))

        lines.append(f"\n## {titles[bucket]}（{len(ok)} 条完成，{n_err} 条错误）\n")
        lines.append("| 指标（均值/截尾/中位） | v5 | v3 修复后 | 感知 O3 |\n")
        lines.append("|---|---|---|---|\n")
        for label, key in (("SWAP 数", "qt_swaps"), ("深度", "qt_depth"),
                           ("2Q 深度", "qt_twoq_depth"),
                           ("估计保真度", "qt_fidelity")):
            lines.append(f"| {label} | {fmt(stat(key))} | "
                         f"{fmt(stat2(v3_ok, key))} | "
                         f"{fmt(stat2(aw_ok, 'aw_o3_' + ('swaps' if key == 'qt_swaps' else 'fidelity' if key == 'qt_fidelity' else 'depth' if key == 'qt_depth' else 'twoq_depth')))} |\n")
    (OUT / "summary.md").write_text("".join(lines), encoding="utf-8")
    print(f"\n汇总 -> {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
