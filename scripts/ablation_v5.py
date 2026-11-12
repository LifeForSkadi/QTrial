# -*- coding: utf-8 -*-
"""v5 管线消融测试（2026-11-09，改进完成后按用户指令执行）。

组件贡献分解：v5 完整（B+A2+C1+L3）vs 逐个关闭组件。
线路子集：v3 修复后口径三层各抽 10 条（固定种子抽取，清单落盘）。
基线（感知/盲目 O1/O3、pytket）不重跑——对比读旧表。
输出：tables/ablation_v5/rows.jsonl + summary.md
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.qasm import load_qasm_file
from qtrail.pure.router import sabre_route
from qtrail.pure.post import post_route, decompose_to_platform
from qtrail.pure.post_v2 import (collect_resynthesize_2q_blocks,
                                 _decompose_ablation,
                                 learned_relabel, noise_aware_relabel)
from qtrail.pure.metrics import compute_metrics

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OUT = Path("tables/ablation_v5")
VARIANTS = ["v4", "v5", "v5-noA1", "v5-noA2", "v5-noA3",
            "v5-noC1", "v5-noL3", "v5-noB"]


def find_file(name):
    if name.startswith("bp_"):
        # 命名 = bp_{类别}_{文件名}（文件名可含下划线）
        parts = name.split("_", 2)
        if len(parts) == 3:
            cat, stem = parts[1], parts[2]
            p = Path("data/benchpress/qasm") / cat / f"{stem}.qasm"
            if p.exists():
                return p
        return None
    for d in ("data/mqtbench/stratified", "data/Queko/BIGD"):
        p = Path(d) / f"{name}.qasm"
        if p.exists():
            return p
    return None


def run_variant(policy, cfg, spec, circ, variant, seed=42):
    """固定布局（平凡）下跑各变体后段——消融聚焦组件本身，布局噪声
    由同一 routed 输入消除。"""
    from qtrail.pure.post_v2 import fold_and_relink
    routed, swaps, fin = sabre_route(circ, spec,
                                     {i: i for i in range(circ.n)}, seed=seed)
    post_route(routed, fin, spec.adj)
    if variant == "v4":
        final_qc = decompose_to_platform(routed)
        return compute_metrics(final_qc, routed.count("swap"), spec.calib)
    r = routed.copy()
    f = dict(fin)
    if "noA1" not in variant:
        folded, fold_arr = fold_and_relink(r, spec)
        if fold_arr is not None:
            r = folded
            for k, v in f.items():
                f[k] = int(fold_arr[int(v)])
    if "noC1" not in variant:
        noise_aware_relabel(r, spec, f)
    if "noL3" not in variant and policy is not None:
        learned_relabel(policy, spec, r, f, cfg, dev="cpu", seed=seed)
    if "noA2" not in variant:
        r = collect_resynthesize_2q_blocks(r, spec)
    final_qc = _decompose_ablation(
        r, cancel_cx=("noA3" not in variant),
        fixpoint=("noB" not in variant))
    return compute_metrics(final_qc, r.count("swap"), spec.calib)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    # 子集：三层各抽 10 条（固定种子）
    rng = random.Random(7)
    subset = []
    for bucket in ("small", "medium", "large"):
        rows = [json.loads(l) for l in
                Path(f"tables/final_report_v3/{bucket}_fidelity_rows.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        picked = rng.sample(rows, min(10, len(rows)))
        subset += [(bucket, r["circuit"], r["n"]) for r in picked]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subset.json").write_text(
        json.dumps(subset, ensure_ascii=False, indent=1), encoding="utf-8")

    rows = []
    for bucket, name, n in subset:
        p = find_file(name)
        if p is None:
            print(f"[skip] 文件缺失 {name}", flush=True)
            continue
        try:
            circ = load_qasm_file(p)
        except Exception as e:
            print(f"[skip] 解析失败 {name}: {e}", flush=True)
            continue
        if circ.n > 105:
            continue
        row = {"bucket": bucket, "circuit": name, "n": circ.n}
        for variant in VARIANTS:
            torch.manual_seed(42)
            t0 = time.time()
            try:
                m = run_variant(policy, cfg, spec, circ, variant)
                row[f"{variant}_fid"] = m["est_fidelity"]
                row[f"{variant}_swap"] = m["swap_count"]
                row[f"{variant}_depth"] = m["depth"]
                row[f"{variant}_2qd"] = m["twoq_depth"]
                row[f"{variant}_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row[f"{variant}_error"] = f"{type(e).__name__}: {e}"
            print(f"  [{bucket}] {name[:40]} {variant} "
                  f"fid={row.get(f'{variant}_fid', 'err')}", flush=True)
        rows.append(row)
        with open(OUT / "rows.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总：v5 与各消融变体的均值/截尾/中位 + vs v4 提升
    import statistics

    def trimmed(vals):
        vals = sorted(vals)
        k = max(1, int(len(vals) * 0.1))
        return (statistics.mean(vals[k:len(vals) - k])
                if 2 * k < len(vals) else statistics.mean(vals))

    lines = ["# v5 管线消融汇总（2026-11-09）\n",
             "口径：平凡布局（组件本身效果，不含布局）；三层各 10 条抽样；\n",
             "变体：v4 基线 / v5 完整（B+A2+C1+L3）/ 逐组件关闭。\n"]
    for bucket in ("small", "medium", "large"):
        sub = [r for r in rows if r["bucket"] == bucket]
        if not sub:
            continue
        lines.append(f"\n## {bucket}（{len(sub)} 条）\n")
        lines.append("| 变体 | 保真度（均/截/中） | vs v4 |\n")
        lines.append("|---|---|---|\n")
        v4_fid = [r["v4_fid"] for r in sub if "v4_fid" in r]
        for v in VARIANTS:
            vals = [r[f"{v}_fid"] for r in sub if f"{v}_fid" in r]
            if not vals:
                continue
            gain = (statistics.mean(vals) / statistics.mean(v4_fid) - 1) * 100 \
                if v != "v4" and v4_fid and statistics.mean(v4_fid) > 1e-300 \
                else 0.0
            lines.append(f"| {v} | {statistics.mean(vals):.4e}/"
                         f"{trimmed(vals):.4e}/{statistics.median(vals):.4e}"
                         f" | {gain:+.1f}% |\n")
    (OUT / "summary.md").write_text("".join(lines), encoding="utf-8")
    print(f"\n汇总 -> {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
