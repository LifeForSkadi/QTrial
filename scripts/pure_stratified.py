"""pure 管线全分层重测：MQTBench ≤10/11-50/51-105 + QUEKO dense/sparse。

模型 = postfid_ft（A/B 门禁胜出）；PureMapper（fidelity 规则，use_post）。
对照 = qiskit 支撑版参照数据（target_post seeds8_top3 + final_stratified）。
输出：tables/pure_stratified/
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

CKPT = ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_"
        "postfid_ft_best.pt")
import argparse as _ap
_ap_ = _ap.ArgumentParser(add_help=False)
_ap_.add_argument("--ckpt", default=None)
_ckpt_arg_, _ = _ap_.parse_known_args()
if _ckpt_arg_.ckpt:
    CKPT = _ckpt_arg_.ckpt
CACHE = Path("data/mqtbench/stratified")
OUT = Path("tables/pure_stratified")
LARGE_PICK = {"qft_105", "qaoa_105", "dj_105", "wstate_105", "ghz_105"}


def circuits():
    small, med, large, dense, sparse = [], [], [], [], []
    for p in sorted(CACHE.glob("*.qasm")):
        name = p.stem
        size = int(name.rsplit("_", 1)[1])
        text = p.read_text(encoding="utf-8")
        if size <= 10:
            small.append((name, text))
        elif size <= 50:
            med.append((name, text))
        elif name in LARGE_PICK:
            large.append((name, text))
    for p in iter_queko_files("BIGD"):
        if re.search(r"\.(3|4)D1_\.(4|5)D2_", p.name):
            dense.append((p.stem, p.read_text(encoding="utf-8")))
        elif re.search(r"\.0D1_\.1D2_", p.name):
            sparse.append((p.stem, p.read_text(encoding="utf-8")))
    return {"mqt-small": small, "mqt-med": med, "mqt-large": large,
            "queko-dense": dense, "queko-sparse": sparse}


def load_ref():
    """qiskit 版参照（target_post 结果表，若存在）。"""
    ref = {}
    for g in ("queko-dense", "queko-sparse"):
        p = OUT.parent / "target_post" / "seeds8_top3" / f"{g}_rows.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                ref[r["circuit"]] = r.get("tp_fidelity")
    for g in ("small", "med", "large"):
        p = OUT.parent / "final_stratified" / f"main_{g}_rows.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                ref[r["circuit"]] = r.get("qt_fidelity")
    return ref


def main():
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    policy, ck = QAPolicy.load_checkpoint(
        CKPT, device_n=spec.n,
        map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    buckets = circuits()
    ref = load_ref()
    OUT.mkdir(parents=True, exist_ok=True)
    sections = ["# pure 管线全分层重测（零 qiskit，天衍-287）\n",
                f"模型 = {Path(CKPT).name}；PureMapper（fidelity 规则 + 完整后处理栈）；"
                "对照列 = qiskit 支撑版（内部参照）；统计：均值/截尾(去10%)/中位数\n"]

    for bname, circuits_all in buckets.items():
        rows = []
        done_names = set()
        done_path = OUT / f"{bname}_rows.jsonl"
        if done_path.exists():
            for line in done_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done_names.add(json.loads(line)["circuit"])
        for name, text in circuits_all:
            if name in done_names:  # 断点续跑
                continue
            circ = parse_qasm(text)
            t0 = time.time()
            row = {"circuit": name, "n": circ.n}
            try:
                mapper = PureMapper(spec, policy=policy, cfg=cfg, dev="cuda",
                                    seed=42, selection_rule="fidelity",
                                    use_post=True)
                res = mapper.map_circuit(circ, circuit_id=name)
                row["pure_swaps"] = res["swap_count"]
                row["pure_depth"] = res["metrics"]["depth"]
                row["pure_twoq_depth"] = res["metrics"]["twoq_depth"]
                row["pure_fidelity"] = res["metrics"]["est_fidelity"]
                row["pure_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["pure_error"] = f"{type(e).__name__}: {str(e)[:60]}"
            row["ref_fidelity"] = ref.get(name)
            rows.append(row)
            with open(OUT / f"{bname}_rows.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                f.flush()
            print(f"  [{bname}] {name:36s} swaps={row.get('pure_swaps', '-')} "
                  f"fid={row.get('pure_fidelity', '-')} wall={row.get('pure_wall', '-')}s",
                  flush=True)
        # 续跑桶：合并存量行（汇总用全量数据）
        if done_names:
            existing = {}
            for line in done_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    existing[r["circuit"]] = r
            merged = {r["circuit"]: r for r in rows}
            merged.update(existing)
            rows = list(merged.values())
        # 汇总
        def trim(vals):
            vals = sorted(vals)
            if not vals:
                return None
            k = max(1, int(len(vals) * 0.1))
            if 2 * k >= len(vals):
                return statistics.mean(vals)
            return statistics.mean(vals[k:len(vals) - k])

        pf = [r["pure_fidelity"] for r in rows if isinstance(r.get("pure_fidelity"), (int, float))]
        rf = [r["ref_fidelity"] for r in rows if isinstance(r.get("ref_fidelity"), (int, float))]
        wins = sum(1 for r in rows if isinstance(r.get("pure_fidelity"), (int, float))
                   and isinstance(r.get("ref_fidelity"), (int, float))
                   and r["pure_fidelity"] > r["ref_fidelity"])
        nref = sum(1 for r in rows if isinstance(r.get("ref_fidelity"), (int, float)))
        sw = [r["pure_swaps"] for r in rows if isinstance(r.get("pure_swaps"), (int, float))]
        sections.append(f"\n## {bname}（{len(rows)} 条线路）\n")
        sections.append(f"- pure SWAP：均值 {statistics.mean(sw):.1f} / 截尾 {trim(sw):.1f} / "
                        f"中位 {statistics.median(sw):.1f}")
        sections.append(f"- pure 保真度：均值 {statistics.mean(pf):.4f} / 截尾 {trim(pf):.4f}")
        if rf:
            sections.append(f"- 对照（qiskit 版）保真度：均值 {statistics.mean(rf):.4f} / "
                            f"截尾 {trim(rf):.4f}")
            sections.append(f"- pure vs 对照：**{wins}/{nref} 胜**")
        sections.append("")
        print(f"[pure_stratified] {bname} done", flush=True)
    (OUT / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"report -> {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
