# -*- coding: utf-8 -*-
"""决赛提交版聚合：v8 生产管线 × 全部基线。

输出：
  tables/submission/summary.md   人类可读分层对比表（报告/PPT 数字来源）
  tables/submission/summary.json 机器可读（作图脚本消费）
  tables/submission/per_circuit.jsonl 合并后的逐线路数据

口径：
- v8 = tables/final_report_v8/{bucket}_fidelity_rows*.jsonl（已合并去重）
- 基线 = tables/final_report/{bucket}_fidelity_rows.jsonl（aw_o1/aw_o3/b_o1/b_o3/tket）
- RL+感知O3 = tables/compare_models/rl_o3_587_rows.jsonl（small+medium）
- 保真度为乘积代理；large 层下溢为 0，跨方法比较看 2Q 深度/深度
- 基线 SWAP 为 qiskit count_ops("swap") 口径（恒 0），与 pytket/QTrial 不同口径
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

V8 = Path("tables/final_report_v8")
BASE = Path("tables/final_report")
RL = Path("tables/compare_models/rl_o3_587_rows.jsonl")
OUT = Path("tables/submission")

BUCKETS = ("small", "medium", "large")
CN = {"small": "≤10 比特", "medium": "11-50 比特", "large": "51-105 比特"}


def load_lines(p: Path) -> dict:
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["circuit"]] = r
    return out


def load_bucket(base: Path, bucket: str) -> dict:
    out = load_lines(base / f"{bucket}_fidelity_rows.jsonl")
    for sp in sorted(base.glob(f"{bucket}_fidelity_rows_shard*.jsonl")):
        out.update(load_lines(sp))
    return out


def trimmed(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * 0.1))
    return statistics.mean(vals[k:len(vals) - k]) if 2 * k < len(vals) else statistics.mean(vals)


def stat(rows, prefix, key):
    vals = [r[f"{prefix}_{key}"] for r in rows
            if f"{prefix}_{key}" in r and r[f"{prefix}_{key}"] is not None]
    if not vals:
        return None
    return {"mean": statistics.mean(vals), "trimmed": trimmed(vals),
            "median": statistics.median(vals), "n": len(vals)}


def fmt(s):
    if not s:
        return "-"
    return f"{s['mean']:.4f} / {s['trimmed']:.4f} / {s['median']:.4f}"


def ratio_stats(pairs):
    """pairs: list[(x, y)] → 中位比值与胜率。"""
    if not pairs:
        return None
    ratios = [x / y for x, y in pairs if y not in (0, None) and x is not None]
    if not ratios:
        return None
    wins = sum(1 for x, y in pairs if x is not None and y is not None and x > y)
    return {"median_ratio": statistics.median(ratios), "win_rate": wins / len(pairs),
            "n": len(pairs)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rl = load_lines(RL)
    summary = {"buckets": {}, "meta": {
        "v8_rows": 0, "baseline_note": "baseline read from tables/final_report"}}
    md = ["# QTrial 决赛提交版分层汇总（v8 生产管线）\n"]
    md.append("\n> 口径：保真度为逐门乘积代理；large 层乘积下溢，跨方法请以 2Q 深度/深度为准。\n"
              "> 基线 SWAP 列为 qiskit `count_ops(\"swap\")` 口径（swap 已分解为 cz，恒 0），与 pytket/QTrial 不可直接比较。\n")

    all_rows = []
    for bucket in BUCKETS:
        rows = list(load_bucket(V8, bucket).values())
        ok = [r for r in rows if "qt_fidelity" in r]
        base = load_bucket(BASE, bucket)
        base_rows = [base[r["circuit"]] for r in ok if r["circuit"] in base]
        rl_rows = [rl[r["circuit"]] for r in ok if r["circuit"] in rl]
        summary["meta"]["v8_rows"] += len(ok)

        entry = {"n_ok": len(ok), "n_err": len(rows) - len(ok), "stats": {}, "ratios": {}}
        md.append(f"\n## {CN[bucket]}（完成 {len(ok)} 条，超时/错误 {len(rows) - len(ok)} 条）\n")
        md.append("| 方法 | SWAP 数 | 深度 | 2Q 深度 | 估计保真度 |")
        md.append("|---|---|---|---|---|")
        specs = [("本方案 v8", ok, "qt"), ("噪声感知 O3", base_rows, "aw_o3"),
                 ("噪声感知 O1", base_rows, "aw_o1"), ("盲目 O3", base_rows, "b_o3"),
                 ("盲目 O1", base_rows, "b_o1"), ("pytket", base_rows, "tket"),
                 ("RL+感知 O3", rl_rows, "rl_o3")]
        for name, rs, prefix in specs:
            if not rs:
                continue
            st = {k: stat(rs, prefix, k) for k in ("swaps", "depth", "twoq_depth", "fidelity")}
            # RL+O3 的字段名是 twoq 而非 twoq_depth
            if prefix == "rl_o3" and st["twoq_depth"] is None:
                st["twoq_depth"] = stat(rs, "rl_o3", "twoq")
            entry["stats"][name] = st
            md.append(f"| {name} | {fmt(st['swaps'])} | {fmt(st['depth'])} | "
                      f"{fmt(st['twoq_depth'])} | {fmt(st['fidelity'])} |")
        md.append("")

        # 比值/胜率（保真度越高越好；深度/SWAP 越低越好）
        def pairs(prefix, key, invert=False):
            out = []
            for r in ok:
                b = base.get(r["circuit"])
                if not b or b.get(f"{prefix}_{key}") is None:
                    continue
                x, y = r[f"qt_{key}"], b[f"{prefix}_{key}"]
                if invert:
                    x, y = y, x  # 越低越好 → 反转成"越大越好"再算胜率
                out.append((x, y))
            return out

        def pairs_rl(key):
            out = []
            for r in ok:
                b = rl.get(r["circuit"])
                if not b or b.get(f"rl_o3_{key}") is None:
                    continue
                out.append((r[f"qt_{key}"], b[f"rl_o3_{key}"]))
            return out

        md.append("| 对比 | 保真度中位比值（胜率） | 深度中位比值（胜率，越低越好） |")
        md.append("|---|---|---|")
        for name, pr, pd in (
            ("vs 噪声感知 O3", pairs("aw_o3", "fidelity"), pairs("aw_o3", "depth", invert=True)),
            ("vs 盲目 O3", pairs("b_o3", "fidelity"), pairs("b_o3", "depth", invert=True)),
            ("vs pytket", pairs("tket", "fidelity"), pairs("tket", "depth", invert=True)),
            ("vs RL+感知 O3", pairs_rl("fidelity"), pairs_rl("depth")),
        ):
            rf, rd = ratio_stats(pr), ratio_stats(pd)
            entry["ratios"][name] = {"fidelity": rf, "depth": rd}
            md.append(f"| {name} | {rf['median_ratio']:.3f}（{100*rf['win_rate']:.0f}%, n={rf['n']}） "
                      f"| {rd['median_ratio']:.3f}（{100*rd['win_rate']:.0f}%, n={rd['n']}） |"
                      if rf and rd else f"| {name} | - | - |")
        md.append("")
        summary["buckets"][bucket] = entry
        for r in ok:
            all_rows.append({"bucket": bucket, **r})

    # 总体（每方法在其与 v8 共有线路上加权；RL+O3 仅覆盖 ≤50 比特，单独标注）
    md.append("\n## 总体（各方法在 v8 完成线路上按层内线路数加权）\n")
    md.append("| 方法 | 覆盖层 | 估计保真度均值 | 2Q 深度均值 | 深度均值 |")
    md.append("|---|---|---|---|---|")
    totals = {}
    for name in ("本方案 v8", "噪声感知 O3", "盲目 O3", "pytket", "RL+感知 O3"):
        fid, twoq, dep, cnt = 0.0, 0.0, 0.0, 0
        cover = []
        for bucket in BUCKETS:
            e = summary["buckets"][bucket]["stats"].get(name)
            if not e or not e["fidelity"]:
                continue
            n = e["fidelity"]["n"]
            fid += e["fidelity"]["mean"] * n
            twoq += (e["twoq_depth"]["mean"] if e["twoq_depth"] else 0) * n
            dep += e["depth"]["mean"] * n
            cnt += n
            cover.append(CN[bucket].split(" ")[0])
        totals[name] = {"n": cnt, "fidelity_mean": fid / cnt if cnt else None,
                        "twoq_mean": twoq / cnt if cnt else None,
                        "depth_mean": dep / cnt if cnt else None,
                        "coverage": "+".join(cover)}
        md.append(f"| {name} | {totals[name]['coverage']} | {totals[name]['fidelity_mean']:.4f} | "
                  f"{totals[name]['twoq_mean']:.1f} | {totals[name]['depth_mean']:.1f} |")
    summary["totals"] = totals
    # 关键比值（均值口径）
    md.append("\n关键比值（均值口径）：")
    for b in ("small", "medium"):
        e = summary["buckets"][b]["stats"]
        line = (f"- {CN[b]}：保真度 本方案 / 噪声感知O3 = "
                f"{e['本方案 v8']['fidelity']['mean']/e['噪声感知 O3']['fidelity']['mean']:.3f}")
        if e.get("pytket") and e["pytket"]["fidelity"]["mean"]:
            line += f"；/ pytket = {e['本方案 v8']['fidelity']['mean']/e['pytket']['fidelity']['mean']:.3f}"
        md.append(line)
    e = summary["buckets"]["large"]["stats"]
    md.append(f"- {CN['large']}（乘积代理下溢，改看 2Q 深度）：本方案 / 噪声感知O3 = "
              f"{e['本方案 v8']['twoq_depth']['mean']/e['噪声感知 O3']['twoq_depth']['mean']:.3f}；"
              f"/ pytket = {e['本方案 v8']['twoq_depth']['mean']/e['pytket']['twoq_depth']['mean']:.3f}；"
              f"深度 / pytket = {e['本方案 v8']['depth']['mean']/e['pytket']['depth']['mean']:.3f}。")
    # 同覆盖面对比：同一批 small+medium 线路上 v8 vs RL+O3
    common = []
    for r in all_rows:
        if r["bucket"] not in ("small", "medium"):
            continue
        b = rl.get(r["circuit"])
        if b and b.get("rl_o3_fidelity") is not None and r.get("qt_fidelity") is not None:
            common.append((r["qt_fidelity"], b["rl_o3_fidelity"]))
    if common:
        v8_mean = statistics.mean(x for x, _ in common)
        rl_mean = statistics.mean(y for _, y in common)
        md.append(f"- 同覆盖（≤50 比特，n={len(common)}）：本方案 {v8_mean:.4f} / "
                  f"RL+感知O3 {rl_mean:.4f} = {v8_mean/rl_mean:.3f}。")
        summary["same_coverage_rl_o3"] = {
            "n": len(common), "v8": v8_mean, "rl_o3": rl_mean,
            "ratio": v8_mean / rl_mean}

    # 发布前审计
    audit = Path("tables/release_audit/summary.md")
    if audit.exists():
        md.append("\n## 发布前审计（真机可行性 + 语义）\n")
        md.append(audit.read_text(encoding="utf-8"))
        summary["audit"] = audit.read_text(encoding="utf-8")

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (OUT / "per_circuit.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows), encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
