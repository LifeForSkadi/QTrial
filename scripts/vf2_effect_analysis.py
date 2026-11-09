# -*- coding: utf-8 -*-
"""VF2PostLayout 净作用分析：A(原生含VF2) vs B(同布局提取注入无VF2)。

A 与 B 的初始布局相同（B 从 A 提取）、管线相同（除两处 VF2PostLayout），
故 A-vs-B 的差异 = VF2PostLayout 的净贡献（三指标逐条统计）。
数据：tables/compare_models/inj_extract_rows.jsonl（B 组）+ layout_inject_*
（A 的 native_* 字段，经 inj_extract 行携带）。
"""
import json
from pathlib import Path

# A（原生）的完整指标从 layout_inject 源行联表
src_rows = {}
for fn in ("layout_inject_benchpress_rows.jsonl",
           "layout_inject_mqtbench_n15_rows.jsonl",
           "layout_inject_queko_BIGD_rows.jsonl"):
    p = Path("tables/compare_models") / fn
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            src_rows[r["circuit"]] = r

rows = [json.loads(l) for l in
        Path("tables/compare_models/inj_extract_rows.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
ok = []
for r in rows:
    if "b_fidelity" not in r:
        continue
    src = src_rows.get(r["circuit"], {})
    r2 = dict(r)
    for k in ("native_twoq", "native_depth", "native_swaps",
              "inj_twoq", "inj_depth", "inj_swaps"):
        r2[k] = src.get(k)
    ok.append(r2)
print(f"有效行 {len(ok)}")
for group in ("benchpress", "queko"):
    sub = [r for r in ok if r["circuit"].startswith(group)]
    if not sub:
        continue
    n = len(sub)
    print(f"\n== {group} ({n} 条)：A(原生含VF2) vs B(同布局无VF2) = VF2 净作用")
    for label, key, better in (("保真度", "fidelity", "gt"),
                               ("2Q门数", "twoq", "lt"),
                               ("深度", "depth", "lt")):
        a_win = 0
        b_win = 0
        tie = 0
        for r in sub:
            av = r.get("native_" + key)
            bv = r.get("b_" + key)
            if av is None or bv is None:
                continue
            if abs(av - bv) < 1e-12:
                tie += 1
            elif (av > bv) if better == "gt" else (av < bv):
                a_win += 1
            else:
                b_win += 1
        print(f"  {label}: VF2有利 {a_win}/{n}  平 {tie}  VF2有害 {b_win}/{n}")
    # 逐条幅度（保真度比值 + 门数差）
    f_ratios = []
    for r in sub:
        if r.get("native_fidelity") and r.get("b_fidelity"):
            f_ratios.append(r["native_fidelity"] / max(r["b_fidelity"], 1e-300))
    if f_ratios:
        import statistics
        print(f"  保真度比值 A/B：中位 {statistics.median(f_ratios):.3f} "
              f"（>1 = VF2 有利）")
