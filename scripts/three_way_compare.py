# -*- coding: utf-8 -*-
"""三方法分层对比（small+medium，2026-11-12）：
修复后 v7 / 感知O3（旧表）/ RL布局+感知O3（新跑）。

按线路对齐后输出：保真度/SWAP/深度 的均值/截尾/中位 + 逐条胜场。
"""
import json
import statistics
from pathlib import Path

V7 = Path("tables/final_report_v7")
OLD = Path("tables/final_report")
RLO3 = Path("tables/compare_models/rl_o3_587_rows.jsonl")


def load_rows(path):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["circuit"]] = r
    return out


def trimmed(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * 0.1))
    return (statistics.mean(vals[k:len(vals) - k])
            if 2 * k < len(vals) else statistics.mean(vals))


def stat(vals):
    if not vals:
        return None
    return (statistics.mean(vals), trimmed(vals), statistics.median(vals))


def fmt(t):
    return f"{t[0]:.4f}/{t[1]:.4f}/{t[2]:.4f}" if t else "-"


def load_rows_merged(path_base, bucket):
    """主文件 + 分片文件合并（分片并行时使用）。"""
    out = {}
    main = Path(f"{path_base}/{bucket}_fidelity_rows.jsonl")
    files = [main] + sorted(Path(str(path_base)).glob(
        f"{bucket}_fidelity_rows_shard*.jsonl"))
    for p in files:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["circuit"], r)
    return out


def main():
    v7 = {b: load_rows_merged(V7, b) for b in ("small", "medium")}
    old = {b: load_rows(OLD / f"{b}_fidelity_rows.jsonl")
           for b in ("small", "medium")}
    rlo3_all = load_rows(RLO3)
    rlo3 = {b: {n: r for n, r in rlo3_all.items() if r.get("bucket") == b}
            for b in ("small", "medium")}

    titles = {"small": "≤10 比特", "medium": "11-50 比特"}
    for b in ("small", "medium"):
        names = [n for n in v7[b] if "qt_fidelity" in v7[b][n]]
        print(f"\n## {titles[b]}（对齐 {len(names)} 条）")
        print("| 指标（均/截/中） | v7 | 感知O3 | RL+感知O3 |")
        print("|---|---|---|---|")
        for label, key, aw_key in (("SWAP", "qt_swaps", "aw_o3_swaps"),
                                   ("深度", "qt_depth", "aw_o3_depth"),
                                   ("2Q 深度", "qt_twoq_depth",
                                    "aw_o3_twoq_depth"),
                                   ("保真度", "qt_fidelity",
                                    "aw_o3_fidelity")):
            v7v = [v7[b][n][key] for n in names if key in v7[b][n]]
            awv = [old[b][n][aw_key] for n in names
                   if n in old[b] and aw_key in old[b][n]]
            rlv = [rlo3[b][n]["rl_o3_" + key.replace("qt_", "")]
                   for n in names if n in rlo3[b]
                   and "rl_o3_" + key.replace("qt_", "") in rlo3[b][n]]
            print(f"| {label} | {fmt(stat(v7v))} | {fmt(stat(awv))} | "
                  f"{fmt(stat(rlv))} |")
        # 胜场（保真度口径）
        names_rl = [n for n in names if n in rlo3[b]
                    and "rl_o3_fidelity" in rlo3[b][n]]
        w7_aw = sum(1 for n in names if n in old[b]
                    and "aw_o3_fidelity" in old[b][n]
                    and v7[b][n]["qt_fidelity"] > old[b][n]["aw_o3_fidelity"])
        w7_rl = sum(1 for n in names_rl
                    if v7[b][n]["qt_fidelity"] > rlo3[b][n]["rl_o3_fidelity"])
        wrl_aw = sum(1 for n in names_rl if n in old[b]
                     and "aw_o3_fidelity" in old[b][n]
                     and rlo3[b][n]["rl_o3_fidelity"]
                     > old[b][n]["aw_o3_fidelity"])
        print(f"\nv7 胜 感知O3：{w7_aw}/{len(names)}；"
              f"v7 胜 RL+感知O3：{w7_rl}/{len(names_rl)}；"
              f"RL+感知O3 胜 感知O3：{wrl_aw}/{len(names_rl)}")


if __name__ == "__main__":
    main()
