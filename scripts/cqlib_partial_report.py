"""从 bench 日志生成部分报告（任务中途卡死时抢救已完成的线路结果）。

用法: python scripts/cqlib_partial_report.py [log_path]
输出: tables/cqlib_inject/partial_report.md + {bucket}_rows_partial.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.cqlib_inject_bench import render  # noqa: E402

LINE_RE = re.compile(r"^\s*(\w+_\d+)\s+n=\s*(\d+)\s*\|\s*(.*)$")
COMPILERS = {
    "sabre": ("sabre", "swaps", "depth"),
    "tket": ("tket", "swaps", "depth"),
    "csize": ("cqlib_size", "swaps", "depth"),
    "cdepth": ("cqlib_depth", "swaps", "depth"),
    "O+C": ("qtrial_cqlib", "swaps", "depth"),
    "O+S": ("qtrial_sabre", "swaps", "depth"),
}


def parse_log(log_path: Path) -> dict:
    rows = {"small": [], "medium": [], "large": []}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        name, n, rest = m.group(1), int(m.group(2)), m.group(3)
        row = {"circuit": name, "n": n}
        for tok in rest.split("|"):
            tok = tok.strip()
            if not tok:
                continue
            key, _, val = tok.partition(" ")
            if key not in COMPILERS:
                continue
            prefix, _, _ = COMPILERS[key]
            if val.startswith("-/-"):
                row[f"{prefix}_error"] = val[3:].strip() or "unknown"
                continue
            parts = val.split("/")
            if len(parts) == 2 and all(p.strip().lstrip("-").isdigit()
                                       for p in parts):
                row[f"{prefix}_swaps"] = int(parts[0])
                row[f"{prefix}_depth"] = int(parts[1])
        if n <= 10:
            rows["small"].append(row)
        elif n <= 50:
            rows["medium"].append(row)
        else:
            rows["large"].append(row)
    return rows


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "logs/cqlib_inject_bench4.log")
    rows = parse_log(log_path)
    out_dir = Path("tables/cqlib_inject")
    # 优先用已落盘的完整行（含 twoq_depth/保真度/耗时字段）
    for b in rows:
        jfile = out_dir / f"{b}_rows.jsonl"
        if jfile.exists():
            full = {}
            for line in jfile.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    full[r["circuit"]] = r
            if full:
                rows[b] = [full.get(r["circuit"], r) for r in rows[b]]
    titles = {"small": "≤10 比特", "medium": "11-50 比特",
              "large": "51-105 比特（含满占）"}
    sections = ["# Cqlib 注入接口分层对比——部分报告（抢救版）\n",
                f"来源日志 {log_path}；仅含已完成线路（"
                f"{sum(len(v) for v in rows.values())} 条）；"
                "统一口径 decompose_to_platform + compute_metrics；"
                "统计：均值/截尾(去10%)/中位数\n"]
    for b in ("small", "medium", "large"):
        if not rows[b]:
            continue
        with open(out_dir / f"{b}_rows_partial.jsonl", "w",
                  encoding="utf-8") as f:
            for r in rows[b]:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
        render(sections, titles[b], rows[b])
    out = out_dir / "partial_report.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    print(f"partial report -> {out} "
          f"({sum(len(v) for v in rows.values())} circuits)")


if __name__ == "__main__":
    main()
