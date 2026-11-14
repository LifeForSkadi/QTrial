# -*- coding: utf-8 -*-
"""优化前 SWAP 门数度量：平凡布局（恒等布局，零布局优化）+ SABRE 路由。

图12 的排序键 = 线路固有路由难度（与任何方法的布局质量无关）。
逐条落盘 + 断点续跑（已含 trivial_swaps 的线路跳过）。
用法：python scripts/bench_trivial_swaps.py
输出：tables/compare_models/layout_trivial_scaling_rows.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.devices import build_tianyan287_spec
from qtrail.pipeline.baselines import trivial_route
from qtrail.pipeline.routing import coupling_map_from_spec
from qtrail.utils.qasm_io import load_qasm2, strip_measurements


def main():
    spec = build_tianyan287_spec()
    cm = coupling_map_from_spec(spec)
    qdir = Path("data/scaling/qasm")
    files = sorted(qdir.glob("*.qasm"))
    out = Path("tables/compare_models/layout_trivial_scaling_rows.jsonl")
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
        if name in done and "trivial_swaps" in done[name]:
            rows.append(done[name])
            print(f"  [skip] {p.stem}", flush=True)
            write_rows(rows)
            continue
        qc, _ = strip_measurements(load_qasm2(p))
        row = {"circuit": name, "n_qubits": qc.num_qubits}
        try:
            t1 = time.time()
            sw, _ = trivial_route(qc, cm, seed=42)
            row["trivial_swaps"] = sw
            row["wall_s"] = round(time.time() - t1, 3)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:60]}"
        rows.append(row)
        write_rows(rows)
        print(f"  {p.stem:24s} trivial_swaps={row.get('trivial_swaps')} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"saved -> {out} ({len(rows)} 行)")


if __name__ == "__main__":
    main()
