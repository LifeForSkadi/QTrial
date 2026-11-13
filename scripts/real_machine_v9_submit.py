# -*- coding: utf-8 -*-
"""真机对比提交（v8 vs O3 两臂）：逐条提交并等待，落盘 raw/counts。

用法：
  python scripts/real_machine_v9_submit.py qft5 qft6 ghz10 --shots 2000 --wait 2400
  python scripts/real_machine_v9_submit.py qft5 --arms v8
输出：
  tables/real_machine_v9/<name>_<arm>_raw.json
  tables/real_machine_v9/<name>_<arm>_counts.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_key() -> str:
    p = ROOT / ".tianyan_key"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    import os
    return os.environ.get("TIANYAN_LOGIN_KEY", "") or os.environ.get(
        "CQLIB_LOGIN_KEY", "")


def save_counts(name: str, arm: str, rows, out: Path):
    labels = rows[0]
    counts = Counter(tuple(int(x) for x in r) for r in rows[1:])
    total = sum(counts.values())
    (out / f"{name}_{arm}_counts.json").write_text(json.dumps(
        {"labels": labels, "total": total,
         "counts": {"".join(map(str, k)): v for k, v in counts.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[result] {name}_{arm} shots={total} "
          f"top={counts.most_common(3)}", flush=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--machine", default="tianyan176")
    ap.add_argument("--arms", default="v8,o3")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--wait", type=int, default=2400)
    args = ap.parse_args()

    key = load_key()
    if not key:
        raise SystemExit("no key (.tianyan_key / TIANYAN_LOGIN_KEY)")
    from cqlib import QuantumLanguage, TianYanPlatform
    plat = TianYanPlatform(login_key=key, machine_name=args.machine)
    out = Path("tables/real_machine_v9")
    out.mkdir(parents=True, exist_ok=True)
    src = Path("out/real_machine_v9") / args.machine

    for name in args.names:
        for arm in [a for a in args.arms.split(",") if a]:
            qcis_path = src / f"{name}_{arm}.qcis"
            if not qcis_path.exists():
                print(f"[skip] missing {qcis_path}", flush=True)
                continue
            qcis = qcis_path.read_text(encoding="utf-8")
            print(f"[job] {name}_{arm} qcis {len(qcis.splitlines())} lines",
                  flush=True)
            try:
                ok = plat.qcis_check_regular(qcis)
                print(f"[check] {name}_{arm} -> {ok}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[check] {name}_{arm} raised {type(e).__name__}: {e}",
                      flush=True)
            lab_id = plat.create_lab(
                f"qtrail-v9-{name}-{arm}", "QTrial v9 real-machine comparison")
            qids = plat.submit_job(circuit=qcis, exp_name=f"{name}_{arm}",
                                   num_shots=args.shots,
                                   language=QuantumLanguage.QCIS,
                                   lab_id=lab_id, is_verify=False)
            print(f"[submit] {name}_{arm} query_ids={qids}", flush=True)
            data = plat.query_experiment(qids, max_wait_time=args.wait,
                                         sleep_time=10)
            (out / f"{name}_{arm}_raw.json").write_text(
                json.dumps(data, ensure_ascii=False, default=str, indent=1),
                encoding="utf-8")
            item = data[0] if isinstance(data, list) and data else data
            rows = item.get("resultStatus") if isinstance(item, dict) else None
            if rows:
                save_counts(name, arm, rows, out)
            else:
                print(f"[warn] {name}_{arm} no resultStatus", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
