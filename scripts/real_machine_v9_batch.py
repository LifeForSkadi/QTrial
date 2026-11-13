# -*- coding: utf-8 -*-
"""真机对比批量提交 + 统一轮询（v8 vs O3 两臂）。

与 real_machine_v9_submit.py 的区别：先把全部作业提交入队（彼此并行），
再统一轮询——避免逐条串行等待把总时长变成各作业排队时间之和。

用法：
  python scripts/real_machine_v9_batch.py x5 ghz5 ghz10 ghz15 qft5 qft6 \
      --arms v8 --shots 2000 --poll 1200
  python scripts/real_machine_v9_batch.py qaoa12 --arms v8,o3
状态文件：tables/real_machine_v9/jobs.json（qid 落盘，可断点续跑）
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

OUT = Path("tables/real_machine_v9")
SRC = Path("out/real_machine_v9")


def load_key() -> str:
    p = ROOT / ".tianyan_key"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    import os
    return os.environ.get("TIANYAN_LOGIN_KEY", "") or os.environ.get(
        "CQLIB_LOGIN_KEY", "")


def save_counts(name: str, arm: str, rows):
    labels = rows[0]
    counts = Counter(tuple(int(x) for x in r) for r in rows[1:])
    total = sum(counts.values())
    (OUT / f"{name}_{arm}_counts.json").write_text(json.dumps(
        {"labels": labels, "total": total,
         "counts": {"".join(map(str, k)): v for k, v in counts.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[result] {name}_{arm} shots={total} "
          f"top={counts.most_common(3)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--machine", default="tianyan176")
    ap.add_argument("--arms", default="v8,o3")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--poll", type=int, default=1200,
                    help="单次轮询等待上限（秒）")
    ap.add_argument("--deadline", type=int, default=7200,
                    help="总等待上限（秒）")
    args = ap.parse_args()

    key = load_key()
    if not key:
        raise SystemExit("no key (.tianyan_key / TIANYAN_LOGIN_KEY)")
    from cqlib import QuantumLanguage, TianYanPlatform
    plat = TianYanPlatform(login_key=key, machine_name=args.machine)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs_path = OUT / "jobs.json"
    jobs = json.loads(jobs_path.read_text(encoding="utf-8")) \
        if jobs_path.exists() else {}

    def flush():
        jobs_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    # ---- 1) 提交（已提交的跳过）
    for name in args.names:
        for arm in [a for a in args.arms.split(",") if a]:
            key_ = f"{name}_{arm}"
            if key_ in jobs:
                continue
            qcis_path = SRC / args.machine / f"{key_}.qcis"
            if not qcis_path.exists():
                print(f"[skip] missing {qcis_path}", flush=True)
                continue
            qcis = qcis_path.read_text(encoding="utf-8")
            try:
                ok = plat.qcis_check_regular(qcis)
                print(f"[check] {key_} -> {ok}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[check] {key_} raised {type(e).__name__}: {e}",
                      flush=True)
            lab_id = plat.create_lab(f"qtrail-v9-{key_}",
                                     "QTrial v9 real-machine comparison")
            qids = plat.submit_job(circuit=qcis, exp_name=key_,
                                   num_shots=args.shots,
                                   language=QuantumLanguage.QCIS,
                                   lab_id=lab_id, is_verify=False)
            jobs[key_] = {"qids": qids, "shots": args.shots}
            flush()
            print(f"[submit] {key_} qids={qids}", flush=True)

    # ---- 2) 统一轮询
    t0 = time.time()
    pending = [k for k in jobs
               if not (OUT / f"{k}_counts.json").exists()]
    print(f"[poll] pending {len(pending)}: {pending}", flush=True)
    while pending and time.time() - t0 < args.deadline:
        for key_ in list(pending):
            qids = jobs[key_]["qids"]
            try:
                data = plat.query_experiment(qids, max_wait_time=args.poll,
                                             sleep_time=10)
            except Exception as e:  # noqa: BLE001
                print(f"[poll] {key_} not ready: {type(e).__name__}",
                      flush=True)
                continue
            item = data[0] if isinstance(data, list) and data else data
            rows = item.get("resultStatus") if isinstance(item, dict) else None
            if not rows:
                print(f"[poll] {key_} no resultStatus yet", flush=True)
                continue
            name, arm = key_.rsplit("_", 1)
            (OUT / f"{key_}_raw.json").write_text(
                json.dumps(data, ensure_ascii=False, default=str, indent=1),
                encoding="utf-8")
            save_counts(name, arm, rows)
            pending.remove(key_)
        if pending:
            print(f"[poll] remaining {pending}", flush=True)
    print("DONE" if not pending else f"TIMEOUT pending={pending}", flush=True)


if __name__ == "__main__":
    main()
