# -*- coding: utf-8 -*-
"""真机 v9 批量短轮询：每 30s 查一次全部 qid，先完成的先落盘。

与 batch.py 的阻塞式 query_experiment 不同：直接调平台查询接口，
返回列表里有多少条就落多少条，不要求全部同时完成。

用法：python scripts/real_machine_v9_poll.py --deadline 7200
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


def load_key() -> str:
    p = ROOT / ".tianyan_key"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    import os
    return os.environ.get("TIANYAN_LOGIN_KEY", "") or os.environ.get(
        "CQLIB_LOGIN_KEY", "")


def save(key_, item):
    name, arm = key_.rsplit("_", 1)
    (OUT / f"{key_}_raw.json").write_text(
        json.dumps(item, ensure_ascii=False, default=str, indent=1),
        encoding="utf-8")
    rows = item.get("resultStatus")
    if not rows:
        return False
    labels = rows[0]
    counts = Counter(tuple(int(x) for x in r) for r in rows[1:])
    total = sum(counts.values())
    (OUT / f"{key_}_counts.json").write_text(json.dumps(
        {"labels": labels, "total": total,
         "counts": {"".join(map(str, k)): v for k, v in counts.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[result] {key_} shots={total} top={counts.most_common(3)}",
          flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--deadline", type=int, default=7200)
    ap.add_argument("--machine", default="tianyan176")
    args = ap.parse_args()

    key = load_key()
    if not key:
        raise SystemExit("no key")
    from cqlib import TianYanPlatform
    plat = TianYanPlatform(login_key=key, machine_name=args.machine)
    jobs = json.loads((OUT / "jobs.json").read_text(encoding="utf-8"))
    todo = {k: v["qids"] for k, v in jobs.items()
            if not (OUT / f"{k}_counts.json").exists()}
    print(f"[poll] todo {len(todo)}: {list(todo)}", flush=True)
    t0 = time.time()
    while todo and time.time() - t0 < args.deadline:
        qids = [q for v in todo.values() for q in v]
        try:
            r = plat._send_request(path=plat.QUERY_EXP_PATH,
                                   data={"query_ids": qids}, method="POST")
            lst = (r.get("data", {}) or {}).get(
                "experimentResultModelList", []) or []
        except Exception as e:  # noqa: BLE001
            print(f"[poll] query error {type(e).__name__}", flush=True)
            lst = []
        got = set()
        for item in lst:
            qid = str(item.get("queryId") or item.get("query_id") or "")
            key_ = next((k for k, v in todo.items() if qid in v), None)
            if key_ is None:
                # 找不到 qid 字段时按返回顺序兜底（单 job 查询才有意义）
                continue
            if save(key_, item):
                got.add(key_)
        for k in got:
            todo.pop(k, None)
        if todo:
            print(f"[poll] {time.strftime('%H:%M:%S')} remaining "
                  f"{list(todo)}", flush=True)
            time.sleep(args.interval)
    print("DONE" if not todo else f"TIMEOUT {list(todo)}", flush=True)


if __name__ == "__main__":
    main()
