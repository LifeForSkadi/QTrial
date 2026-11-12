# -*- coding: utf-8 -*-
"""真机提交（单条 QCIS）：提交并等待结果，落盘原始返回。

用法：python scripts/real_machine_v8_submit.py x5 --shots 2000 --wait 1800
判据：QCIS 规则检查通过 → 提交成功 → 结果返回；分布判定在后续汇总脚本。
"""
from __future__ import annotations

import argparse
import json
import sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--machine", default="tianyan-287")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--wait", type=int, default=1800)
    ap.add_argument("--dir", default="out/real_machine")
    args = ap.parse_args()

    key = load_key()
    if not key:
        raise SystemExit("no key (.tianyan_key / TIANYAN_LOGIN_KEY)")
    from cqlib import QuantumLanguage, TianYanPlatform
    plat = TianYanPlatform(login_key=key, machine_name=args.machine)
    qcis_path = Path(args.dir) / f"{args.name}.qcis"
    qcis = qcis_path.read_text(encoding="utf-8")
    print(f"[check] {args.name} qcis {len(qcis.splitlines())} lines", flush=True)
    try:
        ok = plat.qcis_check_regular(qcis)
        print(f"[check] qcis_check_regular -> {ok}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[check] qcis_check_regular raised {type(e).__name__}: {e}",
              flush=True)
    lab_id = plat.create_lab(f"qtrail-{args.name}",
                             "QTrial v8 real-machine validation")
    print(f"[lab] {lab_id}", flush=True)
    # 关键：默认 is_verify=True 只校验不执行；真机运行必须 False
    qids = plat.submit_job(circuit=qcis, exp_name=args.name,
                           num_shots=args.shots,
                           language=QuantumLanguage.QCIS,
                           lab_id=lab_id, is_verify=False)
    print(f"[submit] query_ids={qids}", flush=True)
    data = plat.query_experiment(qids, max_wait_time=args.wait,
                                 sleep_time=10)
    out = Path("tables/real_machine_v8")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}_raw.json").write_text(
        json.dumps(data, ensure_ascii=False, default=str, indent=1),
        encoding="utf-8")
    print(f"[result] type={type(data)} n={len(data) if hasattr(data, '__len__') else '?'}", flush=True)
    if isinstance(data, list) and data:
        item = data[0]
        print(f"[result] keys={sorted(item.keys())}", flush=True)
        rows = item.get("resultStatus")
        if rows:
            labels = rows[0]
            from collections import Counter
            counts = Counter(tuple(r) for r in rows[1:])
            total = sum(counts.values())
            print(f"[result] labels={labels} shots={total} "
                  f"top={counts.most_common(4)}", flush=True)
            (out / f"{args.name}_counts.json").write_text(
                json.dumps({"labels": labels, "total": total,
                            "counts": {"".join(map(str, k)): v
                                       for k, v in counts.items()}},
                           ensure_ascii=False, indent=1),
                encoding="utf-8")
    print(f"[saved] {out / f'{args.name}_raw.json'}", flush=True)


if __name__ == "__main__":
    main()
