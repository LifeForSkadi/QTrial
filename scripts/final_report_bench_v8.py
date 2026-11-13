# -*- coding: utf-8 -*-
"""最终分层报告 v8（2026-11-13）：v7 同口径，管线升级为
A2 自研 VW 合成（默认）+ 自研 VF2++ 精确嵌入 + 候选类型化/Inst 冻结
（第 3–6 项）。输出 tables/final_report_v8/。

原 v7 说明：QTrial v5 管线（A1+A2+A3+B+C1+L3+非劣化开关）+
新路由器（评分对齐 SABRE/LightSABRE：恒权、w_ext=0.5/n、无禁忌、
种子随机平局、10n 逃逸、拓扑序执行）全量重测。

与 v3/v6 同线路集（Benchpress ≤105 + MQTBench stratified + QUEKO 子集），
同 c0.05 权重、seed=42、fidelity 规则。基线（感知/盲目 O1/O3、pytket）
不重跑——对比时读旧表（tables/final_report/）。v6 表 = 老路由器对照
（隔离 A1/A2 贡献）。

输出：tables/final_report_v7/（{bucket}_fidelity_rows.jsonl + summary.md）。
断点续跑。--buckets 分片；--seeds 路由多种子（默认 3）。
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import threading
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
from qtrail.utils.qasm_io import sanitize_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OUT = Path("tables/final_report_v8")
V3 = Path("tables/final_report_v3")
OLD = Path("tables/final_report")
V6 = Path("tables/final_report_v6")
CACHE = Path("data/mqtbench/stratified")
BP = Path("data/benchpress/qasm")
RULE = "fidelity"


def load_circuits(use_cache: bool = True):
    """加载分层线路；默认读写 buckets_cache.pkl。

    2026-11-13：benchpress 里含数十万门的大线路，656 个文件全量解析
    单进程需数分钟；并行分片时每个进程都重解析一遍是纯浪费。首次
    解析后 pickle 缓存（文本 + 分层），后续进程秒级加载。
    """
    import pickle
    cache = OUT / "buckets_cache.pkl"
    if use_cache and cache.exists():
        try:
            return pickle.loads(cache.read_bytes())
        except Exception:  # noqa: BLE001 缓存损坏则重算
            pass
    from qiskit import qasm2  # noqa: F401（与 v3 完全同口径）
    buckets = {"small": [], "medium": [], "large": []}
    skipped = 0
    for p in BP.rglob("*.qasm"):
        text = p.read_text(encoding="utf-8", errors="replace")
        try:
            circ = parse_qasm(text)
            if circ.n == 0:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        if circ.n > 105:
            skipped += 1
            continue
        b = "small" if circ.n <= 10 else ("medium" if circ.n <= 50 else "large")
        buckets[b].append((f"bp_{p.parent.name}_{p.stem}", text))
    for p in sorted(CACHE.glob("*.qasm")):
        name = p.stem
        size = int(name.rsplit("_", 1)[1])
        if size > 105:
            continue
        b = "small" if size <= 10 else ("medium" if size <= 50 else "large")
        buckets[b].append((name, sanitize_qasm(p.read_text(encoding="utf-8"))))
    for p in iter_queko_files("BIGD"):
        if re.search(r"\.(3|4)D1_\.(4|5)D2_", p.name) \
                or re.search(r"\.0D1_\.1D2_", p.name):
            buckets["medium"].append(
                (p.stem, sanitize_qasm(p.read_text(encoding="utf-8"))))
    if use_cache:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps((buckets, skipped)))
        except Exception:  # noqa: BLE001
            pass
    return buckets, skipped


def run_with_timeout(fn, timeout):
    """在守护线程里跑 fn，超时返回 (None, 'timeout')（残留线程随进程退出）。
    timeout=None 时不限时。"""
    if timeout is None:
        try:
            return fn(), None
        except Exception as e:
            raise
    box = {}

    def target():
        try:
            box["res"] = fn()
        except Exception as e:  # 传给主线程抛
            box["err"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, "timeout"
    if "err" in box:
        raise box["err"]
    return box.get("res"), None


def trimmed(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * 0.1))
    return (statistics.mean(vals[k:len(vals) - k])
            if 2 * k < len(vals) else statistics.mean(vals))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="small,medium,large",
                    help="只跑指定层（并行分片用：如 --buckets large）")
    ap.add_argument("--seeds", type=int, default=16,
                    help="最终路由多种子数（routing_seeds，2026-11-13 从 20 "
                         "降为 16：与 20 的 min 差距 <1%，墙钟省 20%；"
                         "线程并行 8 worker）")
    ap.add_argument("--shard", default=None,
                    help="层内分片 k/N（并行用：线路按序均分到 N 片，"
                         "每片独立输出文件，汇总时合并）")
    ap.add_argument("--timeout", type=float, default=None,
                    help="每线墙钟超时（秒）。超时写 timeout 错误行并跳过"
                         "（守护线程残留，进程退出时自动回收）")
    ap.add_argument("--dev", default=None,
                    help="cpu/cuda；默认自动。并行分片时建议显式 cpu"
                         "（多进程同时建 CUDA context 会空转/卡死）")
    args = ap.parse_args()
    buckets_sel = [b for b in args.buckets.split(",") if b]
    shard_id = shard_n = None
    if args.shard:
        shard_id, shard_n = (int(x) for x in args.shard.split("/"))
        assert 0 <= shard_id < shard_n
    dev = args.dev or ("cuda" if torch.cuda.device_count() > 0 else "cpu")
    print(f"dev={dev} buckets={buckets_sel}", flush=True)
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    buckets, skipped = load_circuits()
    total = sum(len(v) for v in buckets.values())
    print(f"circuits: small {len(buckets['small'])} / medium "
          f"{len(buckets['medium'])} / large {len(buckets['large'])} "
          f"(total {total}, skipped {skipped})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for bucket in buckets_sel:
        rows = []
        suffix = f"_shard{shard_id}" if shard_id is not None else ""
        out_file = OUT / f"{bucket}_{RULE}_rows{suffix}.jsonl"
        done_names = set()
        if out_file.exists():
            for line in out_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done_names.add(r["circuit"])
                    rows.append(r)
        n_done = len(done_names)
        # 分片：按序均分（k 片取第 shard_id 份连续区间）
        items = list(enumerate(buckets[bucket]))
        if shard_id is not None:
            lo = shard_id * len(items) // shard_n
            hi = (shard_id + 1) * len(items) // shard_n
            items = items[lo:hi]
        for i, (name, ptext) in items:
            if name in done_names:
                continue
            t0 = time.time()
            row = {"circuit": name, "n": parse_qasm(ptext).n}
            try:
                torch.manual_seed(42)  # 布局确定性（与 v3 同口径）
                mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev,
                                    seed=42, selection_rule=RULE, use_post=True,
                                    pipeline="v5", routing_seeds=args.seeds)
                res, terr = run_with_timeout(
                    lambda: mapper.map_circuit(parse_qasm(ptext),
                                               circuit_id=name),
                    args.timeout)
                if terr is not None:
                    row["qt_error"] = f"timeout_{args.timeout:.0f}s"
                else:
                    m = res["metrics"]
                    row["qt_swaps"] = res["swap_count"]
                    row["qt_depth"] = m["depth"]
                    row["qt_twoq_depth"] = m["twoq_depth"]
                    row["qt_twoq_count"] = m.get("twoq_count")
                    row["qt_fidelity"] = m["est_fidelity"]
                    # 独立审计 P1-2：同时落 log 口径（乘积下溢时唯一可用）
                    row["qt_logf_product"] = m.get("log_est_fidelity_product")
                    row["qt_logf_time"] = m.get("log_est_fidelity_time")
                    row["qt_est_time_us"] = m.get("est_time_us")
                    row["qt_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["qt_error"] = f"{type(e).__name__}: {str(e)[:60]}"
            rows.append(row)
            with open(out_file, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i + 1 + n_done) % 5 == 0:
                print(f"  [{bucket}] {i + 1 + n_done}/{len(buckets[bucket])} "
                      f"({name})", flush=True)
        print(f"[{bucket}] 完成 {len(buckets[bucket])} 条 -> {out_file}",
              flush=True)

    # ---- 汇总（v7 vs v6 老路由器 vs v3 修复后 vs 感知 O3 vs pytket）
    titles = {"small": "≤10 比特", "medium": "11-50 比特",
              "large": "51-105 比特"}
    v3, old, v6 = {}, {}, {}
    for bucket in ("small", "medium", "large"):
        for src, store in ((V3, v3), (OLD, old), (V6, v6)):
            p = src / f"{bucket}_{RULE}_rows.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        store[r["circuit"]] = r
    lines = ["# QTrial v7 管线全量汇总（路由器重写 + A1/A2/A3 + 联合选择）\n",
             "v7 = 路由器评分重写（恒权/0.5n 扩展集/随机平局/10n 逃逸/双步前瞻）"
             "+ 管线级联合选择 + v5 组件（A1 折叠+LNN 链/A2/A3/B/C1/L3/"
             "非劣化开关）；v6 = 老路由器对照（隔离路由器贡献）；基线不重跑，"
             "感知 O3/pytket 读旧表。\n"]
    for bucket in ("small", "medium", "large"):
        # 合并主文件与分片文件
        rows = []
        main = OUT / f"{bucket}_{RULE}_rows.jsonl"
        if main.exists():
            rows += [json.loads(l) for l in
                     main.read_text(encoding="utf-8").splitlines()
                     if l.strip()]
        for sp in sorted(OUT.glob(f"{bucket}_{RULE}_rows_shard*.jsonl")):
            rows += [json.loads(l) for l in
                     sp.read_text(encoding="utf-8").splitlines()
                     if l.strip()]
        if not rows:
            continue
        seen = set()
        uniq = []
        for r in rows:
            if r["circuit"] not in seen:
                seen.add(r["circuit"])
                uniq.append(r)
        rows = uniq
        ok = [r for r in rows if "qt_fidelity" in r]
        n_err = len(rows) - len(ok)

        def stat(key):
            vals = [r[key] for r in ok if key in r]
            if not vals:
                return None
            return (statistics.mean(vals), trimmed(vals),
                    statistics.median(vals))

        def fmt(t):
            return (f"{t[0]:.4f}/{t[1]:.4f}/{t[2]:.4f}"
                    if t and t[0] is not None else "-")

        def pick(store, circ, prefix):
            r = store.get(circ)
            if not r:
                return None
            keys = {f"{prefix}_{k}": k for k in
                    ("swaps", "depth", "twoq_depth", "fidelity")}
            out = {}
            for full, k in keys.items():
                if full in r:
                    out[k] = r[full]
            return out if out else None

        def stat3(store, prefix, key):
            vals = [pick(store, r["circuit"], prefix)[key]
                    for r in ok
                    if pick(store, r["circuit"], prefix)
                    and key in pick(store, r["circuit"], prefix)]
            if not vals:
                return None
            return (statistics.mean(vals), trimmed(vals),
                    statistics.median(vals))

        lines.append(f"\n## {titles[bucket]}（{len(ok)} 条完成，{n_err} 条错误）\n")
        lines.append("| 指标（均值/截尾/中位） | v7 | v6 老路由器 | v3 修复后 "
                     "| 感知 O3 | pytket |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for label, key in (("SWAP 数", "qt_swaps"), ("深度", "qt_depth"),
                           ("2Q 深度", "qt_twoq_depth"),
                           ("估计保真度", "qt_fidelity")):
            lines.append(
                f"| {label} | {fmt(stat(key))} | "
                f"{fmt(stat3(v6, 'qt', 'swaps' if key == 'qt_swaps' else 'depth' if key == 'qt_depth' else 'twoq_depth' if key == 'qt_twoq_depth' else 'fidelity'))} | "
                f"{fmt(stat3(v3, 'qt', 'swaps' if key == 'qt_swaps' else 'depth' if key == 'qt_depth' else 'twoq_depth' if key == 'qt_twoq_depth' else 'fidelity'))} | "
                f"{fmt(stat3(old, 'aw_o3', 'swaps' if key == 'qt_swaps' else 'depth' if key == 'qt_depth' else 'twoq_depth' if key == 'qt_twoq_depth' else 'fidelity'))} | "
                f"{fmt(stat3(old, 'tket', 'swaps' if key == 'qt_swaps' else 'depth' if key == 'qt_depth' else 'twoq_depth' if key == 'qt_twoq_depth' else 'fidelity'))} |\n")
    # 独立审计 P1-4：分片脚本不再写 summary.md——该块是 v7 表头 +
    # 分片局部数据，曾与 aggregate_v8 的 summary_v8.md 混淆。
    # 最终分层汇总统一由 scripts/aggregate_v8.py 生成。
    print("分片完成；最终汇总运行 scripts/aggregate_v8.py")


if __name__ == "__main__":
    main()
