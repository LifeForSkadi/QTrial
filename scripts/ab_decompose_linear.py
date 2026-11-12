# -*- coding: utf-8 -*-
"""decompose_linear（线性分解）A/B 机制裁决（2026-11-12 补充四十预注册）。

阶段 1（dump）：每目标线路跑生产管线一次（linear_decompose=False），
  QTRAIL_DUMP_DECOMPOSE_INPUT 抓 decompose 输入（_v5_pipeline 在候选
  评分循环中每候选各写一次、胜者补跑最后写——末次即胜者输入）；
  同时记录生产基线指标（res["metrics"]）。
阶段 2（A/B）：同一 dump 输入上 A=生产 fixpoint（resynth_every=1）
  vs B=线性（linear=True）——1Q/cz/depth/logf/合法性/墙钟。

预注册判据（qv32）：B 1Q ≤16,500、depth ≤3600、logf 不劣化、
零非法；heis_N16/qft_N020 回归不劣化。

用法：
  python scripts/ab_decompose_linear.py --targets qv32,heis16,qft20
  python scripts/ab_decompose_linear.py --targets qv32 --skip-dump
"""
import argparse
import math
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.circuit import Circuit
from qtrail.pure.mapper import PureMapper
from qtrail.pure.metrics import compute_metrics
from qtrail.pure.post_v2 import _decompose_ablation
from qtrail.pure.qasm import parse_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
DUMP_DIR = Path("scripts/tmp_decompose_ab")
TARGETS = {
    "qv32": "data/benchpress/qasm/qasmbench-large/QV_n32/32.qasm",
    "heis16": "data/benchpress/qasm/square-heisenberg/square_heisenberg_N16.qasm",
    "qft20": "data/benchpress/qasm/qft/qft_N020.qasm",
    "hwb12": "data/benchpress/qasm/feynman/hwb12.qasm",
}

_PLATFORM_1Q = {"rz", "sx", "x"}


def log_fidelity(circ, calib):
    err2q = calib.err_2q
    med = float(sum(float(v) for v in err2q.values())
                / max(len(err2q), 1))
    s = 0.0
    for inst in circ.ops:
        if inst.nq == 2:
            e = err2q.get((inst.qubits[0], inst.qubits[1]),
                          err2q.get((inst.qubits[1], inst.qubits[0]), med))
            s += math.log(max(1.0 - float(e), 1e-12))
        else:
            s += math.log(max(1.0 - float(calib.err_1q[inst.qubits[0]]),
                              1e-12))
    for inst in circ.measures:
        s += math.log(max(1.0 - float(calib.err_ro[inst.qubits[0]]), 1e-12))
    return s


def legality_scan(circ, spec):
    """真机合法性三重扫描（门集/cz 相邻/禁用比特/swap 残留/比特界）。"""
    bad = []
    disabled = set(spec.disabled_qubits)
    for inst in circ.ops:
        if inst.name not in _PLATFORM_1Q | {"cz"}:
            bad.append(f"门集:{inst.name}")
        for q in inst.qubits:
            if q >= spec.n or q in disabled:
                bad.append(f"比特:{inst.name}{inst.qubits}")
        if inst.name == "cz":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            if not spec.adj[a, b]:
                bad.append(f"非相邻:{a},{b}")
    return bad


def ab_one(dump_path, spec, label):
    with open(dump_path, "rb") as f:
        circ = pickle.load(f)
    print(f"\n== A/B: {label}（dump 输入 门={len(circ.ops)} "
          f"cx={circ.count('cx')} u3={circ.count('u3')} "
          f"u1={circ.count('u1')} swap={circ.count('swap')}）", flush=True)
    results = {}
    for name, kwargs in (("A_fixpoint",
                          {"spec": spec, "resynth_every": 1}),
                         ("B_linear",
                          {"spec": spec, "linear": True})):
        t0 = time.time()
        out = _decompose_ablation(circ, **kwargs)
        wall = time.time() - t0
        m = compute_metrics(out, 0, spec.calib)
        n1q = sum(1 for i in out.ops if i.name in _PLATFORM_1Q)
        nrz = sum(1 for i in out.ops if i.name == "rz")
        ncz = out.count("cz")
        lf = log_fidelity(out, spec.calib)
        bad = legality_scan(out, spec)
        results[name] = (n1q, nrz, ncz, m["depth"], lf, wall, bad)
        print(f"[{name}] 1Q={n1q}（rz={nrz}） cz={ncz} depth={m['depth']} "
              f"logf={lf:.2f} wall={wall:.1f}s 非法={len(bad)}"
              + (f" {bad[:3]}" if bad else ""), flush=True)
    return results


def run_pipeline(path, spec, policy, cfg, dev, name):
    """生产管线跑一次 + dump decompose 输入（末次写 = 胜者输入）。"""
    circ = parse_qasm(Path(path).read_text(encoding="utf-8",
                                          errors="replace"))
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    DUMP_DIR.mkdir(exist_ok=True)
    dump_path = DUMP_DIR / f"dump_{name}.pkl"
    import os
    os.environ["QTRAIL_DUMP_DECOMPOSE_INPUT"] = str(dump_path)
    t0 = time.time()
    res = mapper.map_circuit(circ, circuit_id=name)
    wall = time.time() - t0
    del os.environ["QTRAIL_DUMP_DECOMPOSE_INPUT"]
    m = res["metrics"]
    rc = res["routed_circuit"]
    lf = log_fidelity(rc, spec.calib)
    print(f"[生产] {name}: fid={m['est_fidelity']:.3e} logf={lf:.2f} "
          f"twoq={m['twoq_count']} depth={m['depth']} wall={wall:.0f}s "
          f"→ dump {dump_path}", flush=True)
    return dump_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="qv32,heis16,qft20")
    ap.add_argument("--skip-dump", action="store_true",
                    help="跳过生产管线阶段，直接用已有 dump")
    args = ap.parse_args()

    spec = build_tianyan287_spec(load_device_config())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    all_res = {}
    for name in args.targets.split(","):
        if name not in TARGETS:
            print(f"未知目标 {name}（可选 {list(TARGETS)}）")
            continue
        dump_path = DUMP_DIR / f"dump_{name}.pkl"
        if not args.skip_dump:
            dump_path = run_pipeline(TARGETS[name], spec, policy, cfg,
                                     dev, name)
        elif not dump_path.exists():
            print(f"[跳过] {name}：无 dump（{dump_path} 不存在）")
            continue
        all_res[name] = ab_one(dump_path, spec, name)

    # 判据裁决
    print("\n===== 预注册判据裁决 =====", flush=True)
    ok_all = True
    for name, res in all_res.items():
        a1q, _, acz, adep, alf, _, abad = res["A_fixpoint"]
        b1q, _, bcz, bdep, blf, _, bbad = res["B_linear"]
        ok = (not bbad and blf >= alf - 1e-9
              and b1q <= a1q + 0.5 and bcz <= acz + 0.5
              and bdep <= adep + 0.5)
        # qv32 绝对判据
        if name == "qv32":
            ok = ok and b1q <= 16500 and bdep <= 3600
        print(f"[{name}] B 1Q {a1q}→{b1q}（qv32 目标 ≤16500） "
              f"depth {adep}→{bdep}（目标 ≤3600） logf {alf:.1f}→{blf:.1f} "
              f"非法 {len(bbad)} → {'达标' if ok else '不达标'}")
        ok_all = ok_all and ok
    print(f"\n总裁决：{'全部达标' if ok_all else '不达标'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
