# -*- coding: utf-8 -*-
"""循环重排模拟 v4（2026-11-13）：heisenberg N16。

根因（v2/v3 实验确认）：v4 产物 = H·rz·H 已半合并为 U5 形
（rz(π/2)·sx·rz(π+θ)·sx·rz(π/2) = rx(θ)，数值验证 20/20 精确）。
现有循环 merge_1q_runs 在 commute_rz_through_cz 之前跑——rz 跨 cz
合并（H 半块消解 → qiskit 裸 sx 结构）的机会被合成吃掉。

模拟：v4 产物上跑 [rz 穿 cz（双向多轮）+ rz 合并 + H 块消解]
先行，再跑完整现有循环，对比 790 基线。
"""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.post import decompose_to_platform
from qtrail.pure.circuit import Inst, Circuit
from qtrail.pure.post_v2 import (commute_rz_through_cz,
                                 commute_rz_through_cz_rev,
                                 _cancel_adjacent_h, _absorb_h,
                                 _strip_identity_1q, merge_1q_runs,
                                 _sort_segment_1q, _merge_sx_pairs,
                                 slide_x_through_cz, _cancel_cz_global)
from qtrail.pure.post import _u3_to_rzsx_min

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
BP = Path("data/benchpress/qasm")
PI = 3.141592653589793


def merge_adjacent_rz(ops):
    """相邻同比特 rz 合并（v4 尾部的同款）。"""
    out = []
    for inst in ops:
        if out and inst.name == "rz" and out[-1].name == "rz" \
                and out[-1].qubits == inst.qubits:
            out[-1] = Inst("rz", inst.qubits,
                           (out[-1].params[0] + inst.params[0],))
        else:
            out.append(inst)
    return out


def h_first_group(ops, rounds=8):
    """H 优先组：rz 穿 cz（双向、每轮 1 层——生产参数：rz 进入相邻段
    即被 merge 消耗，深层推挤会分离 rz/sx 破坏合成）+ rz 合并 +
    H 块消解 + H 吸收。恒等式：A2、A1、B5、C3。"""
    ops = list(ops)
    for _ in range(rounds):
        before = len(ops)
        commute_rz_through_cz(ops, max_rounds=1)
        commute_rz_through_cz_rev(ops, max_rounds=1)
        ops = merge_adjacent_rz(ops)
        _cancel_adjacent_h(ops)
        _absorb_h(ops)
        _strip_identity_1q(ops)
        if len(ops) == before:
            break
    return ops


def full_existing_loop(ops, spec, resynth_every=0, max_rounds=12):
    """现有不动点循环（decompose_to_platform_v2 的 A2 关闭版）。"""
    out = Circuit(ops[0] if False else 0, name="x")  # 占位不用
    return ops


def main():
    t_all = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    p = None
    for cand in BP.rglob("*.qasm"):
        if "square_heisenberg_N16" in cand.name:
            p = cand
            break
    circ = parse_qasm(p.read_text(encoding="utf-8"))

    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    t0 = time.time()
    res = mapper.map_circuit(circ, circuit_id="heis16")
    print(f"[{time.time()-t0:.0f}s] map_circuit 完成", flush=True)

    from qtrail.pure.router import sabre_route
    from qtrail.pure.mapper import post_route
    torch.manual_seed(42)
    layout = {i: int(res["layout"][i]) for i in range(circ.n)}
    routed, swaps, final_layout = sabre_route(
        circ, spec, layout, seed=42, lam_ms=mapper.lam_ms,
        trials=16, mix_rollout=(circ.n > 10))
    if mapper.use_post:
        routed, removed = post_route(routed, final_layout, spec.adj)
    _, _, final_qc = mapper._v5_pipeline(
        routed, final_layout, circ, 0.0, with_resynth=False)
    ops_now = final_qc.ops
    c_now = Counter(i.name for i in ops_now)
    print(f"我们顺序（不动点循环后）: {dict(c_now)}  1Q="
          f"{c_now['rz']+c_now['sx']+c_now.get('x',0)}", flush=True)

    # v4 输出 → H 优先组先行 → 完整现有循环
    ops_v4 = decompose_to_platform(routed.copy()).ops
    c_v4 = Counter(i.name for i in ops_v4)
    print(f"v4 输出: {dict(c_v4)}  1Q={c_v4['rz']+c_v4['sx']+c_v4.get('x',0)}",
          flush=True)

    t0 = time.time()
    ops_h = h_first_group(ops_v4)
    c_h = Counter(i.name for i in ops_h)
    print(f"[{time.time()-t0:.0f}s] H 优先组后: {dict(c_h)}  1Q="
          f"{c_h['rz']+c_h['sx']+c_h.get('x',0)}", flush=True)

    # H 优先组后接完整现有循环（decompose_to_platform_v2 的
    # fixpoint 部分，resynth_every=0 等价）
    from qtrail.pure.post_v2 import decompose_to_platform_v2
    t0 = time.time()
    circ_h = Circuit(circ.n, name="h")
    circ_h.ops = list(ops_h)
    final_h = decompose_to_platform_v2(circ_h, spec=spec,
                                       resynth_every=0, fixpoint=True)
    ops_f = final_h.ops
    c_f = Counter(i.name for i in ops_f)
    print(f"[{time.time()-t0:.0f}s] H 优先 + 现有循环: {dict(c_f)}  1Q="
          f"{c_f['rz']+c_f['sx']+c_f.get('x',0)}", flush=True)
    print(f"基线 790 vs 新 {c_f['rz']+c_f['sx']+c_f.get('x',0)}"
          f"（qiskit 参考 489）", flush=True)
    print(f"总墙钟 {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
