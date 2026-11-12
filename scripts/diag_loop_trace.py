# -*- coding: utf-8 -*-
"""循环轨迹诊断（2026-11-13）：_decompose_ablation 的循环
每轮 1Q 门数——查"过冲"（中途最小点优于最终稳定态）。qiskit O3
循环 = 最小点停止（_optimization_check_minimum_point），我们 =
固定点。若过冲存在 → 早停修复候选。
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
from qtrail.pure.router import sabre_route
from qtrail.pure.mapper import post_route
from qtrail.pure.circuit import Circuit, Inst

# 复刻 _decompose_ablation 的循环（含每轮打印）——直接 monkey
# patch 组件：用 post_v2 的函数手动跑循环
from qtrail.pure.post_v2 import (
    _sort_segment_1q, _merge_sx_pairs, slide_x_through_cz, merge_1q_runs,
    _cancel_cz_global, _cancel_adjacent_h, _absorb_h, _strip_identity_1q,
    commute_rz_through_cz, commute_rz_through_cz_rev, slide_1q_through)
from qtrail.pure.post import _u3_to_rzsx_min, merge_adjacent_rz, \
    decompose_to_platform

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
BP = Path("data/benchpress/qasm")


def n1q(ops):
    return sum(1 for i in ops if i.name in ("rz", "sx", "x"))


def main():
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
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    p = None
    for cand in BP.rglob("*.qasm"):
        if "square_heisenberg_N16" in cand.name:
            p = cand
            break
    circ = parse_qasm(p.read_text(encoding="utf-8"))
    res = mapper.map_circuit(circ, circuit_id="h16")
    torch.manual_seed(42)
    layout = {i: int(res["layout"][i]) for i in range(circ.n)}
    routed, swaps, fl = sabre_route(circ, spec, layout, seed=42,
                                    lam_ms=mapper.lam_ms, trials=16,
                                    mix_rollout=(circ.n > 10))
    routed, removed = post_route(routed, fl, spec.adj)

    # v4 分解 → 手跑循环（resynth_every=1 关闭 A2 简化，只看 1Q 轨迹）
    from qtrail.pure.post_v2 import commute_cancel_cx
    work = Circuit(routed.n, name="h")
    work.ops = list(routed.ops)
    work.measures = list(routed.measures)
    commute_cancel_cx(work.ops)
    out = decompose_to_platform(work)
    print(f"v4 后: 1Q={n1q(out.ops)}", flush=True)
    for rnd in range(12):
        changed = False
        _sort_segment_1q(out.ops)
        _merge_sx_pairs(out.ops)
        slide_x_through_cz(out.ops, max_rounds=4)
        if merge_1q_runs(out.ops) > 0:
            new_ops = []
            for inst in out.ops:
                if inst.name == "u3":
                    for gname, gparams in _u3_to_rzsx_min(*inst.params):
                        new_ops.append(Inst(gname, inst.qubits, gparams))
                else:
                    new_ops.append(inst)
            out.ops = new_ops
            changed = True
        merge_adjacent_rz(out)
        before = len(out.ops)
        slide_1q_through(out.ops)
        commute_rz_through_cz(out.ops, max_rounds=1)
        commute_rz_through_cz_rev(out.ops, max_rounds=1)
        _absorb_h(out.ops)
        _cancel_cz_global(out.ops)
        _cancel_adjacent_h(out.ops)
        _strip_identity_1q(out.ops)
        changed = changed or len(out.ops) != before
        c = Counter(i.name for i in out.ops)
        print(f"轮 {rnd+1}: 1Q={n1q(out.ops)} ({dict(c)}) "
              f"changed={changed}", flush=True)
        if not changed:
            break


if __name__ == "__main__":
    main()
