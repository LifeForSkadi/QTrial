# -*- coding: utf-8 -*-
"""decompose_linear 逐阶段插桩（qv32 dump）：定位 1Q=26,066 的来源。

每阶段输出：门数、1Q 数、cz 数、段内游程分布、孤立 u3H 数。
"""
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.post_v2 import (_U3_H, _cancel_cz_global, _cluster_cz_segments,
                                 _expand_to_u3cz, _mar_from_ops,
                                 _merge_runs_u3cz, _merge_sx_pairs,
                                 _strip_identity_1q, _u3_level_to_platform,
                                 collect_resynthesize_2q_blocks,
                                 commute_cancel_cx, slide_x_through_cz)


def stage_stats(ops, label):
    n1q = sum(1 for i in ops if i.nq == 1)
    ncz = sum(1 for i in ops if i.nq == 2)
    seg_runs = []
    iso_h = 0
    i, n = 0, len(ops)
    while i < n:
        if ops[i].nq == 1:
            j = i
            seg = []
            while j < n and ops[j].nq == 1:
                seg.append(ops[j])
                j += 1
            by_q = {}
            for g in seg:
                by_q.setdefault(g.qubits[0], []).append(g)
            for q, gs in by_q.items():
                seg_runs.append(len(gs))
                if (len(gs) == 1 and gs[0].name == "u3"
                        and gs[0].params == _U3_H):
                    iso_h += 1
            i = j
        else:
            i += 1
    print(f"[{label}] 门={len(ops)} 1Q={n1q} cz={ncz} "
          f"段内游程={len(seg_runs)} 均值={np.mean(seg_runs):.2f} "
          f"分布={dict(sorted(Counter(seg_runs).items()))} "
          f"孤立u3H={iso_h}", flush=True)
    return ops


def main():
    with open("scripts/tmp_decompose_ab/dump_qv32.pkl", "rb") as f:
        circ = pickle.load(f)
    print(f"dump: 门={len(circ.ops)} u3={circ.count('u3')} "
          f"cx={circ.count('cx')} u1={circ.count('u1')} "
          f"swap={circ.count('swap')}", flush=True)

    ops = list(circ.ops)
    removed = commute_cancel_cx(ops)
    print(f"[S1 A3] 消解 cx={removed} 门={len(ops)}", flush=True)

    ops = _expand_to_u3cz(ops)
    stage_stats(ops, "S2 展开")

    m1 = _merge_runs_u3cz(ops)
    stage_stats(ops, f"S2 合并（{m1} runs）")

    ops = _u3_level_to_platform(ops)
    stage_stats(ops, "S3 翻译")

    for rnd in range(2):
        _merge_sx_pairs(ops)
        slide_x_through_cz(ops, max_rounds=12)
        _cluster_cz_segments(ops)
        c = Circuit(circ.n, name=circ.name)
        c.ops = ops
        from qtrail.config import load_device_config
        from qtrail.devices import build_tianyan287_spec
        spec = build_tianyan287_spec(load_device_config())
        c = collect_resynthesize_2q_blocks(c, spec, max_blocks=60)
        ops = _expand_to_u3cz(c.ops, expand_swap=False)
        _merge_runs_u3cz(ops)
        ops = _u3_level_to_platform(ops)
        stage_stats(ops, f"S4 A2 轮 {rnd + 1} 后")

    _cancel_cz_global(ops)
    _mar_from_ops(ops)
    _strip_identity_1q(ops)
    stage_stats(ops, "S5 收尾")


if __name__ == "__main__":
    main()
