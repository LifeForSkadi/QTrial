# -*- coding: utf-8 -*-
"""1Q 差距拆解侦察（2026-11-13）：heisenberg N16 上量化
(1) 理想跨段 rz 合并（rz 与 cz 对易 A2——按比特投影后相邻 rz 必可合并）
(2) H 块理想消解（相邻/滑行 H 对 + H·rz·H → rx）
后我们的产物门数 vs qiskit 436。决定下一机制优先级：
后处理可补 → 全局 rz 重排；不可补 → 上游模板结构。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
BP = Path("data/benchpress/qasm")
TARGET = "bp_square-heisenberg_square_heisenberg_N16"

PI = 3.141592653589793
H_ANGLE = PI / 2


def count_ops(ops):
    c = {}
    for inst in ops:
        c[inst.name] = c.get(inst.name, 0) + 1
    return c


def ideal_rz_merge(ops):
    """理想跨段 rz 合并：按比特投影，相邻 rz（投影序列中无 sx/x 间隔）
    合并。rz 与 cz 对易（A2 恒等式）→ 段边界 rz 必可合并。返回合并后
    (rz 数, sx 数, x 数) 与每比特 rz 组数。"""
    by_q = {}
    for inst in ops:
        if inst.name in ("rz", "sx", "x") and inst.nq == 1:
            q = int(inst.qubits[0])
            by_q.setdefault(q, []).append(inst)
    n_rz = n_sx = n_x = 0
    groups = 0
    for q, seq in by_q.items():
        cur_rz = 0.0
        has_rz = False
        in_rz = False
        for inst in seq:
            if inst.name == "rz":
                if not in_rz:
                    if has_rz:
                        groups += 1  # 上一组结束（被 sx/x 隔开）
                    in_rz = True
                    has_rz = True
                cur_rz += inst.params[0]
            else:
                if inst.name == "sx":
                    n_sx += 1
                else:
                    n_x += 1
                in_rz = False
        if has_rz:
            groups += 1
        n_rz += groups if False else 0  # 占位（下面统一算）
    # 简化：rz 数下界 = 每比特"最大 rz 组"数（每组 1 个 rz）
    n_rz = 0
    for q, seq in by_q.items():
        in_rz = False
        for inst in seq:
            if inst.name == "rz":
                if not in_rz:
                    n_rz += 1
                    in_rz = True
            else:
                in_rz = False
    return n_rz, n_sx, n_x


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

    p = next(BP.rglob(f"{TARGET.split('_', 1)[1]}.qasm".replace(
        "square-heisenberg_square_heisenberg_N16", "square_heisenberg_N16")))
    # 兜底：直接搜
    p = None
    for cand in BP.rglob("*.qasm"):
        if "square_heisenberg_N16" in cand.name:
            p = cand
            break
    assert p is not None, "找不到 heisenberg N16"
    circ = parse_qasm(p.read_text(encoding="utf-8"))

    t0 = time.time()
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    res = mapper.map_circuit(circ, circuit_id=TARGET)
    ops = res["routed_circuit"].ops
    print(f"管线墙钟 {time.time()-t0:.0f}s")
    c = count_ops(ops)
    print(f"[我们产物] {c}  （总 1Q = {c.get('rz',0)+c.get('sx',0)+c.get('x',0)}）")

    n_rz, n_sx, n_x = ideal_rz_merge(ops)
    tot = n_rz + n_sx + n_x
    print(f"[理想 rz 合并后] rz={n_rz} sx={n_sx} x={n_x} "
          f"总 1Q={tot}（qiskit 参考 436 = rz175+sx258+x3）")
    print(f"[差距] rz {c.get('rz',0)}→{n_rz}（qiskit 175）、"
          f"sx {c.get('sx',0)}→{n_sx}（qiskit 258）")

    # 诊断：rz 组为何这么多——每组 rz 数与位置分布
    from collections import Counter
    by_q = {}
    for inst in ops:
        if inst.name in ("rz", "sx", "x") and inst.nq == 1:
            q = int(inst.qubits[0])
            by_q.setdefault(q, []).append(inst)
    dist = Counter()
    for q, seq in by_q.items():
        in_rz = False
        for inst in seq:
            if inst.name == "rz":
                if not in_rz:
                    dist[1] += 1
                    in_rz = True
            else:
                in_rz = False
    print(f"[每比特 rz 组数分布] {dict(dist)}（总组数 = 理想 rz 数）")


if __name__ == "__main__":
    main()
