# -*- coding: utf-8 -*-
"""理想 H 消解模拟（2026-11-13）：A2 off 产物上模拟 qiskit 语义的
1Q 优化链：H·rz·H → rx（吸收）→ rx 池合并 → H·H 对消 → rz 合并 →
最后分解（rx 4 门 / H 3 门 / u3 5 门）。对比 qiskit 489。

模拟在 cz 段内做（H 不跨 cz——B1 恒等式：H 穿 cz 破坏骨架），
H·rz·H 吸收为相邻窗口（与 _absorb_h 同语义），rx 池跨游程合并。
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
from qtrail.pure.circuit import Inst

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
BP = Path("data/benchpress/qasm")
PI = 3.141592653589793
H_A = PI / 2


def is_h_block(ops, i):
    if i + 2 >= len(ops):
        return False
    a, b, c = ops[i], ops[i + 1], ops[i + 2]
    return (a.name == "rz" and b.name == "sx" and c.name == "rz"
            and a.qubits == b.qubits == c.qubits
            and abs(a.params[0] - H_A) < 1e-9
            and abs(c.params[0] - H_A) < 1e-9)


def ideal_h_optimize(ops, verbose=False):
    """理想 H 消解：窗口 H·rz·H → rx(θ)（符号门）、相邻 H·H → I、
    相邻 rz 合并、rx·rx 合并。返回 (ops, 统计)。"""
    ops = list(ops)
    n_absorb = n_cancel_h = 0
    changed = True
    rounds = 0
    while changed and rounds < 50:
        changed = False
        rounds += 1
        i = 0
        n = len(ops)
        while i < n:
            # H·rz(θ)·H → rx(θ)（7 门 → 1 符号门）
            if (is_h_block(ops, i) and i + 6 < n
                    and ops[i + 3].name == "rz"
                    and ops[i + 3].qubits == ops[i].qubits
                    and is_h_block(ops, i + 4)):
                q = ops[i].qubits
                th = ops[i + 3].params[0]
                ops[i:i + 7] = [Inst("rx", q, (th,))]
                n -= 6
                n_absorb += 1
                changed = True
                continue
            # H·H → I（相邻 H 块）
            if is_h_block(ops, i) and is_h_block(ops, i + 3) \
                    and ops[i].qubits == ops[i + 3].qubits:
                del ops[i:i + 6]
                n -= 6
                n_cancel_h += 1
                changed = True
                continue
            i += 1
        # rz·rz 合并 + rx·rx 合并（同比特相邻）
        i = 0
        while i < len(ops) - 1:
            a, b = ops[i], ops[i + 1]
            if a.name == b.name == "rz" and a.qubits == b.qubits:
                ops[i] = Inst("rz", a.qubits, (a.params[0] + b.params[0],))
                del ops[i + 1]
                changed = True
                continue
            if a.name == b.name == "rx" and a.qubits == b.qubits:
                ops[i] = Inst("rx", a.qubits, (a.params[0] + b.params[0],))
                del ops[i + 1]
                changed = True
                continue
            i += 1
    if verbose:
        print(f"  吸收 H·rz·H→rx: {n_absorb} 次、H·H 对消: {n_cancel_h} 次"
              f"、循环 {rounds} 轮")
    return ops


def final_decompose(ops):
    """rx → 4 门（rz(π)·sx·rz(π−θ)·sx）、H 块 → 3 门、游程合成 u3 →
    ZSXXZ 5 门。统计最终门数（乐观：rz 合并到极致）。"""
    out = []
    for inst in ops:
        if inst.name == "rx":
            th = inst.params[0]
            out.append(Inst("rz", inst.qubits, (PI,)))
            out.append(Inst("sx", inst.qubits, ()))
            out.append(Inst("rz", inst.qubits, (PI - th,)))
            out.append(Inst("sx", inst.qubits, ()))
        else:
            out.append(inst)
    # 游程矩阵合成（模拟 merge_1q_runs）+ ZSXXZ
    # 简化：按段（cz 分隔）每比特合成 u3 → 5 门
    final = []
    seg = []
    for inst in out:
        if inst.name == "cz":
            final.extend(seg)
            final.append(inst)
            seg = []
        else:
            seg.append(inst)
    final.extend(seg)
    return final


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
    res = mapper.map_circuit(circ, circuit_id="heis16")

    # A2 off 产物
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
    ops = final_qc.ops
    c = Counter(i.name for i in ops)
    print(f"A2 off 基线: {dict(c)}  1Q={c['rz']+c['sx']+c.get('x',0)}")

    # 理想 H 消解
    ops2 = ideal_h_optimize(ops, verbose=True)
    c2 = Counter(i.name for i in ops2)
    print(f"理想 H 消解后（rx 符号门计 1 门）: {dict(c2)}  "
          f"1Q={c2['rz']+c2['sx']+c2.get('rx',0)+c2.get('x',0)}")
    # rx 分解展开后
    ops3 = final_decompose(ops2)
    c3 = Counter(i.name for i in ops3)
    print(f"rx 展开后（乐观，游程未再合成）: {dict(c3)}  "
          f"1Q={c3['rz']+c3['sx']+c3.get('x',0)}")
    print(f"参考：qiskit O3 = 1Q 489（rz 175 + sx 258 + x 3 + 其他）")


if __name__ == "__main__":
    main()
