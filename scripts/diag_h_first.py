# -*- coding: utf-8 -*-
"""H 优先消解实验 v2（2026-11-13）：heisenberg N16。

假设：qiskit 489 的关键 = pass 顺序（H 对消先行 → 1Q 合成后行），
我们的不动点循环把 H 混入 u3 合成 → H 身份丢失 → sx/rz 膨胀。

v2 修复：零滑动先行（v4 输出相邻 H 结构直接消解，O(n) 单遍），
滑动版为第二轮（若零滑动不足）。输出直接 stdout（无管道缓冲）。
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
from qtrail.pure.post import decompose_to_platform

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


def h_cancel_pass(ops):
    """零滑动单遍：相邻 H·rz(θ)·H → rx(θ)、相邻 H·H → I。
    返回 (ops, n_absorb, n_cancel)。O(n) 单遍。"""
    out = []
    n_absorb = n_cancel = 0
    i = 0
    n = len(ops)
    while i < n:
        # H·rz·H → rx（7 门窗口）
        if (is_h_block(ops, i) and i + 6 < n
                and ops[i + 3].name == "rz"
                and ops[i + 3].qubits == ops[i].qubits
                and is_h_block(ops, i + 4)):
            q = ops[i].qubits
            th = ops[i + 3].params[0]
            out.append(Inst("rx", q, (th,)))
            n_absorb += 1
            i += 7
            continue
        # H·H → I（6 门窗口）
        if (is_h_block(ops, i) and i + 5 < n
                and is_h_block(ops, i + 3)
                and ops[i].qubits == ops[i + 3].qubits):
            n_cancel += 1
            i += 6
            continue
        out.append(ops[i])
        i += 1
    return out, n_absorb, n_cancel


def merge_adjacent(ops):
    """相邻同比特 rz/rx 合并（O(n)）。"""
    out = []
    for inst in ops:
        if out and inst.name in ("rz", "rx") \
                and out[-1].name == inst.name \
                and out[-1].qubits == inst.qubits:
            out[-1] = Inst(inst.name, inst.qubits,
                           (out[-1].params[0] + inst.params[0],))
        else:
            out.append(inst)
    return out


def zero_slide_h_first(ops, max_rounds=30):
    """零滑动 H 优先：H 消解 pass ↔ rz/rx 合并 交替至稳定。"""
    ops = list(ops)
    for _ in range(max_rounds):
        ops2, na, nc = h_cancel_pass(ops)
        ops2 = merge_adjacent(ops2)
        if ops2 == ops:
            return ops2, na, nc
        ops = ops2
    return ops, na, nc


def finalize(ops):
    """rx → 4 门展开 + 段内 u3 合成（模拟现有 merge 语义）。"""
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
    return out


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
    t0 = time.time()
    routed, swaps, final_layout = sabre_route(
        circ, spec, layout, seed=42, lam_ms=mapper.lam_ms,
        trials=16, mix_rollout=(circ.n > 10))
    if mapper.use_post:
        routed, removed = post_route(routed, final_layout, spec.adj)
    print(f"[{time.time()-t0:.0f}s] sabre_route 完成", flush=True)

    t0 = time.time()
    _, _, final_qc = mapper._v5_pipeline(
        routed, final_layout, circ, 0.0, with_resynth=False)
    ops_now = final_qc.ops
    c_now = Counter(i.name for i in ops_now)
    print(f"[{time.time()-t0:.0f}s] 我们顺序（不动点循环后）: {dict(c_now)}"
          f"  1Q={c_now['rz']+c_now['sx']+c_now.get('x',0)}", flush=True)

    # v4 分解输出（不动点循环前）
    routed_v4 = routed.copy()
    final_qc_v4 = decompose_to_platform(routed_v4)
    ops_v4 = final_qc_v4.ops
    c_v4 = Counter(i.name for i in ops_v4)
    n_h = sum(1 for i in range(len(ops_v4) - 2) if is_h_block(ops_v4, i))
    print(f"v4 分解输出（循环前）: {dict(c_v4)}  1Q="
          f"{c_v4['rz']+c_v4['sx']+c_v4.get('x',0)}  H块={n_h}",
          flush=True)

    # 零滑动 H 优先
    t0 = time.time()
    ops_h, na, nc = zero_slide_h_first(ops_v4)
    c_h = Counter(i.name for i in ops_h)
    print(f"[{time.time()-t0:.0f}s] 零滑动 H 优先（H·rz·H→rx {na} 次、"
          f"H·H→I {nc} 次）: {dict(c_h)}  rx 计1门时 1Q="
          f"{c_h['rz']+c_h['sx']+c_h.get('rx',0)+c_h.get('x',0)}",
          flush=True)
    ops_f = finalize(ops_h)
    c_f = Counter(i.name for i in ops_f)
    print(f"rx 展开后: {dict(c_f)}  1Q="
          f"{c_f['rz']+c_f['sx']+c_f.get('x',0)}", flush=True)
    print(f"参考：qiskit O3 = 1Q 489（rz 175 + sx 258 + x 3）", flush=True)
    print(f"总墙钟 {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
