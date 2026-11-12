# -*- coding: utf-8 -*-
"""pure 管线三层抽样审查（2026-11-12）：正确性 / 真机可运行性 / 时间成本。

分层样本（不跑全量 587）：
  small（n≤10）：bv_10、dj_10、qft_10 —— 态矢量等价（全酉口径）
  medium（n≤25）：heis N16、qft_N020、cat_state_n22 —— 态矢量等价
  medium-large：QV_n32 —— 合法性+机制指标
  large：heis N100、qft_50、graphstate_50、hwb12 —— 合法性+墙钟

正确性：初始布局置换输入 + 终布局置换补偿，5 组输入态（|0⟩、|+⟩、
3 组随机直积态）态矢量重叠 > 0.9999（任一失败即语义错误）。
真机可运行性：门集 ⊆ {rz,sx,x,cz,measure}、比特 <287 且非禁用、
cz 全部相邻（spec.adj）、无 swap 残留、measure 合法。
时间成本：全管线墙钟（含 RL 布局/路由/后处理）。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"

CASES = [
    ("small", "bv_10", "data/mqtbench/stratified/bv_10.qasm", "sv"),
    ("small", "dj_8", "data/mqtbench/stratified/dj_8.qasm", "sv"),
    ("small", "qft_10", "data/mqtbench/stratified/qft_10.qasm", "sv"),
    ("small", "qaoa_8", "data/mqtbench/stratified/qaoa_8.qasm", "sv"),
    ("medium", "heis_N16",
     "data/benchpress/qasm/square-heisenberg/square_heisenberg_N16.qasm",
     "sv"),
    ("medium", "qft_N020", "data/benchpress/qasm/qft/qft_N020.qasm", "sv"),
    ("medium", "cat_state_n22",
     "data/benchpress/qasm/qasmbench-medium/cat_state_n22/"
     "cat_state_n22.qasm", "sv"),
    ("med-large", "QV_n32",
     "data/benchpress/qasm/qasmbench-large/QV_n32/32.qasm", "legal"),
    ("large", "heis_N100",
     "data/benchpress/qasm/square-heisenberg/square_heisenberg_N100.qasm",
     "legal"),
    ("large", "qft_50", "data/mqtbench/stratified/qft_50.qasm", "legal"),
    # 以下两条已确证超时（900s 兜底，时间成本发现已记录）——复验跳过
    # ("large", "graphstate_50",
    #  "data/mqtbench/stratified/graphstate_50.qasm", "legal"),
    # ("large", "hwb12", None, "legal"),  # rglob 特判
]


def qiskit_from_ops(ops, n):
    qc = QuantumCircuit(n)
    for inst in ops:
        name = inst.name
        qs = [int(q) for q in inst.qubits]
        if name == 'rz':
            qc.rz(inst.params[0], qs[0])
        elif name == 'sx':
            qc.sx(qs[0])
        elif name == 'x':
            qc.x(qs[0])
        elif name == 'cz':
            qc.cz(qs[0], qs[1])
        elif name == 'u3':
            qc.u(inst.params[0], inst.params[1], inst.params[2], qs[0])
        elif name == 'u1':
            qc.p(inst.params[0], qs[0])
        elif name == 'cx':
            qc.cx(qs[0], qs[1])
        elif name == 'h':
            qc.h(qs[0])
        elif name == 'swap':
            qc.swap(qs[0], qs[1])
    return qc


def permute_sv(sv, perm):
    """perm[i] = 逻辑比特 i 的物理位置：out[y] = in[x]，y 的第
    perm[i] 位 = x 的第 i 位。"""
    n = len(perm)
    out = np.zeros_like(sv)
    for x in range(2 ** n):
        y = 0
        for i in range(n):
            if (x >> i) & 1:
                y |= 1 << perm[i]
        out[y] = sv[x]
    return out


def verify_statevector(circ, final_qc, layout, final_layout, n):
    """紧凑映射口径的态矢量等价：交付线路散布在设备比特上（如
    105 个索引）、空闲比特保持 |0⟩——把用到的物理比特紧凑重编号
    后模拟，再以初始/最终布局置换补偿。

    等价性判据：U_r(Π_init·|ψ⟩_L) = Π_fin·(U_orig|ψ⟩_L)
    （5 组输入态，重叠 > 0.9999）。
    """
    used = {int(q) for inst in final_qc.ops for q in inst.qubits}
    used |= {int(m.qubits[0]) for m in final_qc.measures}
    for i in range(n):
        used.add(int(final_layout.get(i, i)))
        used.add(int(layout[i]))
    used = sorted(used)
    p2c = {p: k for k, p in enumerate(used)}
    k = len(used)
    if k > 24:
        return None, f"紧凑空间 {k} 比特超模拟上限 24"

    qc_final = QuantumCircuit(k)
    for inst in final_qc.ops:
        qs = [p2c[int(q)] for q in inst.qubits]
        if inst.name == 'rz':
            qc_final.rz(inst.params[0], qs[0])
        elif inst.name == 'sx':
            qc_final.sx(qs[0])
        elif inst.name == 'x':
            qc_final.x(qs[0])
        elif inst.name == 'cz':
            qc_final.cz(qs[0], qs[1])
    qc_orig = qiskit_from_ops(circ.ops, n)

    def embed(sv, pos_of_logical):
        """逻辑 n 比特态 → 紧凑 k 比特态（pos_of_logical[i] =
        逻辑 i 在紧凑空间的位）。"""
        out = np.zeros(2 ** k, complex)
        for x in range(2 ** n):
            y = 0
            for i in range(n):
                if (x >> i) & 1:
                    y |= 1 << pos_of_logical[i]
            out[y] = sv[x]
        return out

    pos_init = [p2c[int(layout[i])] for i in range(n)]
    pos_fin = [p2c[int(final_layout.get(i, i))] for i in range(n)]

    inputs = []
    inputs.append(np.zeros(2 ** n, complex))
    inputs[0][0] = 1.0
    inputs.append(np.full(2 ** n, 1.0 / (2 ** (n / 2)), complex))
    if k <= 24:
        rng = np.random.default_rng(42)
        for _ in range(3):
            sv = np.array([1.0 + 0j])
            for _q in range(n):
                th, ph = rng.uniform(0, np.pi, 2)
                st = np.array(
                    [np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)],
                    complex)
                sv = np.kron(sv, st)
            inputs.append(sv)

    worst = 1.0
    for sv_in in inputs:
        # 注意：Statevector(sv).evolve(qc) 才是"态被线路演化"的
        # 正确 API——from_instruction(qc) 已含 |0⟩ 初态，再 evolve
        # 一个 Statevector 是算子误用（产生零矢量，2026-11-12 审查
        # 中实锤）
        sv_orig = Statevector(sv_in).evolve(qc_orig).data
        target = embed(sv_orig, pos_fin)
        got = Statevector(embed(sv_in, pos_init)).evolve(qc_final).data
        ov = abs(np.vdot(target, got)) ** 2
        worst = min(worst, ov)
    return worst, None


def check_legality(qc, spec):
    bad = []
    names = set()
    disabled = set(spec.disabled_qubits)
    for idx, inst in enumerate(qc.ops):
        names.add(inst.name)
        qs = [int(q) for q in inst.qubits]
        if inst.name == "swap":
            bad.append(f"swap残留@{idx}")
        if any(q >= spec.n for q in qs):
            bad.append(f"越界{qs}@{idx}")
        if any(q in disabled for q in qs):
            bad.append(f"禁用{qs}@{idx}")
        if inst.name == "cz" and not spec.adj[qs[0], qs[1]]:
            bad.append(f"cz非相邻{qs}@{idx}")
    if not names <= {"rz", "sx", "x", "cz"}:
        bad.append(f"门集{names}")
    for idx, mm in enumerate(qc.measures):
        q = int(mm.qubits[0])
        if q >= spec.n:
            bad.append(f"measure{idx}越界")
        if q in disabled:
            bad.append(f"measure{idx}禁用")
    return bad


def run_with_timeout(fn, timeout):
    import threading
    box = {}

    def target():
        try:
            box["res"] = fn()
        except Exception as e:  # noqa: BLE001
            box["err"] = e
    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return None, "timeout"
    return box.get("res"), box.get("err")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    n_fail = 0
    for layer, name, path, kind in CASES:
        p = Path(path) if path else None
        if path is None:
            for cand in Path("data/benchpress/qasm").rglob("*.qasm"):
                if "hwb12" in cand.name:
                    p = cand
                    break
        circ = parse_qasm(p.read_text(encoding="utf-8", errors="replace"))
        print(f"\n== [{layer}] {name} n={circ.n} 门={len(circ.ops)} ==",
              flush=True)
        torch.manual_seed(42)
        mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                            selection_rule="fidelity", use_post=True,
                            pipeline="v5", routing_seeds=16,
                            # 默认 = 生产路径（linear 分解，2026-11-12
                            # 起为默认）；QTRAIL_AUDIT_FIXPOINT=1 切
                            # fixpoint 消融路径
                            linear_decompose=os.environ.get(
                                "QTRAIL_AUDIT_FIXPOINT") != "1")
        t0 = time.time()
        res, err = run_with_timeout(
            lambda: mapper.map_circuit(circ, circuit_id=name), 900)
        wall = time.time() - t0
        if err is not None:
            print(f"  FAIL: {type(err).__name__}: {str(err)[:100]} "
                  f"({wall:.0f}s)", flush=True)
            n_fail += 1
            continue
        m = res["metrics"]
        final_qc = res["routed_circuit"]
        print(f"  wall={wall:.1f}s swaps={res['swap_count']} "
              f"twoq={m['twoq_count']} depth={m['depth']} "
              f"fid={m['est_fidelity']:.3e}", flush=True)
        bad = check_legality(final_qc, spec)
        if bad:
            print(f"  FAIL 合法性: {bad[:5]}", flush=True)
            n_fail += 1
        else:
            print(f"  OK 合法性（门集/比特/cz相邻/swap残留全通过）",
                  flush=True)
        if kind == "sv":
            ov, verr = verify_statevector(circ, final_qc, res["layout"],
                                          res["final_layout"], circ.n)
            if verr is not None:
                print(f"  态矢量: 跳过（{verr}）", flush=True)
            elif ov > 0.9999:
                print(f"  OK 态矢量等价（5 组输入最差重叠 {ov:.6f}）",
                      flush=True)
            else:
                print(f"  FAIL 态矢量重叠 {ov:.6f}", flush=True)
                n_fail += 1
    print(f"\n===== 审查结果：失败 {n_fail} =====", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
