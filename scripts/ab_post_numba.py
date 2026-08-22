"""push_swaps Numba 版 vs Python 参考版逐位等价验证。

对每条线路：同一输入副本分别走 Numba 内核与 Python 参考实现
（原版算法逐行保留于此），断言输出 ops 序列（名称/比特/参数）
完全一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qtrail.devices import build_tianyan287_spec
from qtrail.config import load_device_config
from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route
from qtrail.pure.post_numba import numba_push_swaps

spec = build_tianyan287_spec(load_device_config())


def _swap_past(swap_qs, gate):
    """Python 参考版（post.py 原实现逐行保留）。"""
    a, b = swap_qs
    qs = gate.qubits
    if gate.name in ("barrier", "measure"):
        return None
    if gate.name == "cx":
        if a not in qs and b not in qs:
            return gate
        if a in qs and b in qs:
            return Inst("cx", (qs[1], qs[0]), gate.params)
        return Inst("cx", tuple(b if q == a else (a if q == b else q)
                                 for q in qs), gate.params)
    if gate.nq == 2:
        if a not in qs and b not in qs:
            return gate
        if a in qs and b in qs:
            return gate
        return Inst(gate.name, tuple(b if q == a else (a if q == b else q)
                                     for q in qs), gate.params)
    if gate.nq == 1:
        q = qs[0]
        if q == a:
            return Inst(gate.name, (b,), gate.params)
        if q == b:
            return Inst(gate.name, (a,), gate.params)
        return gate
    return None


def py_push_swaps(circ, max_rounds=200):
    changed = True
    rounds = 0
    while changed and rounds < max_rounds:
        rounds += 1
        changed = False
        ops = circ.ops
        i = 0
        while i < len(ops) - 1:
            a, b = ops[i], ops[i + 1]
            if a.name == "swap":
                if b.name == "swap" and set(a.qubits) == set(b.qubits):
                    del ops[i:i + 2]
                    changed = True
                    continue
                new_b = _swap_past(a.qubits, b)
                if new_b is not None:
                    ops[i] = new_b
                    ops[i + 1] = a
                    changed = True
                    i += 1
                    continue
            i += 1
    circ._deps = None
    return circ


def ops_sig(circ):
    return [(i.name, tuple(i.qubits), tuple(i.params)) for i in circ.ops]


def check(circ: Circuit, label: str):
    ref = py_push_swaps(circ.copy())
    num = Circuit(circ.n)
    num.ops = numba_push_swaps(circ.copy())
    sig_r, sig_n = ops_sig(ref), ops_sig(num)
    ok = sig_r == sig_n
    print(f"{label:34s} n={circ.n:3d} ops={len(circ.ops):6d} "
          f"swaps_in={circ.count('swap'):5d} out={len(sig_n):6d} "
          f"ref_out={len(sig_r):6d} {'OK' if ok else 'MISMATCH!'}")
    if not ok:
        for k in range(min(len(sig_r), len(sig_n))):
            if sig_r[k] != sig_n[k]:
                print(f"  first diff at {k}: ref={sig_r[k]} numba={sig_n[k]}")
                break
        raise SystemExit(1)


def main():
    # 1) 小电路（来自 test_post 规则覆盖）
    c = Circuit(4)
    c.cx(0, 1)
    c.swap(0, 1)
    c.cx(1, 2)
    c.append(Inst("u1", (1,), (0.5,)))
    c.swap(0, 1)
    check(c, "touching-gates")

    # 2) 路由后的中大型线路（大量 swap 推挤压力）
    for path in ("data/mqtbench/stratified/qft_25.qasm",
                 "data/benchpress/qasm/qasmbench-medium/"
                 "cat_state_n22/cat_state_n22.qasm",
                 "data/benchpress/qasm/qaoa/qaoa_barabasi_albert_N69_3reps.qasm"):
        if not Path(path).exists():
            print(f"[skip] {path}")
            continue
        text = Path(path).read_text(encoding="utf-8")
        circ = parse_qasm(text)
        layout = {i: int(np.arange(circ.n)[i] % spec.n) for i in range(circ.n)}
        routed, _, _ = sabre_route(circ, spec, layout, seed=0)
        check(routed, Path(path).stem)

    # 3) 手工深链 + 交错 swap
    c = Circuit(6)
    for k in range(20):
        c.cx(k % 5, (k + 1) % 5)
        c.swap(k % 6, (k + 3) % 6)
        c.append(Inst("u1", (k % 6,), (0.1 * k,)))
    check(c, "synthetic-chain")

    print("\nALL EQUAL")


if __name__ == "__main__":
    main()
