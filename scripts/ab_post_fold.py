"""post_route 折叠版语义等价验证：旧推挤管线 vs 新单遍折叠。

对每条测试线路：同一路由输出分别走（A）旧后处理（推挤→消解→尾部吸收，
参考实现逐行保留）与（B）新折叠（fold_swaps），各自按最终布局重标记后
与「原线路按初始布局嵌入」的态矢量对照——两路必须都保真等价，
且 B 的残留 SWAP 必须为 0。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.post import post_route as new_post
from qtrail.pure.router import sabre_route
from qtrail.pure.qasm import parse_qasm

# ---- 与 tests/test_pure.py 同款态矢量机制（小端索引约定） ----
I2 = np.eye(2, dtype=complex)
CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
              dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=complex)


def u3_mat(th, ph, la):
    c, s = np.cos(th / 2), np.sin(th / 2)
    return np.array([[c, -np.exp(1j * la) * s],
                     [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]],
                    dtype=complex)


def u1_mat(la):
    return np.diag([1, np.exp(1j * la)])


def _gate_mat(name, params):
    if name == "cx":
        return CX
    if name == "swap":
        return SWAP
    if name == "u3":
        return u3_mat(*params)
    if name == "u1":
        return u1_mat(params[0])
    if name == "cz":
        m = np.eye(4, dtype=complex)
        m[3, 3] = -1
        return m
    raise ValueError(name)


def _apply(sv, mat, qs, n):
    qs = tuple(sorted(qs))
    perm = list(range(n))
    for q in qs:
        perm.remove(q)
    new_perm = perm + list(qs)
    sv = np.transpose(sv.reshape([2] * n), new_perm).reshape(
        2 ** (n - len(qs)), 2 ** len(qs))
    sv = sv @ mat.T
    inv = np.argsort(new_perm)
    return np.transpose(sv.reshape([2] * n), inv).reshape(-1)


def sim(circ: Circuit, n: int) -> np.ndarray:
    sv = np.zeros(2 ** n, dtype=complex)
    sv[0] = 1
    for inst in circ.ops:
        if inst.name in ("barrier", "id"):
            continue
        sv = _apply(sv, _gate_mat(inst.name, inst.params), inst.qubits, n)
    return sv


def relabel(sv, layout, final, n):
    """把物理空间态矢量按 final/layout 重标记回逻辑空间（tests 同款）。"""
    used_to = set(layout.values())
    free_to = [p for p in range(n) if p not in used_to]
    from_to = {}
    for logical, phys in final.items():
        from_to[phys] = layout[logical]
    free_from = [p for p in range(n) if p not in from_to]
    for f, t in zip(free_from, free_to):
        from_to[f] = t
    axes = [0] * n
    for f, t in from_to.items():
        axes[t] = f
    return np.transpose(sv.reshape([2] * n), axes).reshape(-1)


# ---- 旧后处理参考实现（改动前代码逐行保留） ----
def _swap_past(swap_qs, gate):
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


def _push(circ, max_rounds=200):
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


def _cancel(circ):
    ops = circ.ops
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(ops) - 1:
            a, b = ops[i], ops[i + 1]
            if a.name == "cx" and b.name == "cx" \
                    and set(a.qubits) == set(b.qubits):
                del ops[i:i + 2]
                changed = True
                continue
            if a.name == "u1" and b.name == "u1" and a.qubits == b.qubits:
                ops[i] = Inst("u1", a.qubits, (a.params[0] + b.params[0],))
                del ops[i + 1]
                changed = True
                continue
            i += 1
    circ._deps = None


def _absorb(circ, final_layout):
    ops = circ.ops
    while True:
        last_swap = None
        for i in range(len(ops) - 1, -1, -1):
            if ops[i].name == "swap":
                last_swap = i
                break
        if last_swap is None:
            break
        a, b = ops[last_swap].qubits
        touched = set()
        for inst in ops[last_swap + 1:]:
            touched |= set(inst.qubits)
        if (a in touched) or (b in touched):
            break
        del ops[last_swap]
        for logical, phys in final_layout.items():
            if phys == a:
                final_layout[logical] = b
            elif phys == b:
                final_layout[logical] = a
    circ._deps = None


def old_post(circ, final_layout):
    _push(circ)
    _cancel(circ)
    _push(circ)
    _absorb(circ, final_layout)
    return circ


# ---- 测试主体 ----
def make_spec():
    from qtrail.devices import build_grid3x3_spec
    return build_grid3x3_spec()


def build_cases():
    """多样线路：链式、交错、cz/u1、深链、qft 风格。"""
    cases = []
    c = Circuit(3)
    c.append(Inst("u3", (0,), (np.pi / 2, 0.0, np.pi)))
    c.cx(0, 1)
    c.cx(1, 2)
    c.append(Inst("u1", (2,), (0.37,)))
    cases.append(("chain3", c, {0: 0, 1: 4, 2: 8}))
    c = Circuit(4)
    c.cx(0, 1)
    c.swap(0, 1)
    c.cx(1, 2)
    c.append(Inst("u1", (1,), (0.5,)))
    c.swap(0, 1)
    c.cz(2, 3)
    c.u3(0.7, 0.2, 1.1, 3)
    cases.append(("touching", c, {0: 0, 1: 1, 2: 4, 3: 8}))
    rng = np.random.default_rng(7)
    c = Circuit(6)
    for k in range(25):
        q = int(rng.integers(0, 6))
        c.append(Inst("u1", (q,), (rng.random() * 6.28,)))
        a, b = int(rng.integers(0, 6)), int(rng.integers(0, 6))
        if a != b:
            c.cx(a, b)
        if k % 3 == 0:
            s, t = int(rng.integers(0, 6)), int(rng.integers(0, 6))
            if s != t:
                c.swap(s, t)
    cases.append(("random25", c, {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 8}))
    return cases


def main():
    spec = make_spec()
    n = spec.n
    for name, circ, layout in build_cases():
        for seed in (0, 3):
            routed, swaps, fl = sabre_route(circ, spec, dict(layout),
                                            seed=seed)
            # 参照态：原线路按初始布局嵌入
            sv_ref = np.zeros(2 ** n, dtype=complex)
            sv_ref[0] = 1
            for inst in circ.ops:
                phys = tuple(layout[q] for q in inst.qubits)
                sv_ref = _apply(sv_ref, _gate_mat(inst.name, inst.params),
                                phys, n)
            # A) 旧后处理
            ca = routed.copy()
            fa = dict(fl)
            old_post(ca, fa)
            sv_a = relabel(sim(ca, n), layout, fa, n)
            fid_a = abs(np.vdot(sv_ref, sv_a)) ** 2
            # B) 新折叠
            cb = routed.copy()
            fb = dict(fl)
            removed = new_post(cb, fb)[1]
            sv_b = relabel(sim(cb, n), layout, fb, n)
            fid_b = abs(np.vdot(sv_ref, sv_b)) ** 2
            ok = fid_a > 0.9999 and fid_b > 0.9999 and cb.count("swap") == 0
            print(f"{name:10s} seed={seed} swaps_routed={swaps} "
                  f"fid_old={fid_a:.6f} fid_fold={fid_b:.6f} "
                  f"fold_removed={removed} residual={cb.count('swap')} "
                  f"{'OK' if ok else 'FAIL!'}")
            if not ok:
                raise SystemExit(1)
    print("\nALL EQUIVALENT (fold: zero residual swaps everywhere)")


if __name__ == "__main__":
    main()
