"""生产路径语义回归门禁（academic_review_gaps.md §5 第 2 项）。

预注册判据（2026-11-13，实现前写定）

语料（6 条）：
  - examples/qft5（n=5）、ghz10（n=10）、qaoa12（n=12）——现有生产样例
  - synth_special_rz（n=6）：rz(0)/rz(π/4)/rz(π/2)/rz(π) 特角 + x/sx/h +
    cx 链 + cz 对 + u3(π/2)/u3(π)/u1
  - synth_special_h（n=6）：H 块夹 rz（吸收路径）+ sx 对 + cz 对消 +
    rz(π) 特殊角
  - synth_fold（n=9）：合成可被 fold 吸收的 routed 形态（不相交边
    swap + 同对门），用于覆盖 fold_and_relink 候选路径
架构（2 种）：linear=True（生产默认）/ linear=False（fixpoint 消融）。

P1（默认选择路径）：对每条语料 × 每种架构，用 `_v5_pipeline`（固定
  恒等布局 + 路由 seed 0、relabel_budget=0.5）的**默认决胜规则**产物：
  (a) 语义等价：紧凑空间态矢量（|0⟩、|+⟩、2 个随机直积态，固定种子
      42）最差重叠 > 1-1e-8；初/终布局按 (layout, final_layout) 补偿；
  (b) 合法性：门集 ⊆ {rz,sx,x,cz}、2Q 全相邻、比特 < spec.n 且非禁用、
      无 swap 残留、measure 标签合法。

P2（候选路径覆盖）：同一语料 × 架构，强制 `_select` 返回候选列表的
  每个下标，交付产物同样通过 P1(a)(b)。结构覆盖断言：每条线路候选数
  >= 2；qaoa12 >= 3（全局重标记 + fold_keep + 裸路由）；synth_fold 的
  fold_and_relink 候选存在（arr != None）。

P3（特殊角模式）：synth_special_* 覆盖 rz(0)/rz(π/4)/rz(π/2)/rz(π)、
  H 块、sx 对、cz 对消、u3 特角——默认与全候选路径均须通过 P1。

失败含义：生产后处理/分解路径（候选选择 × 框架）出现语义或合法性
回归——审查代理曾捕获的候选污染、坐标帧、链归属类 bug 会在此门禁变红。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from qtrail.pure.circuit import Circuit, Inst

ROOT = None
PI = math.pi

SPECIAL = [
    ("synth_special_rz", 6),
    ("synth_special_h", 6),
    ("synth_fold", 9),
]
EXAMPLES = ["examples/qft5.qasm", "examples/ghz10.qasm",
            "examples/qaoa12.qasm"]
ARCHS = [("linear", True), ("fixpoint", False)]


def _load_root():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 语料
def _synth_special_rz(n=6):
    c = Circuit(n)
    c.append(Inst("u3", (0,), (PI / 2, 0.3, 0.7)))
    c.append(Inst("h", (0,)))
    c.append(Inst("rz", (1,), (PI / 4,)))
    c.append(Inst("cx", (0, 1)))
    c.append(Inst("rz", (0,), (PI / 2,)))
    c.append(Inst("sx", (0,)))
    c.append(Inst("rz", (2,), (PI,)))
    c.append(Inst("cz", (1, 2)))
    c.append(Inst("h", (2,)))
    c.append(Inst("cx", (2, 3)))
    c.append(Inst("rz", (3,), (0.0,)))
    c.append(Inst("sx", (3,)))
    c.append(Inst("cz", (3, 4)))
    c.append(Inst("u3", (4,), (PI, 0.1, 0.2)))
    c.append(Inst("u1", (5,), (PI,)))
    c.append(Inst("u3", (5,), (PI / 2, 0.4, 1.1)))
    c.append(Inst("u2", (3,), (0.5, 0.9)))
    c.append(Inst("cx", (4, 5)))
    c.append(Inst("rz", (4,), (PI,)))
    c.append(Inst("sx", (4,)))
    c.append(Inst("x", (5,)))
    return c


def _synth_special_h(n=6):
    c = Circuit(n)
    # H 块夹 rz（_absorb_h 路径）+ sx 对
    c.append(Inst("h", (0,)))
    c.append(Inst("rz", (0,), (PI,)))
    c.append(Inst("h", (0,)))
    c.append(Inst("sx", (1,)))
    c.append(Inst("rz", (1,), (PI,)))
    c.append(Inst("sx", (1,)))
    c.append(Inst("cz", (0, 1)))
    # cz 对消（同对偶次）
    c.append(Inst("cz", (2, 3)))
    c.append(Inst("h", (2,)))
    c.append(Inst("rz", (2,), (PI / 2,)))
    c.append(Inst("h", (2,)))
    c.append(Inst("cz", (2, 3)))
    # rz(π/4) 穿过 cz 后合并
    c.append(Inst("rz", (3,), (PI / 4,)))
    c.append(Inst("cz", (3, 4)))
    c.append(Inst("rz", (3,), (PI / 4,)))
    c.append(Inst("sx", (4,)))
    c.append(Inst("cx", (4, 5)))
    c.append(Inst("rz", (5,), (PI,)))
    c.append(Inst("x", (5,)))
    c.append(Inst("rz", (5,), (0.0,)))
    return c


def _synth_fold(spec, n=9, seed=0):
    """合成可被 fold 吸收的 routed 形态（与 T6 测试同构造）：
    不相交边上的 swap(a,b) + 同对 2Q 门；返回 (物理线路, 参考逻辑,
    最终布局 logical→physical)。"""
    rng = np.random.default_rng(6000 + seed)
    content = {p: p for p in range(n)}
    pos = {p: p for p in range(n)}
    edges = [(a, b) for a in range(n) for b in range(a + 1, n)
             if spec.adj[a, b]]
    rng.shuffle(edges)
    chosen, used = [], set()
    for a, b in edges:
        if a not in used and b not in used:
            chosen.append((a, b))
            used |= {a, b}
            if len(chosen) == 2:
                break
    phys, ref = Circuit(n), Circuit(n)
    for a, b in chosen:
        phys.append(Inst("swap", (a, b)))
        content[a], content[b] = content[b], content[a]
        pos[content[a]], pos[content[b]] = a, b
        for k in range(3):
            if k == 1:
                name = "cx" if rng.random() < 0.5 else "cz"
                u, v = content[a], content[b]
                phys.append(Inst(name, (a, b)))
                ref.append(Inst(name, (u, v)))
            else:
                c = int(rng.integers(0, n))
                name = "sx" if k == 0 else "rz"
                ps = () if name == "sx" else (0.7,)
                phys.append(Inst(name, (pos[c],), ps))
                ref.append(Inst(name, (c,), ps))
    return phys, ref, {l: p for p, l in content.items()}


def _corpus(spec):
    from qtrail.pure.qasm import load_qasm_file
    root = _load_root()
    out = {}
    for path in EXAMPLES:
        circ = load_qasm_file(root / path)
        out[path.split("/")[-1].replace(".qasm", "")] = (circ, None)
    for name, n in SPECIAL:
        if name == "synth_special_rz":
            out[name] = (_synth_special_rz(n), None)
        elif name == "synth_special_h":
            out[name] = (_synth_special_h(n), None)
        else:
            phys, ref, fin = _synth_fold(spec, n)
            out[name] = (ref, (phys, fin))
    return out


# ---------------------------------------------------------------- 模拟
def _gate_mat(name, params):
    if name == "rz":
        return np.diag([np.exp(-1j * params[0] / 2),
                        np.exp(1j * params[0] / 2)])
    if name == "sx":
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
    if name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    if name == "cz":
        m = np.eye(4, dtype=complex)
        m[3, 3] = -1
        return m
    if name == "cx":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
    if name == "u1":
        return np.diag([1.0, np.exp(1j * params[0])])
    if name == "u2":
        return _gate_mat("u3", (math.pi / 2, params[0], params[1]))
    if name == "u3":
        th, ph, la = params
        c, s = np.cos(th / 2), np.sin(th / 2)
        return np.array([[c, -np.exp(1j * la) * s],
                         [np.exp(1j * ph) * s,
                          np.exp(1j * (ph + la)) * c]], dtype=complex)
    raise ValueError(name)


def _apply(sv, mat, qs, n):
    qs = tuple(int(q) for q in qs)          # 保持门参数给出的顺序
    perm = [q for q in range(n) if q not in qs]
    new_perm = perm + list(qs)
    sv = np.transpose(sv.reshape([2] * n), new_perm).reshape(
        2 ** (n - len(qs)), 2 ** len(qs))
    sv = sv @ mat.T
    inv = np.argsort(new_perm)
    return np.transpose(sv.reshape([2] * n), inv).reshape(-1)


def _sim(circ, n, sv=None):
    if sv is None:
        sv = np.zeros(2 ** n, dtype=complex)
        sv[0] = 1
    for inst in circ.ops:
        sv = _apply(sv, _gate_mat(inst.name, inst.params), inst.qubits, n)
    return sv


def _random_product_state(rng, n):
    sv = np.ones(1, dtype=complex)
    for _ in range(n):
        th = rng.uniform(0, np.pi)
        ph = rng.uniform(0, 2 * np.pi)
        sv = np.kron(sv, np.array([np.cos(th / 2),
                                   np.sin(th / 2) * np.exp(1j * ph)]))
    return sv


def _embed(sv, pos, k):
    """逻辑 n 比特态 → 紧凑 k 比特态：逻辑 i 放到紧凑位 pos[i]。

    位序必须与 _apply 的大端约定一致（qubit i 是索引第 n-1-i 位）
    ——小端写法只在 |0⟩/|+⟩ 对称输入上巧合成立（2026-11-13 测试
    自身修复）。
    """
    n = len(pos)
    out = np.zeros(2 ** k, dtype=complex)
    for x in range(2 ** n):
        y = 0
        for i in range(n):
            if (x >> (n - 1 - i)) & 1:
                y |= 1 << (k - 1 - int(pos[i]))
        out[y] = sv[x]
    return out


# ---------------------------------------------------------------- 生产调用
def _prepare(case, spec):
    """返回 (circ, routed_post, final_layout, pre_folded, layout, synth)。

    synth=(phys, fin) 时直接用合成 routed 形态（fold 候选覆盖）；
    否则按生产顺序路由 + post_route + fold_keep 前置候选。
    """
    from qtrail.pure.post import post_route
    from qtrail.pure.post_v2 import fold_keep_coords
    from qtrail.pure.router import sabre_route

    circ, synth = case
    if synth is not None:
        phys, fin = synth
        pre = None
        return circ, phys, dict(fin), pre, {i: i for i in range(circ.n)}, True
    n = circ.n
    layout = {i: i for i in range(n)}
    routed, _swaps, fin = sabre_route(circ, spec, dict(layout), seed=0,
                                      trials=2)
    fk_c, fk_fin = fold_keep_coords(routed, fin, layout)
    pre = (fk_c, fk_fin) if all(
        i.nq != 2 or spec.adj[int(i.qubits[0]), int(i.qubits[1])]
        for i in fk_c.ops) else None
    routed2, _removed = post_route(routed, fin, spec.adj)
    return circ, routed2, fin, pre, layout, False


def _run(case_name, arch_linear, pick, spec):
    from qtrail.pure.mapper import PureMapper

    circ, routed, fin, pre, layout, _synth = _prepare(case_name, spec)
    mapper = PureMapper(spec, policy=None, pipeline="v5",
                        selection_rule="fidelity", routing_seeds=1, seed=0,
                        linear_decompose=arch_linear)
    seen = {}
    orig = mapper._select

    def select(scored):
        seen["n"] = len(scored)
        if pick is None:
            return orig(scored)
        entry = scored[min(int(pick), len(scored) - 1)]
        return entry[3], min(s[0] for s in scored)

    mapper._select = select
    try:
        _routed, final_layout, final_qc = mapper._v5_pipeline(
            routed, fin, circ, 0.0, relabel_budget=0.5,
            pre_folded=pre, layout=layout)
    finally:
        mapper._select = orig
    return seen.get("n", 0), final_qc, layout, final_layout


# ---------------------------------------------------------------- 判定
def _check_semantics(circ, qc, layout, final_layout, spec, tag):
    n = circ.n
    # (b) 合法性
    for inst in qc.ops:
        assert inst.name in ("rz", "sx", "x", "cz"), \
            f"{tag} 非法门 {inst.name}"
        for q in inst.qubits:
            q = int(q)
            assert 0 <= q < spec.n and q not in spec.disabled_qubits, \
                f"{tag} 比特非法 {q}"
        if inst.nq == 2:
            assert spec.adj[int(inst.qubits[0]), int(inst.qubits[1])], \
                f"{tag} 非相邻 2Q 门 {inst.qubits}"
    for m in qc.measures:
        q = int(m.qubits[0])
        assert 0 <= q < spec.n and q not in spec.disabled_qubits, \
            f"{tag} measure 比特非法 {q}"

    # 紧凑空间
    used = {int(q) for inst in qc.ops for q in inst.qubits}
    used |= {int(m.qubits[0]) for m in qc.measures}
    for i in range(n):
        used.add(int(layout[i]))
        used.add(int(final_layout.get(i, i)))
    used = sorted(used)
    p2c = {p: k for k, p in enumerate(used)}
    k = len(used)
    assert k <= 18, f"{tag} 紧凑空间 {k} 比特超测试预算"

    compact = Circuit(k)
    for inst in qc.ops:
        compact.append(Inst(inst.name,
                            tuple(p2c[int(q)] for q in inst.qubits),
                            inst.params))

    pos_init = [p2c[int(layout[i])] for i in range(n)]
    pos_fin = [p2c[int(final_layout.get(i, i))] for i in range(n)]
    rng = np.random.default_rng(42)
    inputs = [None, "plus"] + [rng for _ in range(2)]
    worst = 1.0
    for kind in inputs:
        if kind is None:
            sv_in = np.zeros(2 ** n, dtype=complex)
            sv_in[0] = 1
        elif kind == "plus":
            sv_in = np.full(2 ** n, 1.0 / 2 ** (n / 2), dtype=complex)
        else:
            sv_in = _random_product_state(kind, n)
        target = _embed(_sim(circ, n, sv_in), pos_fin, k)
        got = _sim(compact, k, _embed(sv_in, pos_init, k))
        ov = abs(np.vdot(target, got)) ** 2
        worst = min(worst, ov)
    assert worst > 1 - 1e-8, f"{tag} 语义回归 worst_overlap={worst}"


@pytest.fixture(scope="module")
def spec():
    from qtrail.config import load_device_config
    from qtrail.devices import build_tianyan287_spec
    return build_tianyan287_spec(load_device_config())


@pytest.fixture(scope="module")
def corpus(spec):
    return _corpus(spec)


@pytest.mark.parametrize("arch,linear", ARCHS)
@pytest.mark.parametrize("name", EXAMPLES + [s[0] for s in SPECIAL])
def test_production_default_semantics(corpus, spec, name, arch, linear):
    key = name.split("/")[-1].replace(".qasm", "") if "/" in name else name
    case = corpus[key]
    ncand, qc, layout, fin = _run(case, linear, None, spec)
    _check_semantics(case[0], qc, layout, fin, spec,
                     f"{key}/{arch}/default")


@pytest.mark.parametrize("arch,linear", ARCHS)
@pytest.mark.parametrize("name", EXAMPLES + [s[0] for s in SPECIAL])
def test_production_candidate_paths(corpus, spec, name, arch, linear):
    key = name.split("/")[-1].replace(".qasm", "") if "/" in name else name
    case = corpus[key]
    ncand, _qc, _lay, _fin = _run(case, linear, None, spec)
    assert ncand >= 2, f"{key}/{arch} 候选数 {ncand} < 2"
    if key == "qaoa12":
        assert ncand >= 3, f"{key}/{arch} 候选数 {ncand} < 3（覆盖回归）"
    if key == "synth_fold":
        from qtrail.pure.post_v2 import fold_and_relink
        assert case[1] is not None
        _f, arr = fold_and_relink(case[1][0], spec)
        assert arr is not None, "synth_fold 的 fold 候选消失（覆盖回归）"
    for pick in range(ncand):
        nc2, qc, layout, fin = _run(case, linear, pick, spec)
        assert nc2 == ncand
        _check_semantics(case[0], qc, layout, fin, spec,
                         f"{key}/{arch}/pick={pick}")
