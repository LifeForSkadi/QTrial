"""不变式 property-based 测试：T5 链调度顺序 / T6 fold 坐标帧 / T7 双射。

预注册判据（2026-11-13，实现前写定——见 docs/academic_review_gaps.md §5 项 1）

P1（T5 路由器链调度顺序）：
  - 生成器：3×3 网格（build_grid3x3_spec），9 逻辑比特（恒等布局），
    门集 {rz,sx,x,h,cx,cz}，每种子 25~45 门，固定种子 12 个；输入不含
    swap（判定器把输出 swap 一律视为路由交换）。
  - 判定器：沿物理轨迹（输出 swap 更新 phys→logical 映射）把每个输出
    门映射回逻辑比特；断言每个逻辑比特的门序列与输入逐门一致（门名 +
    参数），每门恰好发射一次——即"同比特相对顺序保持"。
  - 语义锚点：同一轨迹重建"恢复逻辑线路"（门序 = 输出序），断言其与
    输入逐态等价（4 个输入态，|ov|^2 > 1-1e-8）。
    注：不能用静态 final_layout 逆映射——原始 routed 电路的门标签是
    **发射时刻**物理坐标，静态映射只对 |0> 巧合成立（2026-11-13 实测
    fid=0.0024；动态内容追踪 fid=1）。
  - 判据：12/12 种子通过 + 固定回归用例（空依赖 1Q 门附着链的最小
    复现族）通过 + 4 输入态语义等价。
  - 失败含义：T5 被破坏（2026-11-12 P0 顺序反转一类的机制回归）。

P2（T6 fold 坐标帧）：
  - 采用路径生成器（合成 routed 形态）：随机 swap（物理边）+ 门只取
    "内容坐标相邻"对（fold 吸收守卫的结构条件）——采用率 12/12。
    以参考逻辑线路（同一门序列在内容坐标下的表示）为锚点，断言
    unmap(F, flb) 与参考在 4 个输入态上等价；全部 2Q 门相邻。
  - 放弃路径生成器（真实路由产物 12 条）：恒等布局下真实 routed 电路
    的 swap 后接内容坐标远距离门，采用率 0/60（结构事实）——断言
    返回对象即输入且输入 ops 逐门未变（放弃折叠 = 无副作用，不得
    "保留 swap + 更新 arr"）。
  - 判据：采用样本 >= 3、放弃样本 >= 6，0 失败。
  - 失败含义：坐标帧约定被破坏（2026-11-12 P0 双重应用一类的机制回归）。

P3（T7 重标记双射，C1）：
  - 生成器：天衍-287 真机拓扑/校准 4 个种子；BFS 取 9 点连通片，
    线路直接构造在最终坐标（无 swap，门标签 = fin 值），随机 fin
    双射；片内取一个 2Q 孤立位 X——它只有 1Q/measure/fin 占用
    （C1 occupied 全集、used 真子集的 P0 场景）。
  - 判据：移动后 final_layout 值两两不同；全部门/measure 标签 ⊆ fin
    值集；2Q 门全部相邻；unmap 后态矢量与移动前等价；log 保真度不
    劣化；4 种子上总移动数 >= 1。
  - 失败含义：C1 双射/合法性/单调性被破坏（2026-11-12 P0 撞车一类）。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from qtrail.pure.circuit import Circuit, Inst

ONE_Q = ("rz", "sx", "x", "h")


# ---------------------------------------------------------------- 随机生成
def _random_circuit(rng, n, n_gates, p_two=0.45):
    circ = Circuit(n)
    for _ in range(n_gates):
        if rng.random() < p_two:
            a = int(rng.integers(0, n))
            b = int(rng.integers(0, n - 1))
            if b >= a:
                b += 1
            name = "cx" if rng.random() < 0.5 else "cz"
            circ.append(Inst(name, (a, b)))
        else:
            q = int(rng.integers(0, n))
            name = ONE_Q[int(rng.integers(0, len(ONE_Q)))]
            if name == "rz":
                circ.append(Inst("rz", (q,),
                                 (float(rng.uniform(0.0, 2 * math.pi)),)))
            else:
                circ.append(Inst(name, (q,)))
    return circ


# ---------------------------------------------------------------- 模拟工具
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
    if name == "swap":
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
    raise ValueError(name)


def _apply(sv, mat, qs, n):
    # 保持门参数给出的比特顺序（cx 控制在前）：sorted 会把反向 cx
    # （控制位下标 > 目标位）的矩阵约定弄反（2026-11-13 测试自身修复）。
    qs = tuple(int(q) for q in qs)
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


def _unmap(circ, fin, n_logical):
    """物理空间线路逆映射回逻辑空间（与 test_pure_v2 同口径）。"""
    inv = {int(v): int(k) for k, v in fin.items() if int(k) < n_logical}
    out = Circuit(n_logical)
    for inst in circ.ops:
        qs = [inv[int(q)] for q in inst.qubits if int(q) in inv]
        if len(qs) == inst.nq:
            out.append(Inst(inst.name, tuple(qs), inst.params))
    return out


def _ops_sig(circ):
    return [(i.name, tuple(int(q) for q in i.qubits),
             tuple(float(p) for p in i.params)) for i in circ.ops]


# ---------------------------------------------------------------- T5 判定器
def _check_router_order(circ, routed, layout, spec_n):
    """逐逻辑比特门序判定：输出映射回逻辑空间后与输入逐门一致。

    返回 (恢复逻辑线路, 内容末态映射)。输入不得含 swap（输出 swap
    全部视为路由插入的物理交换）。
    """
    assert all(i.name != "swap" for i in circ.ops), "判定器要求输入无 swap"
    per_q = {q: [] for q in range(circ.n)}
    for k, inst in enumerate(circ.ops):
        for q in inst.qubits:
            per_q[int(q)].append(k)
    content = {p: -1 for p in range(spec_n)}
    for l, p in layout.items():
        content[int(p)] = int(l)
    rec = Circuit(circ.n)
    nxt = {q: 0 for q in range(circ.n)}
    seen = [0] * len(circ.ops)
    for inst in routed.ops:
        qs = [int(q) for q in inst.qubits]
        if inst.name == "swap":
            a, b = qs
            content[a], content[b] = content[b], content[a]
            continue
        logs = [content[p] for p in qs]
        assert all(l >= 0 for l in logs), (
            f"门作用在未映射的物理位 {tuple(qs)}")
        rec.append(Inst(inst.name, tuple(logs), inst.params))
        idxs = []
        for q in logs:
            i = nxt[q]
            assert i < len(per_q[q]), (
                f"比特 {q}：输出多出门 {inst.name}{tuple(qs)}")
            k = per_q[q][i]
            exp = circ.ops[k]
            assert exp.name == inst.name and \
                len(exp.params) == len(inst.params) and \
                np.allclose(np.asarray(exp.params, dtype=float),
                            np.asarray(inst.params, dtype=float)), (
                f"比特 {q}：门序/门参数不符，期望 #{k} {exp.name}"
                f"{exp.params}，实际 {inst.name}{inst.params}")
            idxs.append(k)
        assert len(set(idxs)) == 1, (
            f"顺序反转：{inst.name}{tuple(qs)} 在两个比特上的"
            f"期望门 id 不一致 {idxs}")
        seen[idxs[0]] += 1
        for q in logs:
            nxt[q] += 1
    assert all(c == 1 for c in seen), (
        f"存在未发射/重复发射的门：{[k for k, c in enumerate(seen) if c != 1]}")
    return rec, content


def _synth_routed(rng, spec, n, n_pairs=2, ops_per_pair=3):
    """合成可被 fold 吸收的 routed 形态（守卫条件的最小结构）。

    fold 吸收守卫（fold_and_relink 内）：对 swap(a,b) 之后每个触碰
    a/b 的 2Q 门，按 arr 调整（a↔b）后仍须相邻。可构造满足它的最小
    形态 = 取互不相交的物理边 (a,b)，每条边上交替插入 swap(a,b)
    与同物理对 (a,b) 上的 2Q 门——内容 a/b 始终留在该对位置，
    调整后恒相邻；1Q 门可任意。
    返回 (物理坐标线路, 参考逻辑线路, 内容映射)。
    """
    content = {p: p for p in range(n)}
    pos = {p: p for p in range(n)}          # 内容 → 当前物理位
    edges = [(a, b) for a in range(n) for b in range(a + 1, n)
             if spec.adj[a, b]]
    rng.shuffle(edges)
    chosen, used = [], set()
    for a, b in edges:
        if a not in used and b not in used:
            chosen.append((a, b))
            used |= {a, b}
            if len(chosen) == n_pairs:
                break
    phys = Circuit(n)
    ref = Circuit(n)
    for a, b in chosen:
        # 每条边先保证一次 swap（采用路径样本不得退化为零 swap 早退）
        phys.append(Inst("swap", (a, b)))
        content[a], content[b] = content[b], content[a]
        pos[content[a]], pos[content[b]] = a, b
        for _ in range(ops_per_pair):
            if rng.random() < 0.5:
                c = int(rng.integers(0, n))
                name = ONE_Q[int(rng.integers(0, len(ONE_Q)))]
                ps = ((float(rng.uniform(0.0, 2 * math.pi)),)
                      if name == "rz" else ())
                phys.append(Inst(name, (pos[c],), ps))
                ref.append(Inst(name, (c,), ps))
                continue
            if rng.random() < 0.3:
                phys.append(Inst("swap", (a, b)))
                content[a], content[b] = content[b], content[a]
                pos[content[a]], pos[content[b]] = a, b
            else:
                name = "cx" if rng.random() < 0.5 else "cz"
                u, v = content[a], content[b]
                phys.append(Inst(name, (a, b)))
                ref.append(Inst(name, (u, v)))
    return phys, ref, content


# ---------------------------------------------------------------- P1（T5）
def test_router_schedule_order_property():
    from qtrail.devices import build_grid3x3_spec
    from qtrail.pure.router import sabre_route

    spec = build_grid3x3_spec(0)
    n = 9
    layout = {i: i for i in range(n)}
    for seed in range(12):
        rng = np.random.default_rng(1000 + seed)
        circ = _random_circuit(rng, n, int(rng.integers(25, 46)))
        routed, _swaps, fin = sabre_route(circ, spec, dict(layout), seed=seed)
        rec, content = _check_router_order(circ, routed, layout, spec.n)
        # 内容末态必须等于路由器返回的 final_layout（坐标帧自洽）
        fin2 = {l: p for p, l in content.items() if l >= 0}
        assert fin2 == {int(k): int(v) for k, v in fin.items()}, (
            f"seed={seed} 内容末态与 final_layout 不一致")
        # 语义锚点：动态恢复的逻辑线路与输入逐态等价
        rng2 = np.random.default_rng(3000 + seed)
        for trial in range(4):
            sv0 = None if trial == 0 else _random_product_state(rng2, n)
            sv_in = _sim(circ, n, sv0)
            sv_out = _sim(rec, n, sv0)
            fid = abs(np.vdot(sv_in, sv_out)) ** 2
            assert fid > 1 - 1e-8, (
                f"seed={seed} trial={trial} 路由语义不等价 fid={fid}")


def test_router_empty_dep_1q_regression():
    """P0 最小复现族：链区间内空依赖 1Q 门不得被延迟到链执行。

    修复前 cz(0,2) 链因 0/2 非相邻被路由阻塞，rz(1)（空依赖）挂在
    链上延迟发射，sx(1)（空依赖、链区间外）立即发射 → 顺序反转。
    """
    from qtrail.devices import build_grid3x3_spec
    from qtrail.pure.router import sabre_route

    spec = build_grid3x3_spec(0)
    n = 3
    layout = {i: i for i in range(n)}
    circ = Circuit(n)
    circ.cz(0, 2)
    circ.append(Inst("rz", (1,), (0.5,)))
    circ.append(Inst("sx", (0,)))
    circ.append(Inst("sx", (2,)))
    circ.cz(0, 2)
    circ.append(Inst("sx", (1,)))
    routed, _swaps, _fin = sabre_route(circ, spec, dict(layout), seed=0)
    rec, _content = _check_router_order(circ, routed, layout, spec.n)
    # 回归语义锚点
    sv_in = _sim(circ, n, None)
    sv_out = _sim(rec, n, None)
    assert abs(np.vdot(sv_in, sv_out)) ** 2 > 1 - 1e-8


# ---------------------------------------------------------------- P2（T6）
def test_fold_coordinate_frame_property():
    from qtrail.devices import build_grid3x3_spec
    from qtrail.pure.post_v2 import fold_and_relink

    spec = build_grid3x3_spec(0)
    n = 9
    adopted = 0
    for seed in range(12):
        rng = np.random.default_rng(6000 + seed)
        phys, ref, content = _synth_routed(
            rng, spec, n, n_pairs=int(rng.integers(2, 4)),
            ops_per_pair=int(rng.integers(3, 6)))
        folded, arr = fold_and_relink(phys, spec)
        assert arr is not None, (
            f"seed={seed} 合成 routed（内容相邻门）应可被吸收")
        adopted += 1
        # 候选 final_layout：flb[k] = arr[fin[k]]（fin = 内容末态映射）
        fin = {l: p for p, l in content.items()}
        flb = {k: int(arr[int(v)]) for k, v in fin.items()}
        u_f = _unmap(folded, flb, n)
        rng2 = np.random.default_rng(8000 + seed)
        for trial in range(4):
            sv0 = None if trial == 0 else _random_product_state(rng2, n)
            sv_f = _sim(u_f, n, sv0)
            sv_r = _sim(ref, n, sv0)
            fid = abs(np.vdot(sv_r, sv_f)) ** 2
            assert fid > 1 - 1e-8, (
                f"seed={seed} trial={trial} fold 坐标帧不等价 fid={fid}")
        for inst in folded.ops:
            if inst.nq == 2:
                assert spec.adj[int(inst.qubits[0]),
                                int(inst.qubits[1])], (
                    f"seed={seed} fold 产物含非相邻 2Q 门 {inst.qubits}")
    assert adopted >= 3, f"采用路径样本不足（{adopted}）——生成器须排查"


def test_fold_abandon_no_side_effects():
    """放弃路径：真实 routed 产物（恒等布局）结构上不可吸收——
    必须返回输入本体且零修改（不得"保留 swap + 更新 arr"）。"""
    from qtrail.devices import build_grid3x3_spec
    from qtrail.pure.post_v2 import fold_and_relink
    from qtrail.pure.router import sabre_route

    spec = build_grid3x3_spec(0)
    n = 9
    layout = {i: i for i in range(n)}
    abandoned = 0
    for seed in range(12):
        rng = np.random.default_rng(1000 + seed)
        circ = _random_circuit(rng, n, int(rng.integers(25, 45)))
        routed, swaps, _fin = sabre_route(circ, spec, dict(layout), seed=seed)
        assert swaps > 0
        sig_before = _ops_sig(routed)
        folded, arr = fold_and_relink(routed, spec)
        if arr is not None:
            continue
        abandoned += 1
        assert folded is routed, "放弃折叠必须返回输入本体"
        assert _ops_sig(routed) == sig_before, "放弃折叠不得修改输入"
    assert abandoned >= 6, (
        f"放弃路径样本不足（{abandoned}）——结构条件变化须排查")


# ---------------------------------------------------------------- P3（T7）
def test_relabel_bijection_property():
    from qtrail.devices import build_tianyan287_real_spec
    from qtrail.pure.metrics import estimate_log_fidelity
    from qtrail.pure.post_v2 import noise_aware_relabel
    from qtrail.pure.router import sabre_route

    spec = build_tianyan287_real_spec()  # 真机校准（合成校准方差不足→0 移动）
    n = 9
    # 从设备图 BFS 取 9 点连通片（真机拓扑非规则索引网格）
    disabled = set(spec.disabled_qubits)
    patch, seen = [], {0}
    stack = [0]
    while stack and len(patch) < n:
        v = stack.pop(0)
        patch.append(v)
        for u in np.nonzero(spec.adj[v])[0]:
            u = int(u)
            if u not in seen and u not in disabled:
                seen.add(u)
                stack.append(u)
    assert len(patch) == n, f"连通片不足 9 点：{patch}"
    x_site = patch[-1]  # 2Q 孤立位（只有 1Q/measure/fin 占用）
    edges = [(a, b) for a in patch for b in patch
             if a < b and spec.adj[a, b] and x_site not in (a, b)]
    assert edges, "连通片内 2Q 边为空"
    total_moved = 0
    for seed in range(4):
        rng = np.random.default_rng(5000 + seed)
        base = Circuit(spec.n)  # 物理坐标线路（无 swap，标签即最终坐标）
        for _ in range(30):
            if rng.random() < 0.45:
                a, b = edges[int(rng.integers(0, len(edges)))]
                base.append(Inst("cx" if rng.random() < 0.5 else "cz",
                                 (int(a), int(b))))
            else:
                q = int(patch[int(rng.integers(0, len(patch)))])
                name = ONE_Q[int(rng.integers(0, len(ONE_Q)))]
                ps = ((float(rng.uniform(0.0, 2 * math.pi)),)
                      if name == "rz" else ())
                base.append(Inst(name, (q,), ps))
        routed = base.copy()
        perm = rng.permutation(len(patch))
        fin = {i: int(patch[int(perm[i])]) for i in range(n)}
        for i in range(n):
            routed.measures.append(Inst("measure", (int(fin[i]),),
                                        cbits=(i,)))
        fin_before = dict(fin)
        u_before = _unmap(routed.copy(), fin_before, n)
        logf_before = estimate_log_fidelity(routed, spec.calib)
        total_moved += noise_aware_relabel(routed, spec, fin)
        # (a) 双射：final_layout 值两两不同
        vals = [int(v) for v in fin.values()]
        assert len(set(vals)) == len(vals), (
            f"seed={seed} 重标记破坏双射 fin={fin}")
        # (b) 全部门/measure 标签 ⊆ fin 值集；2Q 相邻
        vset = set(vals)
        for inst in routed.ops:
            assert all(int(q) in vset for q in inst.qubits), (
                f"seed={seed} 门标签越出 fin 值集 {inst.qubits}")
            if inst.nq == 2:
                assert spec.adj[int(inst.qubits[0]),
                                int(inst.qubits[1])], (
                    f"seed={seed} 重标记产生非法 2Q 门 {inst.qubits}")
        for m in routed.measures:
            assert int(m.qubits[0]) in vset, (
                f"seed={seed} measure 未随重标记同步 {m.qubits}")
        # (c) 语义等价：unmap 后与移动前逐态一致
        u_after = _unmap(routed, fin, n)
        sv_b = _sim(u_before, n)
        sv_a = _sim(u_after, n)
        fid = abs(np.vdot(sv_b, sv_a)) ** 2
        assert fid > 1 - 1e-8, (
            f"seed={seed} 重标记破坏语义 fid={fid}")
        # (d) 保真度不劣化
        logf_after = estimate_log_fidelity(routed, spec.calib)
        assert logf_after >= logf_before - 1e-12, (
            f"seed={seed} 重标记劣化 {logf_before} -> {logf_after}")
    assert total_moved >= 1, "扫描未触发任何移动——生成器/校准须排查"
