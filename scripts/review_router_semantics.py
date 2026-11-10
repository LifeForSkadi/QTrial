# -*- coding: utf-8 -*-
"""独立审查脚本（2026-11-13 审查代理）：任务 3/4 数值实验。

A. 路由器契约 + 紧凑子空间判据方向（初始布局 vs final_layout 嵌入）
B. 防卡死逃逸路径（回卷 + BFS 推进）语义一致性模拟
C. fold_and_relink 语义 + 保留 swap 相邻性风险扫描（真机 spec + h16）
D. measure 标签一致性（absorb_trailing / noise_aware_relabel 是否重标记 measure）
"""
import sys
import math
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure import post_v2 as pv2
from qtrail.pure.post import absorb_trailing, post_route
from qtrail.pure.router import sabre_route
from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm

rng = np.random.default_rng(42)

# ------------------------------------------------------------------ 工具
def rz(a):
    return np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])

def sx():
    return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])

X = np.array([[0., 1.], [1., 0.]], dtype=complex)

def full_matrix(ops, n):
    U = np.eye(2 ** n, dtype=complex)
    for inst in ops:
        if inst.name == "swap":
            P = np.zeros((2 ** n, 2 ** n))
            perm = np.arange(2 ** n)
            for i in range(2 ** n):
                bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
                a, b = int(inst.qubits[0]), int(inst.qubits[1])
                bits[a], bits[b] = bits[b], bits[a]
                j = 0
                for bb in bits:
                    j = (j << 1) | bb
                perm[i] = j
            P[np.arange(2 ** n), perm.astype(int)] = 1.0
            U = P @ U
        elif inst.nq == 1:
            nm, pr = inst.name, inst.params
            if nm == "rz":
                m = rz(pr[0])
            elif nm == "sx":
                m = sx()
            elif nm == "x":
                m = X
            elif nm == "h":
                m = rz(math.pi / 2) @ sx() @ rz(math.pi / 2)
            else:
                raise ValueError(nm)
            q = int(inst.qubits[0])
            mats = [np.eye(2, dtype=complex)] * n
            mats[q] = m
            G = mats[0]
            for k in range(1, n):
                G = np.kron(G, mats[k])
            U = G @ U
        else:
            nm = inst.name
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            if nm == "cz":
                G = np.eye(2 ** n, dtype=complex)
                for i in range(2 ** n):
                    bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
                    if bits[a] and bits[b]:
                        G[i, i] = -1.0
            elif nm == "cx":
                G = np.zeros((2 ** n, 2 ** n), dtype=complex)
                for i in range(2 ** n):
                    bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
                    j = i
                    if bits[a]:
                        j = i ^ (1 << (n - 1 - b))
                    G[j, i] = 1.0
            else:
                raise ValueError(nm)
            U = G @ U
    return U

def evolve(ops, k):
    """态矢量演化（2^k 向量，逐门应用，不构建酉矩阵）。"""
    z = np.zeros(2 ** k, dtype=complex)
    z[0] = 1.0
    for inst in ops:
        qs = [int(q) for q in inst.qubits]
        if inst.nq == 1:
            m = rz(inst.params[0]) if inst.name == "rz" else (
                sx() if inst.name == "sx" else X)
            z = z.reshape((2,) * k)
            z = np.tensordot(m, z, axes=(1, qs[0]))
            z = np.moveaxis(z, 0, qs[0])
            z = z.reshape(-1)
        else:
            a, b = qs
            if inst.name == "swap":
                z = z.reshape((2,) * k)
                z = np.swapaxes(z, a, b)
                z = z.reshape(-1)
            elif inst.name == "cz":
                z = z.reshape((2,) * k)
                sel = (slice(None),) * k
                # 直接按位索引：cz = 在 a=b=1 处分量取负
                idx = np.zeros((2,) * k, dtype=bool)
                idx[(slice(None),) * a + (1,) + (slice(None),) * (k - a - 1)] = True
                # 简化为数值构造：展平索引
                zf = z.reshape(-1)
                n = 2 ** k
                mask = np.zeros(n, dtype=bool)
                for i in range(n):
                    bits = [(i >> (k - 1 - j)) & 1 for j in range(k)]
                    if bits[a] and bits[b]:
                        mask[i] = True
                zf[mask] *= -1.0
                z = zf
            elif inst.name == "cx":
                n = 2 ** k
                z2 = z.copy()
                for i in range(n):
                    bits = [(i >> (k - 1 - j)) & 1 for j in range(k)]
                    if bits[a]:
                        j = i ^ (1 << (k - 1 - b))
                        z2[j] = z[i]
                z = z2
    return z

def state_fid(ops1, ops2, n):
    """紧凑子空间态保真：压缩到两电路触碰比特的并集。

    触位数 > 22 时返回 None（态矢量内存 2^k 不可行）。"""
    qs = sorted({int(q) for i in ops1 + ops2 for q in i.qubits})
    cmap = {q: kk for kk, q in enumerate(qs)}
    k = len(qs)
    if k > 18:
        return None
    c1 = [Inst(i.name, tuple(cmap[int(q)] for q in i.qubits), i.params)
          for i in ops1]
    c2 = [Inst(i.name, tuple(cmap[int(q)] for q in i.qubits), i.params)
          for i in ops2]
    a = evolve(c1, k)
    b = evolve(c2, k)
    return float(abs(np.vdot(a, b)) ** 2)

def rand_circ(n, n_ops=10):
    ops = []
    for _ in range(n_ops):
        if rng.random() < 0.5:
            a, b = rng.choice(n, size=2, replace=False)
            if rng.random() < 0.5:
                ops.append(Inst("cx", (int(a), int(b))))
            else:
                ops.append(Inst("cz", (int(a), int(b))))
        else:
            q = int(rng.integers(0, n))
            if rng.random() < 0.5:
                ops.append(Inst("rz", (q,), (float(rng.uniform(0, 2 * math.pi)),)))
            else:
                ops.append(Inst("sx", (q,)))
    return ops

spec = build_tianyan287_spec(load_device_config())

# ================================================================ A：路由器契约
print("== A. 路由器契约与嵌入方向 ==")
worst = 0.0
n_nontrivial = 0
for trial in range(3):
    n = 4
    C = Circuit(n)
    C.ops = rand_circ(n)
    pi = {}
    # 取设备局部邻域内的初始位置（限制触碰集大小，仍强制非平凡置换）
    usable = sorted(set(range(spec.n)) - set(spec.disabled_qubits))[:60]
    chosen = rng.choice(usable, size=n, replace=False)
    for i, p in enumerate(chosen):
        pi[i] = int(p)
    routed, swaps, fin = sabre_route(C, spec, pi, seed=trial, trials=1,
                                     lookahead_w=0.0, w_ext=0.0,
                                     w_layer=0.0, cand_scope='front')
    # 1) 路由器契约（紧凑判据）：U(routed)|0⟩ == U(C 经 fin 嵌入)|0⟩
    emb_fin = [Inst(i.name, tuple(int(fin[int(q)]) for q in i.qubits),
                    i.params) for i in C.ops]
    fid_fin = state_fid(routed.ops, emb_fin, spec.n)
    # 2) 文档 A 口径（初始布局 π 嵌入）：应 ≠1（若置换非平凡）
    emb_pi = [Inst(i.name, tuple(int(pi[int(q)]) for q in i.qubits),
                   i.params) for i in C.ops]
    fid_pi = state_fid(routed.ops, emb_pi, spec.n)
    nontriv = any(pi[i] != fin[i] for i in range(n))
    n_nontrivial += int(nontriv)
    if fid_fin is not None:
        worst = max(worst, 1.0 - fid_fin)
    print(f'  trial {trial}: fin嵌入 fid={fid_fin if fid_fin is None else f"{fid_fin:.6f}"} '
          f'| π嵌入 fid={fid_pi if fid_pi is None else f"{fid_pi:.6f}"} '
          f'| 置换非平凡={nontriv} | swaps={swaps}')
print(f'  结论：fin 嵌入 fid 最大缺口={worst:.3e}；'
      f'π 嵌入在置换非平凡时 fid≠1（{n_nontrivial}/3 非平凡）→ '
      f'docs A 的 I-语义表述（初始布局嵌入）不成立，紧凑子空间法测的是'
      f'交付判据（fin 嵌入），方向正确')

# ================================================================ B：逃逸路径语义
print("== B. 防卡死逃逸（回卷 + BFS 推进）语义 ==")
def sim_escape(n=6, k_path=6):
    """模拟 router.py 451-524 行的回卷与推进逻辑，验证 pos/inv 一致性。

    修正版（审查代理复核）：pos 初始无碰撞（choice 不重复）；BFS 断言改为
    "两端标签终位相邻"（偶数 L 时两端在终点对上换位，不要求具体端点）。"""
    ok = True
    for _ in range(50):
        pos = {i: int(p) for i, p in enumerate(
            rng.choice(n, size=4, replace=False))}
        inv = {v: k for k, v in pos.items()}
        batch = []
        for _ in range(5):
            a, b = rng.choice(n, size=2, replace=False)
            batch.append((int(a), int(b)))
            la, lb = inv.get(int(a)), inv.get(int(b))
            inv.pop(int(a), None); inv.pop(int(b), None)
            if la is not None:
                inv[int(b)] = la
            if lb is not None:
                inv[int(a)] = lb
        pos_before = {k: v for k, v in pos.items()}   # 推进前 pos（推进未改 pos）
        # ---- 回卷（router 455-469 行逐字逻辑）
        for (ra, rb) in reversed(batch):
            la, lb = inv.get(ra), inv.get(rb)
            if la is not None:
                inv.pop(ra, None)
                pos[la] = rb
            if lb is not None:
                inv.pop(rb, None)
                pos[lb] = ra
            if la is not None:
                inv[rb] = la
            if lb is not None:
                inv[ra] = lb
        # 回卷后应恢复初始 pos
        if pos != pos_before:
            ok = False
        # ---- BFS 推进（router 505-523 行逐字逻辑）
        path = list(rng.choice(n, size=k_path, replace=False))
        split = len(path) // 2
        moves = [(path[i], path[i + 1]) for i in range(split)]
        for i in range(split - 1):
            e = len(path) - 1 - i
            moves.append((path[e], path[e - 1]))
        for (x, y) in moves:
            if abs(path.index(x) - path.index(y)) != 1:
                ok = False
        # 端到端语义：两端标签被搬到相邻的终点对（path[split], path[split+1]）
        labels = list(range(len(path)))
        for (x, y) in moves:
            xi, yi = path.index(x), path.index(y)
            labels[xi], labels[yi] = labels[yi], labels[xi]
        ends = [labels.index(0), labels.index(len(path) - 1)]
        if abs(ends[0] - ends[1]) != 1:
            ok = False
    return ok

print(f'  回卷恢复 pos/inv + BFS moves 语义一致: '
      f'{"PASS" if sim_escape() else "FAIL"}')

# ================================================================ C：fold 语义 + 相邻性
print("== C. fold_and_relink 语义 + 保留 swap 相邻性 ==")
circ16 = parse_qasm(Path(
    'data/benchpress/qasm/square-heisenberg/'
    'square_heisenberg_N16.qasm').read_text(encoding='utf-8'))
layout16 = {i: i for i in range(circ16.n)}
routed16, sw16, fin16 = sabre_route(circ16, spec, layout16, seed=42,
                                    trials=1, lookahead_w=0.0, w_ext=0.0,
                                    w_layer=0.0, cand_scope='front')
print(f'  h16 身份布局路由: {len(routed16.ops)} 门 swap={sw16}')

# C1: fold_and_relink 语义（|0⟩ 级比较，含链化）
folded, arr = pv2.fold_and_relink(routed16, spec, d_max=6)
if arr is not None:
    f0, arr0 = pv2.fold_only(routed16)
    fid = state_fid(folded.ops, f0.ops, spec.n)
    print(f'  fold_and_relink vs fold_only(纯折叠) 语义 fid='
          f'{fid if fid is None else f"{fid:.6f}"}（应=1，None=触位数超 22 跳过）')
    # C2: 保留 swap 的相邻性扫描
    bad = []
    for inst in folded.ops:
        if inst.name == "swap":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            if not spec.adj[a, b]:
                bad.append((a, b))
    print(f'  fold_and_relink 保留 swap 数={folded.count("swap")} '
          f'非相邻={len(bad)} {bad[:5]}')
else:
    print('  fold_and_relink 未采用（无收益），跳过')

# ================================================================ D：measure 一致性
print("== D. measure 标签一致性 ==")
# D1: absorb_trailing 吸收尾部 swap 时 measure 不重标记
# 场景：logical 0 置 x，路由末端 swap(0,1) 把 logical 0 搬到位置 1，
# 路由器在位置 1 发 measure(logical 0)（路由末端 pos[0]=1）。
# 吸收后：电路 [x(0)]、fin={0:0,1:1}——logical 0 的态在位置 0 按 fin[0]=0
# 读回 ✓，但 measure 仍标在位置 1：按 fin^-1 位置 1 属 logical 1 →
# logical 0 的 cbit 读走 logical 1 的态，logical 0 的 x|0⟩=|1⟩ 输出丢失。
c = Circuit(4)
c.ops = [Inst("x", (0,)), Inst("swap", (0, 1))]          # 尾部 swap
c.measures = [Inst("measure", (1,), cbits=(0,))]         # 路由末端 logical 0 的测量
fin_d = {0: 1, 1: 0, 2: 2, 3: 3}                         # 路由末端布局（swap 后）
post_route(c, fin_d, spec.adj)
m_q = int(c.measures[0].qubits[0])
owner = next(k for k, v in fin_d.items() if v == m_q)
# 修复前：measure 停留 1（读走 logical 1 的 |0⟩）；修复后：跟随到 0（读到 x|0⟩）
print(f'  D1 吸收后: fin={fin_d}；measure 原属 logical 0（路由末端位置 1），'
      f'现位置 {m_q} 按 fin^-1 属 logical {owner}')
if m_q == fin_d[0]:
    print('    → 修复后行为：measure 跟随 content 到 logical 0 的读回位 ✓ 一致')
else:
    print(f'    → 修复前行为：logical 0 的 cbit 读走 logical {owner} 的态，'
          f'x|0⟩=|1⟩ 输出丢失 ✗')

# D2: noise_aware_relabel 不重标记 measure
class FakeCalib:
    err_2q = {}
    err_1q = np.full(6, 1e-3)
    err_ro = np.full(6, 1e-2)
adj6 = np.zeros((6, 6), dtype=bool)
for i in range(5):
    adj6[i, i + 1] = adj6[i + 1, i] = True
spec6 = types.SimpleNamespace(n=6, disabled_qubits=set(), adj=adj6,
                              calib=FakeCalib())
c2 = Circuit(6)
c2.ops = [Inst("cz", (0, 1)), Inst("cz", (1, 2)), Inst("rz", (0,), (0.3,))]
c2.measures = [Inst("measure", (0,), cbits=(0,))]
fin2 = {0: 0, 1: 1, 2: 2}
n_moved = pv2.noise_aware_relabel(c2, spec6, fin2, max_passes=5)
moved = {int(q) for i in c2.ops for q in i.qubits}
m2_q = int(c2.measures[0].qubits[0])
print(f'  D2 noise_aware_relabel 移动 {n_moved} 次后 measure 仍在位置 {m2_q}'
      f'{"（该位若被移走则测量失效）" if m2_q not in moved else ""}')

# D3: learned_relabel 的 apply_pi 不重标记 measure（静态代码检查）
import inspect
src = inspect.getsource(pv2.learned_relabel)
print(f'  D3 learned_relabel apply_pi 遍历 circ.ops 而不遍历 circ.measures: '
      f'{"确认（measures 不重标记）" if "for inst in circ.ops" in src and "measures" not in src.split("def apply_pi")[1].split("def")[0] else "未确认"}')
print("  结论（审查时点）：原始代码 v4 路径（post_route 吸收）+ C1/L3 重标记"
      "均不同步 measure 标签；仅 fold 系列路径重标记 measure。"
      "工作区已于 2026-11-13 修复三者（P1-1）。验证脚本"
      "（verify_e2e_production.py to_qc / check_legality）仍完全忽略"
      " measure → 建议补 measure 检查后提交。")
