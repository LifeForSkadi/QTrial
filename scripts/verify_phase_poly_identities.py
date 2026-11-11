# -*- coding: utf-8 -*-
"""相位多项式机制（CZ/CNOT 域）候选恒等式数值验证（2026-11-13）。

验证后通过的恒等式才能进入设计文档与实现。失败项 = 机制设计错误。
门集：{rz, sx, x, cz} 平台 + H 块 + CNOT（虚拟域）。
第一版教训（已修正）：H 穿 cz 骨架 → CNOT（无 cz 域封闭行走）；
kron 比特顺序 = 高位在左；x ≠ rz·x·rz。
"""
import sys
import numpy as np

sys.path.insert(0, ".")

PI = np.pi


def rz(a):
    return np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])


def sx():
    return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])


def x():
    return np.array([[0.0, 1.0], [1.0, 0.0]])


def h():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def cz():
    return np.diag([1.0, 1.0, 1.0, -1.0])


def cx01():
    """CNOT（控制 0 目标 1）= |0><0|⊗I + |1><1|⊗X。"""
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                     [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def cx10():
    """CNOT（控制 1 目标 0）。"""
    return np.array([[1, 0, 0, 0], [0, 0, 0, 1],
                     [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)


def rzz(th):
    """RZZ_01(θ) = e^{−iθ Z_0 Z_1 / 2}。"""
    return np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2),
                    np.exp(1j * th / 2), np.exp(-1j * th / 2)])


def kron(q0, m1):
    """m1 作用在 q0 上（q0=0 → 左侧因子），2 比特空间。"""
    return (np.kron(m1, np.eye(2)) if q0 == 0
            else np.kron(np.eye(2), m1))


def close(A, B, tol=1e-10):
    """酉等价（模全局相位）检查。"""
    if A.shape != B.shape:
        return False
    ov = np.vdot(A.flatten(), B.flatten())
    return abs(abs(ov) / (np.linalg.norm(A) * np.linalg.norm(B)) - 1.0) < tol


results = []


def check(name, lhs, rhs, tol=1e-10):
    ok = close(lhs, rhs, tol)
    results.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


th, al, be = 0.7, 1.3, -0.9

# ============ A 组：平台基基本恒等式（第一版已过，复验） ============
check("A1  rz(a)·rz(b) = rz(a+b)",
      rz(al) @ rz(be), rz(al + be))
check("A2  rz_q 与 cz 对易（q=0 侧）",
      kron(0, rz(th)) @ cz(), cz() @ kron(0, rz(th)))
check("A2b  rz_q 与 cz 对易（q=1 侧）",
      kron(1, rz(th)) @ cz(), cz() @ kron(1, rz(th)))
check("A3  sx·sx = x", sx() @ sx(), x())
check("A4  x_0·cz = cz·x_0·rz_1(π)·i",
      kron(0, x()) @ cz(),
      cz() @ kron(0, x()) @ kron(1, rz(PI)) * 1j)
check("A5  x·rz(θ) = rz(−θ)·x",
      x() @ rz(th), rz(-th) @ x())
check("A7  sx·rz(θ)·sx = X·R_y(θ)",
      sx() @ rz(th) @ sx(),
      x() @ np.array([[np.cos(th / 2), -np.sin(th / 2)],
                      [np.sin(th / 2), np.cos(th / 2)]]))
check("A8  rz(π/2)·sx·rz(π/2) = H（模全局相位）",
      rz(PI / 2) @ sx() @ rz(PI / 2), h())
check("A9  sx·rz(π)·sx = Z",
      sx() @ rz(PI) @ sx(), np.diag([1.0, -1.0]))

# A12（2026-11-13 新增）：H·rz·H 的 v4 merge 形态 = rx(θ)
# （rz(π/2)·sx·rz(π+θ)·sx·rz(π/2)，merge_adjacent_rz 后产物形态）
def rx(t):
    return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)],
                     [-1j * np.sin(t / 2), np.cos(t / 2)]])


check("A12  rz(π/2)·sx·rz(π+θ)·sx·rz(π/2) = rx(θ)（v4 merge 形态）",
      rz(PI / 2) @ sx() @ rz(PI + th) @ sx() @ rz(PI / 2), rx(th))
# A13（反例确认）：π/2 版（构造错误版）≠ rx
check("A13[反例]  rz(π/2)·sx·rz(π/2+θ)·sx·rz(π/2) ≠ rx(θ)（预期 FAIL）",
      rz(PI / 2) @ sx() @ rz(PI / 2 + th) @ sx() @ rz(PI / 2), rx(th))
# A14：U5 的 X·rx 形式
check("A14  rz(π/2)·sx·rz(π/2+θ)·sx·rz(π/2) = X·rx(π/2+θ)",
      rz(PI / 2) @ sx() @ rz(PI / 2 + th) @ sx() @ rz(PI / 2),
      x() @ rx(PI / 2 + th))

# ============ B 组：H 与 cz/cx 的交换（修正版） ============
H0 = kron(0, h())
H1 = kron(1, h())

# B1（第一版通过）：H 在 cz 的目标位侧穿 → CNOT（控制=另一比特）
check("B1  H_b·cz(a,b) = cx(a,b)·H_b（H 在第二比特侧）",
      H1 @ cz(), cx01() @ H1)
# B1b 修正：H 在控制位侧穿 → CNOT 方向翻转
check("B1b  H_a·cz(a,b) = cx(b,a)·H_a（H 在第一比特侧，方向翻转）",
      H0 @ cz(), cx10() @ H0)
# B3 修正：H 在目标位共轭 → CNOT（标准）
check("B3  H_b·cz(a,b)·H_b = cx(a,b)（H 目标位共轭）",
      H1 @ cz() @ H1, cx01())
check("B3b  H_a·cz(a,b)·H_a = cx(b,a)（H 控制位共轭，方向翻转）",
      H0 @ cz() @ H0, cx10())
# B4 修正：H_2 与 cz(1,0) 对易（cz 在低位，H 在比特 2）
check("B4  H_2 与 cz(1,0) 对易（不相交，3 比特）",
      np.kron(h(), np.eye(4)) @ np.kron(np.eye(2), cz()),
      np.kron(np.eye(2), cz()) @ np.kron(h(), np.eye(4)))
check("B5  H·H = I", h() @ h(), np.eye(2))

# B6（新）：H 穿 cz 的复制形式——H_b·cz(a,b) = H_b·cx(a,b) 侧写
# cz·H_b = H_b·cx(a,b)（B1 取逆：cz = H_b·cx·H_b → cz·H_b = H_b·cx）
check("B6  cz(a,b)·H_b = H_b·cx(a,b)（B1 镜像）",
      cz() @ H1, H1 @ cx01())
check("B6b  cz(a,b)·H_a = H_a·cx(b,a)（B1b 镜像）",
      cz() @ H0, H0 @ cx10())

# ============ C 组：H 半块与游程归约（第一版已过，复验） ============
check("C1  sx·rz(π/2)·rz(π/2)·sx = Z",
      sx() @ rz(PI / 2) @ rz(PI / 2) @ sx(), np.diag([1.0, -1.0]))
check("C3  H·rz(θ)·H = rx(θ)（模全局相位）",
      h() @ rz(th) @ h(),
      np.array([[np.cos(th / 2), -1j * np.sin(th / 2)],
                [-1j * np.sin(th / 2), np.cos(th / 2)]]))
check("C4  H·rz(θ) = rx(θ)·H（模全局相位）",
      h() @ rz(th),
      np.array([[np.cos(th / 2), -1j * np.sin(th / 2)],
                [-1j * np.sin(th / 2), np.cos(th / 2)]]) @ h())
# C5（新）：H 半块拼合——两个后半块夹 cz 不成 H，但 H·rz 吸收型
check("C5  rz(π/2)·sx·rz(θ)·sx·rz(π/2) 的 H 夹 rx 形（数值参考）",
      rz(PI / 2) @ sx() @ rz(th) @ sx() @ rz(PI / 2),
      rz(PI / 2) @ x() @ np.array([[np.cos(th / 2), -np.sin(th / 2)],
                                   [np.sin(th / 2), np.cos(th / 2)]])
      @ rz(PI / 2))

# ============ P 组：CNOT 域相位多项式核心 ============
# P1: rz 与 CNOT 控制位对易
check("P1  rz_a 与 cx(a,b) 控制位对易",
      kron(0, rz(th)) @ cx01(), cx01() @ kron(0, rz(th)))
# P2: rz 穿 CNOT 目标位 → RZZ（两比特，非封闭于 1Q）
check("P2  cx(a,b)·rz_b(θ) = RZZ_ab(θ)·cx(a,b)",
      cx01() @ kron(1, rz(th)), rzz(th) @ cx01())
check("P2b  rz_b(θ)·cx(a,b) = cx(a,b)·RZZ_ab(θ)",
      kron(1, rz(th)) @ cx01(), cx01() @ rzz(th))
# P3: X 与 CNOT 目标位对易
check("P3  x_b 与 cx(a,b) 目标位对易",
      kron(1, x()) @ cx01(), cx01() @ kron(1, x()))
# P4: X 穿 CNOT 控制位 → 复制
check("P4  x_a·cx(a,b) = cx(a,b)·x_a·x_b",
      kron(0, x()) @ cx01(), cx01() @ kron(0, x()) @ kron(1, x()))
# P5: H 与 CNOT 的交换（2026-11-13 修正：H 目标位共轭 CNOT → CZ；
# 方向翻转需双 H：H_a·H_b·cx(a,b)·H_a·H_b = cx(b,a)）
check("P5  H_b·cx(a,b) = cz(a,b)·H_b（H 目标位穿 CNOT → CZ）",
      H1 @ cx01(), cz() @ H1)
check("P5b  H_b·cx(a,b)·H_b = cz(a,b)（H 目标位共轭 → CZ）",
      H1 @ cx01() @ H1, cz())
check("P5c  H_a·H_b·cx(a,b)·H_a·H_b = cx(b,a)（双 H 方向翻转）",
      H0 @ H1 @ cx01() @ H0 @ H1, cx10())

# P6: CNOT 域相位多项式标准引理——RZZ 与 CNOT 控制位 rz 的交互
# RZZ_ab(θ)·rz_a(φ) = rz_a(φ)·RZZ_ab(θ)（对角，对易）
check("P6  RZZ_ab(θ) 与 rz_a 对易",
      rzz(th) @ kron(0, rz(al)), kron(0, rz(al)) @ rzz(th))

# P7: cz 域等价——sx 穿 cz 不封闭（预期 FAIL，机制边界确认）
print("   （预期 FAIL）sx_b·cz 的封闭性检查：")
check("P7[反例]  sx_b 与 cz 对易（预期 FAIL）",
      kron(1, sx()) @ cz(), cz() @ kron(1, sx()))

# P8: X 穿 CZ 的双向（A4 镜像）
check("P8  cz·x_0 = x_0·rz_1(π)·cz·(−i)",
      cz() @ kron(0, x()),
      kron(0, x()) @ kron(1, rz(PI)) @ cz() * (-1j))

print()
n_fail = sum(1 for _, ok in results if not ok)
n_expected_fail = sum(1 for n, ok in results if "反例" in n and not ok)
print(f"总计 {len(results)} 项：通过 {len(results)-n_fail}、"
      f"失败 {n_fail}（其中预期反例 {n_expected_fail} 项）")
if n_fail != n_expected_fail:
    print("!! 存在非预期失败——机制设计有错误，禁止进入实现")
    sys.exit(1)
print("全部非反例恒等式通过——可进入设计文档")
