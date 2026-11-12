"""自研 Weyl/Vatan-Williams 3-CX 合成测试（执行顺序第 3 项）。

预注册判据（2026-11-13，实现前写定——见 academic_review_gaps.md §5 项 3）

P1（Weyl 分解自洽）：200 条随机 SU(4)（固定种子）+ 8 个特殊门
  （I/CX/CZ/SWAP/iSWAP/对角相位/受控相位/小角度），decompose 的
  重建 U = e^{iφ}(K1l⊗K1r)·Ud(a,b,c)·(K2l⊗K2r) 重叠 > 1-1e-9，
  且 a,b,c 满足 Weyl 腔 π/4 ≥ a ≥ b ≥ |c|。
P2（VW 合成精确）：同语料，synth_insts → _block_unitary 重建重叠
  > 1-1e-9；CX 数 ≤ 3；门集 ⊆ {u3, cx}；比特合法（pair 两端）。
P3（零 qiskit 依赖）：子进程仅 import qtrail.pure.weyl 并执行
  synth_block_analytic（默认 vw 路径）后，"qiskit" 不在 sys.modules。
P4（与 qiskit 对照，若可用）：同一 U 的两种后端各自重建均 > 1-1e-9；
  vw 的 CX 数 ≤ qiskit 的 CX 数 + 1（VW 固定 3-CX，允许特殊类多 1）。

失败含义：自研闭式合成器数学错误或引入 qiskit 运行时依赖。

管线 A/B 判据（不在 pytest，脚本裁决，见 pending_decisions）：
heis16/qft20/N25（+qv32 若可承受）在 QTRAIL_A2_BACKEND=vw 下
2Q 门数不增、1Q 增幅 ≤2%、depth 增幅 ≤2%、logf 不劣于 -0.01、
全管线墙钟增幅 ≤5%；不达标即默认回退 qiskit 后端并记录负结果。
"""
from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from qtrail.pure.circuit import Inst
from qtrail.pure import weyl as W


def _rand_su4(rng):
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q / np.linalg.det(Q) ** 0.25


def _special_unitaries():
    cx = W._CX.copy()
    cz = np.diag([1, 1, 1, -1]).astype(complex)
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)
    iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0],
                      [0, 0, 0, 1]], dtype=complex)
    diag = np.diag([1, 1j, -1, -1j]).astype(complex)
    cphase = np.diag([1, 1, 1, np.exp(0.7j)]).astype(complex)
    small = np.eye(4, dtype=complex)
    small[0, 3] = 0.1
    small[3, 0] = -0.1
    # 小角度纠缠门 exp(i·0.05·XX)
    xx = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
                   [1, 0, 0, 0]], dtype=complex)
    from scipy.linalg import expm
    smallis = expm(1j * 0.05 * xx)
    return [("I", np.eye(4, dtype=complex)), ("CX", cx), ("CZ", cz),
            ("SWAP", swap), ("iSWAP", iswap), ("diag", diag),
            ("cphase", cphase), ("small_xx", smallis)]


def test_weyl_decomposition_roundtrip():
    rng = np.random.default_rng(0)
    cases = [(_rand_su4(rng), "random") for _ in range(200)]
    cases += [(U, tag) for tag, U in _special_unitaries()]
    worst = 1.0
    for U, tag in cases:
        t = W.decompose(U)
        assert t["a"] >= t["b"] - 1e-12
        assert t["b"] >= abs(t["c"]) - 1e-12
        assert t["a"] <= math.pi / 4 + 1e-12
        # Ud(a,b,c) = exp(i(aXX+bYY+cZZ))（用 synthesis.w_matrix）
        from qtrail.pure.synthesis import w_matrix
        Ud = w_matrix((t["a"], t["b"], t["c"]))
        Urec = np.exp(1j * t["global_phase"]) \
            * np.kron(t["K1l"], t["K1r"]) @ Ud \
            @ np.kron(t["K2l"], t["K2r"])
        ov = abs(np.vdot(Urec, U)) ** 2 / 16
        assert ov > 1 - 1e-9, f"{tag} Weyl 重建失败 ov={ov}"
        worst = min(worst, ov)
    assert worst > 1 - 1e-9


def test_vw_synthesis_exact():
    from qtrail.pure.post_v2 import _block_unitary
    rng = np.random.default_rng(1)
    cases = [(_rand_su4(rng), "random") for _ in range(200)]
    cases += [(U, tag) for tag, U in _special_unitaries()]
    worst = 1.0
    for U, tag in cases:
        insts = W.synth_insts(U, (0, 1))
        assert all(i.name in ("u3", "cx") for i in insts)
        assert sum(1 for i in insts if i.name == "cx") <= 3
        for inst in insts:
            assert all(int(q) in (0, 1) for q in inst.qubits)
        Uc = _block_unitary(insts, (0, 1))
        ov = abs(np.vdot(Uc, U)) ** 2 / 16
        assert ov > 1 - 1e-9, f"{tag} VW 合成失败 ov={ov}"
        worst = min(worst, ov)
    assert worst > 1 - 1e-9


def test_no_qiskit_runtime_dependency():
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from qtrail.pure.weyl import synth_insts\n"
        "A = np.eye(4, dtype=complex)\n"
        "A[0, 3] = 0.2; A[3, 0] = 0.2\n"
        "Q, R = np.linalg.qr(A + 1j*np.eye(4)*0.1)\n"
        "U = Q @ np.diag(np.diag(R)/np.abs(np.diag(R)))\n"
        "U = U / np.linalg.det(U)**0.25\n"
        "synth_insts(U, (0, 1))\n"
        "assert 'qiskit' not in sys.modules, 'qiskit imported!'\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_vw_vs_qiskit_reference():
    pytest.importorskip("qiskit")
    from qtrail.pure.post_v2 import _block_unitary
    from qtrail.pure.synthesis import _synth_block_qiskit
    rng = np.random.default_rng(2)
    for _ in range(30):
        U = _rand_su4(rng)
        insts = W.synth_insts(U, (0, 1))
        ref = _synth_block_qiskit(U, (0, 1))
        ov = abs(np.vdot(_block_unitary(insts, (0, 1)), U)) ** 2 / 16
        ov2 = abs(np.vdot(_block_unitary(ref, (0, 1)), U)) ** 2 / 16
        assert ov > 1 - 1e-9 and ov2 > 1 - 1e-9, (ov, ov2)
        n_vw = sum(1 for i in insts if i.name == "cx")
        n_qk = sum(1 for i in ref if i.name == "cx")
        assert n_vw <= n_qk + 1
