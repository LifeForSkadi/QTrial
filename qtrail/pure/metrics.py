"""qiskit-free 度量：深度、2Q 深度、2Q 计数、SWAP 计数、乘积保真度估计。"""
from __future__ import annotations

import math

import numpy as np

from qtrail.pure.circuit import Circuit


def circuit_depth(circ: Circuit) -> int:
    return circ.layer_depth()


def twoq_depth(circ: Circuit) -> int:
    return circ.longest_path_2q()


def twoq_count(circ: Circuit) -> int:
    return circ.count_2q()


def estimate_log_fidelity(circ: Circuit, calib, predictor=None) -> float:
    """对数保真度 Σlog(1−ε)（数值稳定、同序保序；2026-11-12 审查
    P0-3：乘积模型在 10^4+ 门线路上下溢为 0，绝对容差比较失效——
    v7 批次 189/504 条 <1e-12，非劣化守卫/L3 仲裁对 37.5% 样本
    失去判别力）。所有"比较"口径应使用本函数；乘积值仅作展示。
    """
    if predictor is not None:
        f = predictor.predict(circ, calib)
        return float(math.log(max(float(f), 1e-300)))
    s = 0.0
    err2q = calib.err_2q
    median2q = float(np.median(list(err2q.values()))) if err2q else 1e-3
    for inst in circ.ops:
        qs = inst.qubits
        if inst.nq == 2:
            e = err2q.get((qs[0], qs[1]), err2q.get((qs[1], qs[0]), median2q))
            s += math.log(max(1.0 - float(e), 1e-12))
        elif inst.nq == 1:
            s += math.log(max(1.0 - float(calib.err_1q[qs[0]]), 1e-12))
    for inst in circ.measures:
        q = inst.qubits[0]
        s += math.log(max(1.0 - float(calib.err_ro[q]), 1e-12))
    return float(s)


def estimate_fidelity(circ: Circuit, calib, predictor=None) -> float:
    """保真度估计：predictor 为 None 走乘积模型 Π(1−ε_1q)·Π(1−ε_2q)·Π(1−ε_ro)
    （与 qiskit 版同口径）；否则走 QuEst 图 Transformer 预测器（可选）。

    ⚠️ 2026-11-12 审查 P0-3：乘积在大型线路上下溢（<1e-300 为 0），
    不可用于候选比较——比较一律用 estimate_log_fidelity。
    """
    if predictor is not None:
        return predictor.predict(circ, calib)
    f = 1.0
    err2q = calib.err_2q
    median2q = float(np.median(list(err2q.values()))) if err2q else 1e-3
    for inst in circ.ops:
        qs = inst.qubits
        if inst.nq == 2:
            e = err2q.get((qs[0], qs[1]), err2q.get((qs[1], qs[0]), median2q))
            f *= max(1.0 - float(e), 1e-12)
        elif inst.nq == 1:
            f *= max(1.0 - float(calib.err_1q[qs[0]]), 1e-12)
    for inst in circ.measures:
        q = inst.qubits[0]
        f *= max(1.0 - float(calib.err_ro[q]), 1e-12)
    return float(f)


def mean_twoq_error(circ: Circuit, calib) -> float | None:
    """门数加权的 2Q 误差均值（布局噪声规避质量的机制证据）。"""
    err2q = calib.err_2q
    median2q = float(np.median(list(err2q.values()))) if err2q else 1e-3
    total_w, total_e = 0.0, 0.0
    for inst in circ.ops:
        if inst.nq != 2:
            continue
        qs = inst.qubits
        e = err2q.get((qs[0], qs[1]), err2q.get((qs[1], qs[0]), median2q))
        total_w += 1.0
        total_e += float(e)
    return total_e / total_w if total_w else None


def compute_metrics(circ: Circuit, swap_count: int, calib, predictor=None) -> dict:
    return {
        "swap_count": swap_count,
        "twoq_count": twoq_count(circ),
        "depth": circuit_depth(circ),
        "twoq_depth": twoq_depth(circ),
        "est_fidelity": estimate_fidelity(circ, calib, predictor),
        # 对数保真度（2026-11-12 审查 P0-3）：候选比较/守卫的唯一
        # 合法口径（乘积在大型线路上下溢）
        "log_est_fidelity": estimate_log_fidelity(circ, calib, predictor),
    }
