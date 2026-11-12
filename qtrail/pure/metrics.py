"""qiskit-free 度量：深度、2Q 深度、2Q 计数、SWAP 计数、保真度估计。

保真度模型（2026-11-13 第 5 项）：
  - "product"（默认，历史口径）：Π(1−ε)，rz 与 sx 同误差、无时间维度。
  - "time"（时间感知）：ASAP 调度 → 门误差 + 空闲 T1/Tφ 退相干 +
    rz 零误差/零时间 + swap 按 3 个 2Q 门折算 + 读出时间/误差。
  开关：环境变量 QTRAIL_FIDELITY_MODEL=time（默认 product，A/B 达标后
  再切默认；见 pending_decisions 补充五十一）。
"""
from __future__ import annotations

import math
import os

import numpy as np

from qtrail.pure.circuit import Circuit

# ---------------------------------------------------------------- 时间模型常数
# 超导平台量级（学术审查 §3.3：sx ~30ns、cz ~60ns、读出 ~1μs；
# 天衍-287 / 祖冲之三号类平台公开量级）。单位：微秒。
# 注：平台未提供逐门时间表，故用类别常数；模型用于相对比较。
GATE_TIME_US = {
    "rz": 0.0, "u1": 0.0,          # 虚拟 Z 旋转：零时间、零误差
    "sx": 0.030, "x": 0.030, "u3": 0.030,
    "cz": 0.060, "cx": 0.060,
    "swap": 0.180,                  # 3 个 cx 等效
}
READOUT_TIME_US = 1.0


def circuit_depth(circ: Circuit) -> int:
    return circ.layer_depth()


def twoq_depth(circ: Circuit) -> int:
    return circ.longest_path_2q()


def twoq_count(circ: Circuit) -> int:
    return circ.count_2q()


def _fidelity_model() -> str:
    return os.environ.get("QTRAIL_FIDELITY_MODEL", "product").strip().lower()


def _product_log_fidelity(circ: Circuit, calib, predictor=None) -> float:
    """历史乘积口径 Σlog(1−ε)（predictor 存在时用预测器）。"""
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


def _idle_log_factor(t: float, t1: float, t2: float) -> float:
    """空闲 t 微秒的平均保真度对数因子。

    Tφ 由 1/T2 = 1/(2T1) + 1/Tφ 反解（T2 ≤ 2T1 的物理约束）。
    平均门保真度近似：ε_idle ≈ (1−e^{−t/T1})/3 + (1−e^{−t/Tφ})/3
    （能量弛豫 + 纯退相位各贡献 1/3，标准退极化近似；模型用于相对
    比较，绝对声称需真机标定）。
    """
    if t <= 0.0:
        return 0.0
    t1 = max(float(t1), 1e-9)
    t2 = max(float(t2), 1e-9)
    inv_tphi = 1.0 / t2 - 1.0 / (2.0 * t1)
    eps1 = 1.0 - math.exp(-t / t1)
    epsphi = (1.0 - math.exp(-t * inv_tphi)) if inv_tphi > 1e-12 else 0.0
    eps = (eps1 + epsphi) / 3.0
    return math.log(max(1.0 - eps, 1e-12))


def _time_log_fidelity(circ: Circuit, calib) -> tuple[float, float]:
    """时间感知对数保真度 + 线路时间（µs）。

    结构：ASAP 调度（逐门取两端比特的最早空闲时刻）→
      ① 门误差：1Q 用 err_1q（rz/u1 虚拟门为 0）；2Q 用 err_2q；
         swap 按 3 个 2Q 门折算（误差 1−(1−ε)³、时间 3×t_2q）；
      ② 空闲退相干：等待时间按 _idle_log_factor（T1/Tφ）；
      ③ 读出：测量前空闲到全局末时刻 + err_ro（读出时间 1µs 量级，
         不额外计退相位——err_ro 已含读出过程误差）。
    假设与边界：门误差为标定值（已含门内退相干）；无串扰/泄漏；
    时间常数为类别常数（平台无逐门时间表）。
    """
    err2q = calib.err_2q
    median2q = float(np.median(list(err2q.values()))) if err2q else 1e-3
    free: dict[int, float] = {}
    active: set[int] = set()
    s = 0.0
    for inst in circ.ops:
        qs = [int(q) for q in inst.qubits]
        name = inst.name
        dur = GATE_TIME_US.get(name, 0.0)
        start = max(free.get(q, 0.0) for q in qs)
        for q in qs:
            wait = start - free.get(q, 0.0)
            if wait > 0.0:
                s += _idle_log_factor(wait, calib.t1[q], calib.t2[q])
        if inst.nq == 2:
            e = err2q.get((qs[0], qs[1]),
                          err2q.get((qs[1], qs[0]), median2q))
            if name == "swap":
                e = 1.0 - (1.0 - float(e)) ** 3
            s += math.log(max(1.0 - float(e), 1e-12))
        elif inst.nq == 1:
            if name not in ("rz", "u1"):
                s += math.log(max(1.0 - float(calib.err_1q[qs[0]]), 1e-12))
        for q in qs:
            free[q] = start + dur
            active.add(q)
    t_end = max((free[q] for q in active), default=0.0)
    for inst in circ.measures:
        q = int(inst.qubits[0])
        wait = max(0.0, t_end - free.get(q, 0.0))
        if wait > 0.0:
            s += _idle_log_factor(wait, calib.t1[q], calib.t2[q])
        s += math.log(max(1.0 - float(calib.err_ro[q]), 1e-12))
    return float(s), float(t_end)


def estimate_log_fidelity(circ: Circuit, calib, predictor=None,
                          model: str | None = None) -> float:
    """对数保真度（数值稳定；所有比较口径的唯一合法函数）。

    model=None → 读环境变量 QTRAIL_FIDELITY_MODEL（"product" 默认 /
    "time" 时间感知）。predictor 存在时始终走预测器（旧特征模型）。
    """
    if predictor is not None:
        f = predictor.predict(circ, calib)
        return float(math.log(max(float(f), 1e-300)))
    m = (model or _fidelity_model()).lower()
    if m == "time":
        return _time_log_fidelity(circ, calib)[0]
    return _product_log_fidelity(circ, calib)


def estimate_fidelity(circ: Circuit, calib, predictor=None,
                      model: str | None = None) -> float:
    """保真度估计（= exp(对数口径)）。默认乘积模型；QTRAIL_FIDELITY_MODEL
    =time 时走时间感知模型。

    ⚠️ 乘积在大型线路上下溢（<1e-300 为 0），不可用于候选比较——
    比较一律用 estimate_log_fidelity。
    """
    if predictor is not None:
        return predictor.predict(circ, calib)
    m = (model or _fidelity_model()).lower()
    if m == "time":
        return float(math.exp(_time_log_fidelity(circ, calib)[0]))
    return float(math.exp(_product_log_fidelity(circ, calib)))


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
    """指标（2026-11-13 第 5 项：双模型并存）。

    log_est_fidelity = 当前决胜口径（QTRAIL_FIDELITY_MODEL=product|time）；
    另恒返回 product/time 两个分量与线路时间估计，便于 A/B 与披露。
    """
    log_prod = _product_log_fidelity(circ, calib, predictor)
    log_time, time_us = _time_log_fidelity(circ, calib)
    use_time = (_fidelity_model() == "time" and predictor is None)
    active = log_time if use_time else log_prod
    return {
        "swap_count": swap_count,
        "twoq_count": twoq_count(circ),
        "depth": circuit_depth(circ),
        "twoq_depth": twoq_depth(circ),
        "est_fidelity": float(math.exp(active)),
        # 对数保真度（2026-11-12 审查 P0-3）：候选比较/守卫的唯一
        # 合法口径（乘积在大型线路上下溢）
        "log_est_fidelity": float(active),
        "log_est_fidelity_product": float(log_prod),
        "log_est_fidelity_time": float(log_time),
        "est_time_us": float(time_us),
    }
