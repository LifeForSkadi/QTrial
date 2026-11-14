"""时间感知保真度模型测试（执行顺序第 5 项）。

预注册判据（2026-11-13，实现前写定——见 academic_review_gaps.md §5 项 5）

P1（rz 虚拟门）：仅 rz 的线路：时间 = 0、时间模型 log 保真度 = 0；
  乘积模型仍按 err_1q 计（历史口径保留用于对照）。
P2（时间量纲与门时间）：单 cz 线路时间 = 0.06 µs；swap 线路时间 =
  0.18 µs（3 个 2Q），且其时间模型误差等于 3 个串联 cz（1−(1−ε)³）。
P3（空闲退相干）：同门序列但强制比特等待的线路，T2（或 T1）降低时
  时间模型对数保真度严格下降；无空闲的最小线路与乘积模型一致
  （仅门误差 + 读出误差）。
P4（口径切换）：默认 product 模式下 compute_metrics.log_est_fidelity
  == log_est_fidelity_product；QTRAIL_FIDELITY_MODEL=time 时
  == log_est_fidelity_time；两种模式下另两个分量始终返回。

失败含义：时间模型实现错误（rz 计误差/时间、swap 折算、空闲漏计或
符号错误），或切换开关改变默认口径。
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from qtrail.devices import build_grid3x3_spec
from qtrail.devices.calibration import CalibrationData
from qtrail.pure import metrics as M
from qtrail.pure.circuit import Circuit, Inst


@pytest.fixture(scope="module")
def spec():
    return build_grid3x3_spec(0)


def _copy_calib(spec, t2_scale=1.0):
    c = spec.calib
    return CalibrationData(t1=c.t1.copy(), t2=c.t2 * t2_scale,
                           err_1q=c.err_1q.copy(), err_ro=c.err_ro.copy(),
                           err_2q=dict(c.err_2q))


def test_rz_is_virtual(spec):
    circ = Circuit(2)
    for _ in range(20):
        circ.append(Inst("rz", (0,), (0.3,)))
        circ.append(Inst("rz", (1,), (1.1,)))
    log_time, t_us = M._time_log_fidelity(circ, spec.calib)
    assert log_time == 0.0
    assert t_us == 0.0
    assert M.estimate_log_fidelity(circ, spec.calib, model="time") == 0.0
    assert M.estimate_log_fidelity(circ, spec.calib,
                                   model="product") < 0.0


def test_gate_times_and_swap_conversion(spec):
    e = spec.calib.err_2q[(0, 1)]
    c1 = Circuit(2)
    c1.append(Inst("cz", (0, 1)))
    l1, t1 = M._time_log_fidelity(c1, spec.calib)
    assert abs(t1 - 0.06) < 1e-12
    assert abs(l1 - math.log(1 - e)) < 1e-12

    sw = Circuit(2)
    sw.append(Inst("swap", (0, 1)))
    ls, ts = M._time_log_fidelity(sw, spec.calib)
    assert abs(ts - 0.18) < 1e-12
    assert abs(ls - 3 * math.log(1 - e)) < 1e-12   # 1−(1−ε)³ 的对数

    # 三个串联 cz：与 swap 的时间模型误差相同
    c3 = Circuit(2)
    for _ in range(3):
        c3.append(Inst("cz", (0, 1)))
    l3, t3 = M._time_log_fidelity(c3, spec.calib)
    assert abs(t3 - 0.18) < 1e-12
    assert abs(l3 - ls) < 1e-12


def test_idle_decoherence_monotone(spec):
    # 4 比特链式 cz：q2/q3 必须等待前一对的繁忙比特（ASAP 等待）
    circ = Circuit(4)
    circ.append(Inst("cz", (0, 1)))
    circ.append(Inst("cz", (1, 2)))
    circ.append(Inst("cz", (2, 3)))
    log_hi, t_hi = M._time_log_fidelity(circ, _copy_calib(spec, 1.0))
    log_lo, t_lo = M._time_log_fidelity(circ, _copy_calib(spec, 0.5))
    assert abs(t_hi - t_lo) < 1e-12
    assert log_lo < log_hi - 1e-9, "T2 降低应使空闲退相干更严重"


def test_minimal_circuit_matches_product(spec):
    # 无空闲：单 cz + measure（测量比特的 free == t_end）
    circ = Circuit(2)
    circ.append(Inst("cz", (0, 1)))
    circ.measure(0, 0)
    lp = M._product_log_fidelity(circ, spec.calib)
    lt, _t = M._time_log_fidelity(circ, spec.calib)
    assert abs(lp - lt) < 1e-12


def test_model_switch(spec):
    circ = Circuit(2)
    circ.append(Inst("sx", (0,)))
    circ.append(Inst("cz", (0, 1)))
    circ.measure(0, 0)
    old = os.environ.pop("QTRAIL_FIDELITY_MODEL", None)
    try:
        m = M.compute_metrics(circ, 0, spec.calib)
        assert m["log_est_fidelity"] == m["log_est_fidelity_product"]
        assert "log_est_fidelity_time" in m and "est_time_us" in m
        os.environ["QTRAIL_FIDELITY_MODEL"] = "time"
        m2 = M.compute_metrics(circ, 0, spec.calib)
        assert m2["log_est_fidelity"] == m2["log_est_fidelity_time"]
        assert m2["log_est_fidelity_product"] == m["log_est_fidelity_product"]
    finally:
        os.environ.pop("QTRAIL_FIDELITY_MODEL", None)
        if old is not None:
            os.environ["QTRAIL_FIDELITY_MODEL"] = old
