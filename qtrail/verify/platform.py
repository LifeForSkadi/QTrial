"""天衍量子计算云平台验证接口（预留）。

验证协议：把「平台自身编译的原线路」与「我们映射优化后的线路」分别提交
真机执行，比较两者的测量分布（经典保真度 / 总变差距离）——若分布一致，
说明我们的优化保持了线路功能。

API key 获取与注入（用户后续自行添加）：
  方式一：环境变量 TIANYAN_LOGIN_KEY=<你的 login_key>
  方式二：TianyanVerifier(login_key="...") 显式传入
未提供 key 时 available() 为 False，verify() 抛出信息明确的错误。

依赖 cqlib（guarded import，未安装不影响本地仿真验证）。
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np


class TianyanVerifier:
    """天衍平台功能等价性验证器。"""

    def __init__(self, login_key: str | None = None,
                 machine: str = "tianyan-287"):
        self.machine = machine
        self.login_key = login_key or os.environ.get("TIANYAN_LOGIN_KEY", "") \
            or os.environ.get("CQLIB_LOGIN_KEY", "")
        self._platform = None

    # -------------------------------------------------------------- status
    def available(self) -> bool:
        """是否有可用的 key（cqlib 是否安装另查）。"""
        return bool(self.login_key)

    def _get_platform(self):
        if self._platform is None:
            if not self.login_key:
                raise RuntimeError(
                    "未配置天衍 API key：请设置环境变量 TIANYAN_LOGIN_KEY "
                    "（个人中心获取），或 TianyanVerifier(login_key=...) 显式传入")
            try:
                from cqlib import TianYanPlatform  # guarded import
            except ImportError as e:
                raise RuntimeError("cqlib 未安装（pip install cqlib）") from e
            self._platform = TianYanPlatform(login_key=self.login_key,
                                             machine_name=self.machine)
        return self._platform

    # ------------------------------------------------------------ submit
    def submit(self, qcis_text: str, name: str, shots: int = 12000) -> str:
        """提交 QCIS 线路，返回 query_id（供异步查询）。"""
        platform = self._get_platform()
        query_ids = platform.submit_job(circuit=qcis_text, name=name,
                                        num_shots=shots)
        return query_ids[0] if isinstance(query_ids, list) else str(query_ids)

    def fetch_results(self, query_id: str, max_wait_time: int = 3600):
        """查询结果（计数分布），返回 {bitstring: count}。"""
        platform = self._get_platform()
        res = platform.query_experiment(query_id=query_id,
                                        max_wait_time=max_wait_time)
        return _parse_counts(res)

    # ------------------------------------------------------------ verify
    def verify(self, mapped_qcis: str, reference_qcis: str = None,
               shots: int = 12000, max_wait_time: int = 3600) -> dict:
        """真机验证映射线路功能等价性。

        reference_qcis 缺省时以平台自身编译的原线路为参照——需要同时
        提供原始 QASM 字符串（由调用方负责转换，见 verify CLI）。

        Returns: {"equivalent", "classical_fidelity", "tvd", "shots",
                  "machine", "query_ids"}
        """
        if reference_qcis is None:
            raise ValueError("reference_qcis 必填：请提供原线路经平台编译"
                             "的 QCIS 作为功能参照")
        q_mapped = self.submit(mapped_qcis, "qtrail_verify_mapped", shots)
        q_ref = self.submit(reference_qcis, "qtrail_verify_ref", shots)
        c_mapped = self.fetch_results(q_mapped, max_wait_time)
        c_ref = self.fetch_results(q_ref, max_wait_time)
        return _compare_distributions(c_mapped, c_ref,
                                      shots=shots, query_ids=[q_mapped, q_ref],
                                      machine=self.machine)


# ------------------------------------------------------------------ helpers
def _parse_counts(result) -> dict:
    """防御性解析 cqlib 结果（计数分布格式兼容多种返回结构）。"""
    if isinstance(result, dict):
        for key in ("counts", "Counts", "probability", "distribution", "data"):
            v = result.get(key)
            if isinstance(v, dict):
                return v
        # 深层兜底：找第一个 dict 且值为数值的
        for v in result.values():
            if isinstance(v, dict) and v and all(
                    isinstance(x, (int, float)) for x in v.values()):
                return v
    return {}


def _compare_distributions(c1: dict, c2: dict, shots: int = 0,
                           **meta) -> dict:
    """比较两个测量分布：经典保真度与总变差距离。"""
    keys = sorted(set(c1) | set(c2))
    if not keys:
        return {"equivalent": False, "classical_fidelity": 0.0, "tvd": 1.0,
                **meta}
    p1 = np.array([float(c1.get(k, 0)) for k in keys])
    p2 = np.array([float(c2.get(k, 0)) for k in keys])
    p1 = p1 / max(p1.sum(), 1e-12)
    p2 = p2 / max(p2.sum(), 1e-12)
    classical_fid = float((np.sqrt(p1 * p2)).sum() ** 2)
    tvd = float(np.abs(p1 - p2).sum() / 2.0)
    # 阈值：经典保真度 ≥ 0.98 且 TVD ≤ 0.05 判定等价
    equivalent = classical_fid >= 0.98 and tvd <= 0.05
    return {"equivalent": equivalent,
            "classical_fidelity": round(classical_fid, 4),
            "tvd": round(tvd, 4), **meta}
