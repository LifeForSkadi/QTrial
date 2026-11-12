# -*- coding: utf-8 -*-
"""设备容量守卫（独立审计 P2-2）。

真机 105 比特中 4 个禁用 → 101 可用；线路 n 超过可用数时，启发式
布局的 `path[k % len(path)]` 会回绕出重复物理位（静默语义错误）。
"""
from __future__ import annotations

import numpy as np
import pytest

from qtrail.config import load_device_config
from qtrail.devices.tianyan287 import build_tianyan287_real_spec
from qtrail.problems import ProgramGraph
from qtrail.pure.layout import heuristic_layout


def test_layout_rejects_over_capacity():
    spec = build_tianyan287_real_spec(load_device_config())
    usable = spec.n - len(set(spec.disabled_qubits))
    n = usable + 1
    graph = ProgramGraph(n=n, adj=np.zeros((n, n), dtype=np.float32),
                         node_features=None, logical_order=np.arange(n))
    with pytest.raises(ValueError, match="usable qubits"):
        heuristic_layout(graph, spec)
