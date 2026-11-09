# -*- coding: utf-8 -*-
"""端到端硬件可执行性断言（2026-11-09 缺口 3）。

覆盖之前缺失的验证：管线最终输出的所有 2Q 门必须落在设备耦合图上、
所有比特不得是禁用比特、QCIS 只含平台合法指令。态矢量等价测试只查
语义、不查连通性，本文件补上硬件合法性闭环。
"""
from pathlib import Path

import numpy as np
import pytest

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec, build_tianyan287_real_spec
from qtrail.pure.layout import heuristic_layout
from qtrail.pure.mapper import PureMapper
from qtrail.pure.post import decompose_to_platform, post_route
from qtrail.pure.qasm import load_qasm_file
from qtrail.pure.router import sabre_route

ROOT = Path(__file__).resolve().parent.parent
CASES = ["examples/qft5.qasm", "examples/ghz10.qasm", "examples/qaoa12.qasm"]


def _assert_hardware_valid(circ, spec):
    disabled = set(spec.disabled_qubits)
    for inst in circ.ops:
        for q in inst.qubits:
            assert int(q) not in disabled, f"禁用比特 {q} 出现在输出中"
        if inst.name == "cz":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            assert spec.adj[a, b], f"非相邻 CZ ({a},{b}) 出现在输出中"


@pytest.mark.parametrize("case", CASES)
def test_post_route_output_hardware_valid_synthetic(case):
    """合成口径：路由+后处理+分解产物全部 2Q 门相邻。"""
    spec = build_tianyan287_spec(load_device_config())
    circ = load_qasm_file(ROOT / case)
    routed, swaps, fin = sabre_route(circ, spec, {i: i for i in range(circ.n)},
                                     seed=42)
    post_route(routed, fin, spec.adj)
    final_qc = decompose_to_platform(routed)
    _assert_hardware_valid(final_qc, spec)


@pytest.mark.parametrize("case", CASES)
def test_post_route_output_hardware_valid_real(case):
    """真机口径：同上（真机拓扑 + 禁用比特）。"""
    spec = build_tianyan287_real_spec()
    circ = load_qasm_file(ROOT / case)
    routed, swaps, fin = sabre_route(circ, spec, {i: i for i in range(circ.n)},
                                     seed=42)
    post_route(routed, fin, spec.adj)
    final_qc = decompose_to_platform(routed)
    _assert_hardware_valid(final_qc, spec)


def test_heuristic_layout_avoids_disabled():
    """启发式蛇形布局不得使用禁用比特（缺口 1）。"""
    from qtrail.problems import random_program_graph
    spec = build_tianyan287_real_spec()
    graph = random_program_graph(100, p=0.1)
    pi = heuristic_layout(graph, spec)
    disabled = set(spec.disabled_qubits)
    assert not (set(pi.tolist()) & disabled), "启发式布局占用了禁用比特"


def test_trivial_candidate_avoids_disabled():
    """平凡布局候选不得使用禁用比特（缺口 1）。"""
    spec = build_tianyan287_real_spec()
    usable = spec.usable_positions
    n = 100
    pi = usable[np.arange(n) % len(usable)]
    assert len(set(pi.tolist()) & set(spec.disabled_qubits)) == 0


def test_qcis_platform_instructions():
    """QCIS 只含平台合法指令（X2P/X2M/Y2P/Y2M/RZ/CZ/M，无 X）。"""
    import re
    from qtrail.pure.export import to_qcis
    spec = build_tianyan287_spec(load_device_config())
    circ = load_qasm_file(ROOT / "examples/qft5.qasm")
    routed, _, fin = sabre_route(circ, spec, {i: i for i in range(circ.n)},
                                 seed=42)
    post_route(routed, fin, spec.adj)
    final_qc = decompose_to_platform(routed)
    text = to_qcis(final_qc, labels=spec.qubit_labels)
    for line in text.splitlines():
        if line.strip():
            assert re.match(r"^(X2P|X2M|Y2P|Y2M|RZ|CZ|M)\s", line), \
                f"非法 QCIS 指令: {line}"


def test_full_pipeline_real_no_disabled_qubits():
    """全管线（含 RL 权重）真机口径：最终布局与输出不含禁用比特。

    运行时较长（多起点解码），只测一条小线路。
    """
    import torch
    from qtrail.config import Config
    from qtrail.models import QAPolicy
    spec = build_tianyan287_real_spec()
    ckpt = ROOT / "checkpoints/" \
        "tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
    if not ckpt.exists():
        pytest.skip("checkpoint 不存在")
    policy, ck = QAPolicy.load_checkpoint(str(ckpt), device_n=spec.n,
                                          map_location="cpu")
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config("configs/device_tianyan287_real.yaml")
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev="cpu", seed=42,
                        selection_rule="fidelity")
    circ = load_qasm_file(ROOT / "examples/qft5.qasm")
    res = mapper.map_circuit(circ, circuit_id="qft5")
    disabled = set(spec.disabled_qubits)
    assert not (set(res["layout"].values()) & disabled), \
        "最终布局占用了禁用比特"
    assert not (set(res["final_layout"].values()) & disabled)
    _assert_hardware_valid(res["routed_circuit"], spec)
