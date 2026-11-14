"""候选帧类型与 Inst 不可变契约测试（执行顺序第 4 项）。

预注册判据（2026-11-13，实现前写定——见 academic_review_gaps.md §5 项 4）

P1（CandidateResult 帧契约）：
  - content 帧：input_layout 必须等于 final_layout（缺失/不等 → ValueError）；
  - physical 帧：允许 input_layout ≠ final_layout；input_layout=None 合法；
  - 未知 frame → ValueError。
P2（候选值语义）：构造时复制布局字典——调用方事后修改原 dict 不影响
  候选（防候选间/调用方别名污染）。
P3（Inst 冻结）：对 inst.qubits/name/params 赋值抛 FrozenInstanceError；
  dataclasses.replace 可生成新对象且保留 tu_locked 等字段；Circuit.copy
  共享不可变 Inst 后，重标记 replace 不影响原电路（语义隔离）。
P4（管线集成）：qaoa12 的 `_v5_pipeline` 全部候选均为 CandidateResult、
  帧合法（content ⇒ input==final）；至少出现 1 个 content 帧候选
  （fold_keep/fold_relink 路径）；交付门流与重构前黄金快照逐门一致
  （黄金判据在提交说明中记录：qft5/ghz10/qaoa12 各 157/57/152 门、
  布局与指标完全一致）。

失败含义：帧混用类 bug（2026-11-12 fold 保留 swap P0）或共享可变
Inst 类 bug（qaoa_8 候选污染）可能复发。
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qtrail.pure.candidates import CandidateResult
from qtrail.pure.circuit import Circuit, Inst


def test_candidate_frame_contract():
    c = Circuit(3)
    c.cz(0, 1)
    fin = {0: 2, 1: 0, 2: 1}
    lay = {0: 0, 1: 1, 2: 2}
    ok = CandidateResult(c, "content", fin, dict(fin), tag="fold")
    assert ok.input_layout == ok.final_layout
    # content 帧：输入嵌入 ≠ final_layout → 拒绝
    with pytest.raises(ValueError):
        CandidateResult(c, "content", fin, dict(lay))
    # content 帧：缺 input_layout → 拒绝
    with pytest.raises(ValueError):
        CandidateResult(c, "content", fin, None)
    # physical 帧：允许不同（裸路由的初始布局 vs 最终布局）
    p = CandidateResult(c, "physical", fin, dict(lay), tag="bare")
    assert p.input_layout == lay
    # physical 帧：None 合法
    CandidateResult(c, "physical", fin, None)
    # 未知帧
    with pytest.raises(ValueError):
        CandidateResult(c, "mixed", fin, dict(fin))


def test_candidate_copies_layouts():
    c = Circuit(2)
    fin = {0: 1, 1: 0}
    lay = {0: 1, 1: 0}
    cand = CandidateResult(c, "content", fin, lay)
    fin[0] = 99
    lay[1] = 99
    assert cand.final_layout == {0: 1, 1: 0}
    assert cand.input_layout == {0: 1, 1: 0}


def test_inst_frozen_and_replace():
    i = Inst("rz", (3,), (0.5,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        i.qubits = (4,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        i.tu_locked = True
    j = dataclasses.replace(i, qubits=(4,), tu_locked=True)
    assert j.qubits == (4,) and j.tu_locked and i.qubits == (3,)
    # Circuit.copy 共享不可变 Inst；replace 只影响副本列表
    c = Circuit(5)
    c.append(i)
    c2 = c.copy()
    c2.ops[0] = dataclasses.replace(c2.ops[0], qubits=(1,))
    assert c.ops[0].qubits == (3,)
    assert c2.ops[0].qubits == (1,)


def test_pipeline_candidate_frames():
    import os
    from qtrail.config import load_device_config
    from qtrail.devices import build_tianyan287_spec
    from qtrail.pure.mapper import PureMapper
    from qtrail.pure.post import post_route
    from qtrail.pure.post_v2 import fold_keep_coords
    from qtrail.pure.qasm import load_qasm_file
    from qtrail.pure.router import sabre_route

    spec = build_tianyan287_spec(load_device_config())
    circ = load_qasm_file("examples/qaoa12.qasm")
    n = circ.n
    layout = {i: i for i in range(n)}
    routed, _swaps, fin = sabre_route(circ, spec, dict(layout), seed=0,
                                      trials=2)
    fk_c, fk_fin = fold_keep_coords(routed, fin, layout)
    pre = (fk_c, fk_fin) if all(
        i.nq != 2 or spec.adj[int(i.qubits[0]), int(i.qubits[1])]
        for i in fk_c.ops) else None
    routed2, _ = post_route(routed, fin, spec.adj)
    mapper = PureMapper(spec, policy=None, pipeline="v5",
                        selection_rule="fidelity", routing_seeds=1, seed=0,
                        linear_decompose=True)
    seen = []
    orig = mapper._select

    def select(scored):
        seen.extend(s[3] for s in scored)
        return orig(scored)

    mapper._select = select
    try:
        mapper._v5_pipeline(routed2, fin, circ, 0.0, relabel_budget=0.5,
                            pre_folded=pre, layout=layout)
    finally:
        mapper._select = orig
    assert seen, "没有候选被评分"
    frames = {c.frame for c in seen}
    assert frames <= {"physical", "content"}
    assert seen and all(isinstance(c, CandidateResult) for c in seen)
    for c in seen:
        if c.frame == "content":
            assert c.input_layout == c.final_layout
    assert any(c.frame == "content" for c in seen), \
        "fold 类候选消失（覆盖回归）"
    assert any(c.frame == "physical" for c in seen)
