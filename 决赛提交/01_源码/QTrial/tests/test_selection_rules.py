"""混合竞技决胜规则单元测试：swap / fidelity / depth 三种规则的选优逻辑。"""
import numpy as np
import pytest

from qtrail.pipeline.mapper import Mapper


def _mapper(rule):
    from qtrail.devices import build_grid3x3_spec
    m = Mapper(build_grid3x3_spec(), selection_rule=rule)
    return m


def test_swap_rule_prefers_min_swaps():
    m = _mapper("swap")
    scored = [
        (10, 100, 0.9, np.array([0])),   # 最少 SWAP
        (12, 50, 0.99, np.array([1])),   # +2 容差内，保真度更高
        (5, 200, 0.5, np.array([2])),    # SWAP 更少
    ]
    pi, best = m._select_candidate(scored)
    assert np.array_equal(pi, [2]) and best == 5


def test_swap_rule_tiebreak_by_fidelity():
    m = _mapper("swap")
    scored = [
        (10, 100, 0.8, np.array([0])),
        (10, 100, 0.95, np.array([1])),
    ]
    pi, _ = m._select_candidate(scored)
    assert np.array_equal(pi, [1])


def test_fidelity_rule_prefers_highest_fidelity():
    m = _mapper("fidelity")
    scored = [
        (100, 50, 0.95, np.array([0])),   # 保真度最高，SWAP 多
        (10, 50, 0.90, np.array([1])),    # 容差外（差 ~5%）
    ]
    pi, best = m._select_candidate(scored)
    assert np.array_equal(pi, [0])


def test_fidelity_rule_tiebreak_by_swaps():
    m = _mapper("fidelity")
    scored = [
        (20, 50, 0.95, np.array([0])),
        (12, 50, 0.945, np.array([1])),   # 2% 容差内 → 选 SWAP 少者
    ]
    pi, _ = m._select_candidate(scored)
    assert np.array_equal(pi, [1])


def test_depth_rule_prefers_shallow():
    m = _mapper("depth")
    scored = [
        (50, 80, 0.9, np.array([0])),     # 深度最浅
        (10, 200, 0.95, np.array([1])),   # 容差外（>5%+3）
    ]
    pi, _ = m._select_candidate(scored)
    assert np.array_equal(pi, [0])


def test_depth_rule_tiebreak_by_swaps():
    m = _mapper("depth")
    scored = [
        (20, 100, 0.9, np.array([0])),
        (12, 102, 0.9, np.array([1])),    # 容差内 → 选 SWAP 少者
    ]
    pi, _ = m._select_candidate(scored)
    assert np.array_equal(pi, [1])


def test_pair_better_swap_rule():
    m = _mapper("swap")
    ref = (10, 100, 0.9, np.array([0]))
    assert m._pair_better((5, 100, 0.1, None), ref)          # 严格少 SWAP
    assert m._pair_better((11, 100, 0.99, None), ref)        # 容差内保真度高
    assert not m._pair_better((11, 100, 0.5, None), ref)     # 容差内保真度低
    assert not m._pair_better((10, 100, 0.8, None), ref)     # 相同不更优


def test_pair_better_depth_rule():
    m = _mapper("depth")
    ref = (10, 100, 0.9, None)
    assert m._pair_better((10, 50, 0.1, None), ref)          # 深度显著浅
    assert m._pair_better((8, 104, 0.9, None), ref)          # 容差内 SWAP 少
    assert not m._pair_better((11, 104, 0.9, None), ref)     # 容差内 SWAP 多
    assert not m._pair_better((11, 109, 0.9, None), ref)     # 深度容差外且不浅
