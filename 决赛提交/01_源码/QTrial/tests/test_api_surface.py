"""生产 API 面收敛测试（执行顺序第 7 项）。

预注册判据（2026-11-13）：
  P1 公共分解入口 = 3 参数（circ, spec=None, linear=True）；
  P2 消融入口 `_decompose_ablation` 保留完整 15 开关面（circ + 15 = 16）；
  P3 公共入口与"消融入口 + 生产默认 resynth_every=1"在 linear=True /
     linear=False / 带 spec（触发 A2）三种情形下产物逐门一致；
  P4 黄金不变性（仓库级，脚本核对）：qft5/ghz10/qaoa12/N25 映射产物
     与重构前快照逐门一致（外部快照 golden_item7.json）。
"""
from __future__ import annotations

import inspect

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.post_v2 import _decompose_ablation, \
    decompose_to_platform_v2


def test_public_signature_is_three_params():
    sig = inspect.signature(decompose_to_platform_v2)
    names = list(sig.parameters)
    assert names == ["circ", "spec", "linear"], names
    assert sig.parameters["spec"].default is None
    assert sig.parameters["linear"].default is True


def test_ablation_signature_retained():
    sig = inspect.signature(_decompose_ablation)
    assert len(sig.parameters) == 16, list(sig.parameters)


def _sig(circ):
    return [(i.name, tuple(int(q) for q in i.qubits),
             tuple(float(p) for p in i.params)) for i in circ.ops]


def _small_circuit():
    c = Circuit(4)
    c.cx(0, 1)
    c.append(Inst("rz", (0,), (0.3,)))
    c.swap(2, 3)
    c.append(Inst("u3", (1,), (0.2, 0.3, 0.4)))
    c.cx(1, 2)
    return c


def test_public_equals_ablation_defaults():
    from qtrail.devices import build_grid3x3_spec
    spec = build_grid3x3_spec(0)
    # 无 spec：linear 路径
    a = decompose_to_platform_v2(_small_circuit(), linear=True)
    b = _decompose_ablation(_small_circuit(), spec=None,
                            resynth_every=1, linear=True)
    assert _sig(a) == _sig(b)
    # 带 spec：linear 路径（spec 仅控制 A2 轮数）
    a = decompose_to_platform_v2(_small_circuit(), spec=spec, linear=True)
    b = _decompose_ablation(_small_circuit(), spec=spec,
                            resynth_every=1, linear=True)
    assert _sig(a) == _sig(b)
    # fixpoint 路径
    a = decompose_to_platform_v2(_small_circuit(), spec=spec, linear=False)
    b = _decompose_ablation(_small_circuit(), spec=spec,
                            resynth_every=1, linear=False)
    assert _sig(a) == _sig(b)
