"""LAUREL 残差（arXiv:2411.07501）测试：模式语义、形状、梯度、旧权重兼容。"""
import numpy as np
import pytest
import torch

from qtrail.config import ModelConfig
from qtrail.devices import build_grid3x3_spec
from qtrail.models import QAPolicy
from qtrail.models.laurel import LaurelResidual
from qtrail.problems import build_program_graph, collate_instances, random_program_graph


@pytest.fixture(scope="module")
def spec():
    return build_grid3x3_spec()


@pytest.fixture(scope="module")
def instances():
    rng = np.random.default_rng(0)
    gs = [random_program_graph(int(rng.integers(2, 6)), p=0.5, rng=rng) for _ in range(3)]
    gs.append(build_program_graph(4, [("cx", (0, 1)), ("cx", (1, 2)), ("cx", (2, 3))]))
    return gs


def _branch_resid(d=8, B=2, n=5):
    torch.manual_seed(0)
    return torch.randn(B, n, d), torch.randn(B, n, d)


def test_none_mode_is_identity():
    m = LaurelResidual(8, mode="none")
    b, r = _branch_resid()
    assert torch.equal(m(b, r), b + r)


def test_rw_forward_shape_and_convex():
    m = LaurelResidual(8, mode="rw")
    b, r = _branch_resid()
    out = m(b, r)
    assert out.shape == b.shape
    assert torch.all(torch.isfinite(out))
    a, beta = torch.softmax(m.logits, dim=0)
    assert a.item() + beta.item() == pytest.approx(1.0)


def test_lr_reduces_to_identity_at_init():
    """初始化 A,B ~ N(0,0.01) -> AB≈0 -> 输出 ≈ 原样残差。"""
    m = LaurelResidual(8, mode="lr")
    b, r = _branch_resid()
    out = m(b, r)
    assert torch.allclose(out, b + r, atol=0.05)


def test_rw_lr_forward_shape():
    m = LaurelResidual(8, mode="rw_lr", rank=4)
    b, r = _branch_resid()
    out = m(b, r)
    assert out.shape == b.shape
    assert torch.all(torch.isfinite(out))


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        LaurelResidual(8, mode="bogus")


@pytest.mark.parametrize("encoder", ["gat", "gt"])
@pytest.mark.parametrize("laurel", ["none", "rw", "lr", "rw_lr"])
def test_policy_forward_with_laurel(spec, instances, encoder, laurel):
    torch.manual_seed(0)
    cfg = ModelConfig(encoder=encoder, d=16, gat_layers=2, gat_heads=4,
                      gt_layers=2, gt_heads=4, gt_ff=64, laurel=laurel,
                      laurel_rank=2)
    policy = QAPolicy(cfg, spec.n)
    batch = collate_instances(instances, spec)
    logp, reward, pi = policy(batch, mode="greedy")
    assert pi.shape == (batch.p_idx.shape[0], batch.p_idx.shape[1])
    assert torch.all(torch.isfinite(reward))


def test_gradient_flows_with_laurel(spec, instances):
    torch.manual_seed(1)
    cfg = ModelConfig(laurel="rw_lr", laurel_rank=2)
    policy = QAPolicy(cfg, spec.n)
    batch = collate_instances(instances, spec)
    logp, reward, _ = policy(batch, mode="sample")
    loss = -((reward - reward.mean()) * logp).mean()
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_laurel_checkpoint_roundtrip(spec, instances, tmp_path):
    torch.manual_seed(2)
    cfg = ModelConfig(encoder="gat", laurel="rw_lr", laurel_rank=2)
    policy = QAPolicy(cfg, spec.n)
    policy.eval()
    batch = collate_instances(instances, spec)
    with torch.no_grad():
        _, _, pi_before = policy(batch, mode="greedy")

    path = tmp_path / "laurel_ckpt.pt"
    policy.save_checkpoint(path, epoch=0, val_cost=1.0)
    policy2, ckpt = QAPolicy.load_checkpoint(path, device_n=spec.n)
    policy2.eval()
    with torch.no_grad():
        _, _, pi_after = policy2(batch, mode="greedy")
    assert torch.equal(pi_before, pi_after)


def test_old_checkpoint_without_laurel_field_loads(spec, instances):
    """旧权重 pickled 的 ModelConfig 无 laurel 字段 -> getattr 兜底为 none。"""
    cfg = ModelConfig(encoder="gat")
    del cfg.laurel
    del cfg.laurel_rank
    policy = QAPolicy(cfg, spec.n)  # 不应抛 AttributeError
    batch = collate_instances(instances, spec)
    with torch.no_grad():
        _, _, pi = policy(batch, mode="greedy")
    assert pi.shape == (batch.p_idx.shape[0], batch.p_idx.shape[1])
