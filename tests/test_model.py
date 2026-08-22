"""Model tests: shapes, mask correctness, padding invariance, determinism,
checkpoint round-trip."""
import numpy as np
import pytest
import torch

from qtrail.config import ModelConfig
from qtrail.devices import build_grid3x3_spec
from qtrail.models import QAPolicy
from qtrail.problems import build_program_graph, collate_instances, random_program_graph


@pytest.fixture(scope="module")
def spec():
    return build_grid3x3_spec()


@pytest.fixture(scope="module")
def instances():
    rng = np.random.default_rng(0)
    gs = [random_program_graph(int(rng.integers(2, 6)), p=0.5, rng=rng) for _ in range(4)]
    gs.append(build_program_graph(4, [("cx", (0, 1)), ("cx", (1, 2)), ("cx", (2, 3))]))
    return gs


@pytest.mark.parametrize("encoder", ["gat", "gt"])
@pytest.mark.parametrize("mode", ["greedy", "sample"])
def test_forward_shapes_and_valid_layout(spec, instances, encoder, mode):
    torch.manual_seed(0)
    cfg = ModelConfig(encoder=encoder)
    policy = QAPolicy(cfg, spec.n)
    batch = collate_instances(instances, spec)
    logp, reward, pi = policy(batch, mode=mode)
    B, n_max = batch.p_idx.shape
    assert pi.shape == (B, n_max)
    assert reward.shape == (B,)
    assert logp.shape == (B,)
    # each instance's real nodes get distinct physical qubits in [0, N)
    for b in range(B):
        n = int(batch.p_n[b])
        row = pi[b, :n].tolist()
        assert len(set(row)) == n, f"duplicate assignment: {row}"
        assert all(0 <= q < spec.n for q in row)
    assert torch.all(torch.isfinite(reward))


def test_greedy_never_selects_assigned(spec, instances):
    torch.manual_seed(1)
    policy = QAPolicy(ModelConfig(), spec.n)
    batch = collate_instances(instances, spec)
    with torch.no_grad():
        _, _, pi = policy(batch, mode="greedy")
    for b in range(int(batch.p_n.shape[0])):
        n = int(batch.p_n[b])
        assert len(set(pi[b, :n].tolist())) == n


def test_padding_invariance(spec):
    """Appending padded (all-zero) slots must not change real-node decisions."""
    torch.manual_seed(3)
    policy = QAPolicy(ModelConfig(), spec.n)
    policy.eval()  # dropout must be off for determinism
    g = build_program_graph(3, [("cx", (0, 1)), ("cx", (1, 2))])
    b1 = collate_instances([g], spec)
    with torch.no_grad():
        _, _, pi1 = policy(b1, mode="greedy")

    # same instance padded into a batch of larger n_max with dummy instances
    dummy = build_program_graph(2, [])  # isolated nodes
    b2 = collate_instances([g, dummy], spec)
    with torch.no_grad():
        _, _, pi2 = policy(b2, mode="greedy")
    assert torch.equal(pi1[0, :3], pi2[0, :3]), "padding changed real-node layout"


def test_seed_determinism(spec, instances):
    batch = collate_instances(instances, spec)
    outs = []
    for _ in range(2):
        torch.manual_seed(11)
        policy = QAPolicy(ModelConfig(), spec.n)
        with torch.no_grad():
            outs.append(policy(batch, mode="greedy")[2])
    assert torch.equal(outs[0], outs[1])


def test_checkpoint_roundtrip(spec, instances, tmp_path):
    torch.manual_seed(5)
    cfg = ModelConfig(encoder="gat", rich_context=True)
    policy = QAPolicy(cfg, spec.n)
    policy.eval()
    batch = collate_instances(instances, spec)
    with torch.no_grad():
        _, _, pi_before = policy(batch, mode="greedy")

    path = tmp_path / "ckpt.pt"
    policy.save_checkpoint(path, epoch=0, val_cost=1.0)

    policy2, ckpt = QAPolicy.load_checkpoint(path, device_n=spec.n)
    policy2.eval()
    assert ckpt["epoch"] == 0
    with torch.no_grad():
        _, _, pi_after = policy2(batch, mode="greedy")
    assert torch.equal(pi_before, pi_after)


def test_both_encoders_produce_different_layouts(spec, instances):
    batch = collate_instances(instances, spec)
    p1 = QAPolicy(ModelConfig(encoder="gat"), spec.n)
    p2 = QAPolicy(ModelConfig(encoder="gt"), spec.n)
    with torch.no_grad():
        _, _, pi1 = p1(batch, mode="greedy")
        _, _, pi2 = p2(batch, mode="greedy")
    assert not torch.equal(pi1, pi2)


def test_gradient_flows(spec, instances):
    torch.manual_seed(7)
    policy = QAPolicy(ModelConfig(), spec.n)
    batch = collate_instances(instances, spec)
    logp, reward, _ = policy(batch, mode="sample")
    loss = -((reward - reward.mean()) * logp).mean()
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
