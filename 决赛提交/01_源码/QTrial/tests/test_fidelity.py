"""QuEst 保真度预测器（arXiv:2210.16724）测试：门图构建、前向/反向、往返。"""
import numpy as np
import pytest
import torch

from qtrail.devices import build_grid3x3_spec
from qtrail.models.fidelity import D_IN, FidelityPredictor, build_gate_graph
from qtrail.pure.circuit import Circuit, Inst


@pytest.fixture(scope="module")
def spec():
    return build_grid3x3_spec()


@pytest.fixture(scope="module")
def circuit():
    circ = Circuit(3)
    circ.append(Inst("rz", (0,), (0.5,)))
    circ.append(Inst("sx", (1,)))
    circ.append(Inst("x", (2,)))
    circ.append(Inst("cz", (0, 1)))
    circ.measure(0, 0)
    return circ


def test_build_gate_graph_shapes(spec, circuit):
    x, adj, mask = build_gate_graph(circuit, spec.calib)
    n_ops = len(circuit.ops)
    n_meas = len(circuit.measures)
    assert x.shape == (n_ops + n_meas, D_IN)
    assert adj.shape == (n_ops + n_meas, n_ops + n_meas)
    assert mask.shape == (n_ops + n_meas,)
    assert mask.all()
    assert np.isfinite(x).all()


def test_build_gate_graph_gate_types(spec, circuit):
    x, _, _ = build_gate_graph(circuit, spec.calib)
    onehot = x[:, :7]
    assert np.allclose(onehot.sum(axis=1), 1.0)
    assert x[0, 0] == 1.0    # 第 0 个 op 是 rz（索引 0）
    assert x[3, 3] == 1.0    # 第 3 个 op 是 cz（索引 3）
    assert x[-1, 5] == 1.0   # 最后是 measure（索引 5）
    assert x[-1, 7 + 7] > 0  # measure 读出误差列非零
    assert x[0, 7 + 7] == 0.0  # 非 measure 门读出误差为 0


def test_predictor_forward_and_predict(spec, circuit):
    torch.manual_seed(0)
    model = FidelityPredictor()
    x, adj, mask = build_gate_graph(circuit, spec.calib)
    xt = torch.from_numpy(x)
    at = torch.from_numpy(adj)
    mt = torch.from_numpy(mask)
    out = model.forward(xt, at, mt)
    assert out.numel() == 1
    assert torch.isfinite(out)

    pred = model.predict(circuit, spec.calib)
    assert 0.0 <= pred <= 1.0


def test_predictor_backward(spec, circuit):
    torch.manual_seed(1)
    model = FidelityPredictor()
    x, adj, mask = build_gate_graph(circuit, spec.calib)
    out = model.forward(torch.from_numpy(x), torch.from_numpy(adj),
                        torch.from_numpy(mask))
    loss = (out - torch.tensor(0.5)) ** 2
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_predictor_checkpoint_roundtrip(spec, circuit, tmp_path):
    torch.manual_seed(2)
    model = FidelityPredictor(d=16, layers=1, heads=2, ff=32)
    model.eval()
    before = model.predict(circuit, spec.calib)

    path = tmp_path / "fid.pt"
    model.save_checkpoint(path, epoch=1, val_loss=0.1)
    model2, ckpt = FidelityPredictor.load_checkpoint(path)
    model2.eval()
    assert ckpt["epoch"] == 1
    assert model2.predict(circuit, spec.calib) == pytest.approx(before, abs=1e-6)


def test_predictor_on_empty_circuit(spec):
    """空线路（无门、无测量）也能构图并预测。"""
    circ = Circuit(2)
    model = FidelityPredictor()
    pred = model.predict(circ, spec.calib)
    assert 0.0 <= pred <= 1.0
