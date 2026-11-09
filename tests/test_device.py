"""DeviceSpec unit tests: grid geometry, distances, calibration, normalization."""
import numpy as np
import pytest

from qtrail.devices import (build_grid3x3_spec, build_tianyan287_spec,
                            build_tianyan287_real_spec)
from qtrail.devices.calibration import generate_synthetic_calibration


def test_3x3_geometry():
    spec = build_grid3x3_spec()
    assert spec.n == 9
    assert spec.adj.sum() == 24  # 12 undirected edges
    # center (1,1) has degree 4, corners have degree 2
    assert spec.adj[4].sum() == 4
    assert spec.adj[0].sum() == 2
    # corner-to-opposite-corner distance = 4
    assert spec.dist[0, 8] == 4
    assert spec.dist[1, 7] == 2


def test_3x3_noise_distance_positive_and_bounded():
    spec = build_grid3x3_spec()
    assert np.all(spec.noise_dist >= 0)
    assert np.all(np.isfinite(spec.noise_dist))
    assert np.allclose(np.diag(spec.noise_dist), 0)
    # corner-to-corner path of 4 edges; weight <= 1 + 0.5*(5-1) + 0.1*2 = 3.2
    assert 0 < spec.noise_dist[0, 8] <= 4 * 3.2


def test_3x3_node_features_normalized():
    spec = build_grid3x3_spec()
    assert spec.node_features.shape == (9, 7)
    assert np.all(spec.node_features >= 0) and np.all(spec.node_features <= 1)


def test_tianyan287_geometry():
    spec = build_tianyan287_spec()
    assert spec.n == 105
    n_edges = spec.adj.sum() // 2
    assert n_edges == 188  # full 15x7 grid (absent_couplers empty in config)
    # planar grid consistency: interior node degree 4, edges only manhattan-1
    for i in range(spec.n):
        r, c = divmod(i, 7)
        expected_deg = 4 - (r in (0, 14)) - (c in (0, 6))
        assert spec.adj[i].sum() == expected_deg, f"qubit {i} deg mismatch"
    # corner distance: (0,0) -> (14,6) = 14 + 6 = 20
    assert spec.dist[0, 104] == 20


def test_tianyan287_calibration_ranges():
    spec = build_tianyan287_spec()
    calib = spec.calib
    # defect centers halve T1 (down to 10us); normal qubits >= 30us
    assert np.all(calib.t1 >= 10) and np.all(calib.t1 <= 200)
    assert np.all(calib.t2 >= 5)
    assert np.all(calib.err_1q >= 5e-5) and np.all(calib.err_1q <= 5e-2)
    assert np.all(calib.err_ro >= 1e-3) and np.all(calib.err_ro <= 1e-1)
    assert len(calib.err_2q) == 188
    med_t1 = float(np.median(calib.t1))
    assert 40 < med_t1 < 110  # around 72us documented median


def test_absent_couplers_removed():
    from qtrail.config import DeviceConfig
    cfg = DeviceConfig(name="grid-3x3-test", rows=3, cols=3,
                       absent_couplers=[[0, 1]], calibration="none")
    from qtrail.devices import build_grid_spec
    from qtrail.devices.calibration import CalibrationData
    edges = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (3, 6),
             (4, 5), (4, 7), (5, 8), (6, 7), (7, 8)]
    calib = CalibrationData(
        t1=np.full(9, 72.0), t2=np.full(9, 50.0),
        err_1q=np.full(9, 1e-3), err_ro=np.full(9, 8e-3),
        err_2q={(a, b): 3.8e-3 for (a, b) in edges})
    spec = build_grid_spec(name=cfg.name, rows=3, cols=3, calib=calib,
                           absent_couplers=cfg.absent_couplers)
    assert spec.adj[0, 1] == 0 and spec.adj[1, 0] == 0
    assert spec.adj.sum() == 22  # 11 edges left
    # distance now routes around: 0 -> 1 becomes 0-3-4-1 = 3
    assert spec.dist[0, 1] == 3
    assert (0, 1) in spec.absent_couplers


def test_synthetic_calibration_seed_reproducible():
    edges = [(i, i + 1) for i in range(8)]
    c1 = generate_synthetic_calibration(9, edges, seed=7)
    c2 = generate_synthetic_calibration(9, edges, seed=7)
    assert np.allclose(c1.t1, c2.t1)
    assert c1.err_2q == c2.err_2q
    c3 = generate_synthetic_calibration(9, edges, seed=8)
    assert not np.allclose(c1.t1, c3.t1)


def test_distance_matrix_lambda_interpolation():
    spec = build_grid3x3_spec()
    d0 = spec.distance_matrix(noise_lambda=0.0)
    d1 = spec.distance_matrix(noise_lambda=1.0)
    assert np.allclose(d0, spec.dist)
    assert np.allclose(d1, spec.noise_dist)
    half = spec.distance_matrix(noise_lambda=0.5)
    assert np.allclose(half, 0.5 * (d0 + d1))


# ---------------------------------------------------------------- 真机拓扑
def test_tianyan287_real_spec_geometry():
    """真机拓扑 spec：168 可用边、4 禁用比特、平台标签、真实校准快照。"""
    spec = build_tianyan287_real_spec()
    assert spec.n == 105
    assert spec.adj.sum() // 2 == 168          # 182 设计边 - 14 禁用耦合器
    assert spec.disabled_qubits == [21, 26, 34, 89]
    assert spec.qubit_labels is not None
    assert spec.qubit_labels[0] == 1 and spec.qubit_labels[-1] == 105
    # 禁用比特无任何边
    for q in spec.disabled_qubits:
        assert spec.adj[q].sum() == 0
    # 校准为平台快照（T1 均值 ~50us 量级，远小于合成 72us 口径）
    assert 30 < spec.calib.t1.mean() < 70
    # 距离/噪声距离有限且对角为 0
    assert np.all(np.isfinite(spec.dist))
    assert np.all(np.isfinite(spec.noise_dist))
    assert np.allclose(np.diag(spec.dist), 0)


def test_tianyan287_real_topology_differs_from_grid():
    """真机拓扑与评测口径（188 矩形网格）结构不同（D2 核验结论）。"""
    real = build_tianyan287_real_spec()
    grid = build_tianyan287_spec()
    real_edges = {(i, j) for i in range(105) for j in range(i + 1, 105)
                  if real.adj[i, j]}
    grid_edges = {(i, j) for i in range(105) for j in range(i + 1, 105)
                  if grid.adj[i, j]}
    assert len(real_edges) == 168
    assert len(grid_edges) == 188
    # 交集明显小于任一方（真机为弯曲排布，非矩形网格）
    inter = real_edges & grid_edges
    assert len(inter) < 150
