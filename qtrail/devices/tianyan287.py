"""Tianyan-287 device factory (Zuchongzhi-3.0-class 105-qubit 15x7 grid)."""
from __future__ import annotations

from pathlib import Path

from qtrail.config import DeviceConfig, NoiseConfig, load_device_config
from qtrail.devices.calibration import CalibrationData, generate_synthetic_calibration
from qtrail.devices.spec import DeviceSpec, build_grid_spec


def build_tianyan287_spec(cfg: DeviceConfig | None = None,
                          calib: CalibrationData | None = None) -> DeviceSpec:
    """Build the tianyan-287 DeviceSpec.

    Geometry comes from configs/device_tianyan287.yaml (single source of truth).
    Pass a CalibrationData to override the synthetic generator (e.g. the
    runtime adapter with live platform data).
    """
    cfg = cfg or load_device_config()
    n = cfg.rows * cfg.cols
    if calib is None:
        if cfg.calibration == "none":
            # degenerate calibration (all-median) so the stack still runs
            edges = _full_grid_edges(cfg.rows, cfg.cols, set(cfg.absent_couplers),
                                     set(cfg.disabled_qubits))
            calib = _median_calibration(n, edges)
        else:
            edges = _full_grid_edges(cfg.rows, cfg.cols, set(cfg.absent_couplers),
                                     set(cfg.disabled_qubits))
            calib = generate_synthetic_calibration(n, edges, seed=cfg.calibration_seed,
                                                   correlated_defects=cfg.correlated_defects)
    return build_grid_spec(name=cfg.name, rows=cfg.rows, cols=cfg.cols,
                           calib=calib, absent_couplers=cfg.absent_couplers,
                           disabled_qubits=cfg.disabled_qubits,
                           noise=cfg.noise)


def build_grid8x8_spec(seed: int = 0) -> DeviceSpec:
    """8x8 grid (CO-MAP paper's main evaluation device) for reproduction parity."""
    import numpy as np
    from qtrail.config import DeviceConfig
    cfg = DeviceConfig(name="grid-8x8", rows=8, cols=8, calibration_seed=seed,
                       correlated_defects=False)
    return build_tianyan287_spec(cfg)


def build_grid3x3_spec(seed: int = 0) -> DeviceSpec:
    """3x3 grid for unit/integration tests."""
    from qtrail.config import DeviceConfig
    cfg = DeviceConfig(name="grid-3x3", rows=3, cols=3, calibration_seed=seed,
                       correlated_defects=False)
    return build_tianyan287_spec(cfg)


def _full_grid_edges(rows: int, cols: int, absent: set, disabled: set) -> list:
    edges = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if i in disabled:
                continue
            for (dr, dc) in ((1, 0), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    j = rr * cols + cc
                    if j in disabled or (min(i, j), max(i, j)) in absent:
                        continue
                    edges.append((i, j))
    return edges


def _median_calibration(n: int, edges: list) -> CalibrationData:
    import numpy as np
    return CalibrationData(
        t1=np.full(n, 72.0, dtype=np.float64),
        t2=np.full(n, 50.0, dtype=np.float64),
        err_1q=np.full(n, 1e-3, dtype=np.float64),
        err_ro=np.full(n, 8.2e-3, dtype=np.float64),
        err_2q={(q0, q1): 3.8e-3 for (q0, q1) in edges},
    )
