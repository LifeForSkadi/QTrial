"""Synthetic calibration data generator (Zuchongzhi-3.0 realistic distributions).

Reference values (arXiv 2412.11924):
  T1 ~ 72 us, T2 ~ 50 us, 1Q fidelity 99.90%, 2Q fidelity 99.62%,
  readout fidelity 99.18%.

The runtime adapter (adapter.py) consumes real download_config data with the
same field names, so the rest of the stack is source-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationData:
    """Per-qubit / per-edge calibration values.

    Arrays use internal device indices. err_2q is a dict {(q0, q1): err}.
    ``norm`` holds per-device min/max used for feature normalization so live
    data can be re-normalized with the live device's own statistics.
    """
    t1: np.ndarray            # [N] microseconds
    t2: np.ndarray            # [N] microseconds
    err_1q: np.ndarray        # [N] single-qubit gate error
    err_ro: np.ndarray        # [N] readout error
    err_2q: dict              # {(q0, q1): error}
    norm: dict = field(default_factory=dict)  # feature -> (lo, hi) per device

    def __post_init__(self):
        self.norm.setdefault("t1", (float(self.t1.min()), float(self.t1.max())))
        self.norm.setdefault("t2", (float(self.t2.min()), float(self.t2.max())))
        self.norm.setdefault("err_1q", (float(self.err_1q.min()), float(self.err_1q.max())))
        self.norm.setdefault("err_ro", (float(self.err_ro.min()), float(self.err_ro.max())))


def _trunc_normal(rng: np.random.Generator, mu, sigma, lo, hi, size):
    out = np.full(size, np.nan)
    need = size
    while need > 0:
        v = rng.normal(mu, sigma, need * 3)
        v = v[(v >= lo) & (v <= hi)]
        take = min(need, len(v))
        if take == 0:
            v = np.full(need, float(np.clip(mu, lo, hi)))
            take = need
        out[size - need:size - need + take] = v[:take]
        need -= take
    return out


def generate_synthetic_calibration(n: int, edges: list, seed: int = 0,
                                   correlated_defects: bool = True,
                                   defect_factor: float | None = None) -> CalibrationData:
    """Generate Zuchongzhi-3.0-realistic calibration for n qubits.

    Args:
        n: number of qubits.
        edges: list of undirected edge tuples (internal indices).
        seed: RNG seed.
        correlated_defects: seed ~2% spatially-correlated "hot" qubits with
            5-10x elevated errors (makes noise avoidance learnable).
        defect_factor: override the defect multiplier range (5.0-10.0).
            Used by sensitivity sweeps; None = default 5-10x.
    """
    rng = np.random.default_rng(seed)
    t1 = _trunc_normal(rng, 72.0, 15.0, 30.0, 200.0, n)
    t2 = _trunc_normal(rng, 50.0, 20.0, 10.0, 144.0, n)
    t2 = np.minimum(t2, 2.0 * t1)  # physical consistency

    err_1q = _trunc_normal(rng, 1e-3, 4e-4, 5e-5, 1e-2, n)
    err_ro = _trunc_normal(rng, 8.2e-3, 4e-3, 1e-3, 5e-2, n)

    err_2q = {}
    for (q0, q1) in edges:
        err_2q[(q0, q1)] = float(_trunc_normal(rng, 3.8e-3, 1.5e-3, 1e-3, 2e-2, 1)[0])

    if correlated_defects:
        n_defects = max(1, int(n * 0.02))
        defect_centers = rng.choice(n, size=n_defects, replace=False)
        # spread defects to neighbors of centers (spatially correlated regions)
        defected = set()
        for c in defect_centers:
            defected.add(c)
            for (q0, q1) in edges:
                if q0 == c:
                    defected.add(q1)
                elif q1 == c:
                    defected.add(q0)
        lo, hi = (5.0, 10.0) if defect_factor is None else (defect_factor, defect_factor)
        for q in defected:
            factor = rng.uniform(lo, hi)
            err_1q[q] = float(np.clip(err_1q[q] * factor, 1e-4, 5e-2))
            err_ro[q] = float(np.clip(err_ro[q] * factor, 1e-3, 1e-1))
            t1[q] = float(np.clip(t1[q] / 2.0, 10.0, 200.0))
        for e in list(err_2q):
            if e[0] in defected or e[1] in defected:
                err_2q[e] = float(np.clip(err_2q[e] * rng.uniform(lo, hi), 1e-3, 5e-2))

    return CalibrationData(t1=t1, t2=t2, err_1q=err_1q, err_ro=err_ro, err_2q=err_2q)
