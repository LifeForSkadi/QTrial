"""Tianyan platform adapter: live download_config -> DeviceSpec.

Guarded: without a login token (or on any failure) returns None so callers
fall back to the synthetic device spec. Never raises.
"""
from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger("qtrail")


def _scan_value(cfg: dict, *names: str):
    """Depth-2 case-insensitive key scan; returns first matching value."""
    names = tuple(n.lower() for n in names)

    def match(v):
        if isinstance(v, dict):
            for k, vv in v.items():
                if any(n in str(k).lower() for n in names):
                    return vv
            for vv in v.values():
                r = match(vv)
                if r is not None:
                    return r
        return None

    return match(cfg)


def _parse_float_list(v, n_expected: int) -> np.ndarray | None:
    """Parse {list | dict(label->val) | 'a,b,c' string} into a float array."""
    try:
        if isinstance(v, dict):
            vals = [float(x) for x in v.values()]
        elif isinstance(v, str):
            vals = [float(x) for x in v.replace("，", ",").split(",") if x.strip()]
        elif isinstance(v, (list, tuple)):
            vals = [float(x) for x in v]
        else:
            return None
        if len(vals) == 0:
            return None
        if len(vals) < n_expected:
            # pad with the mean (defensive)
            vals = vals + [float(np.mean(vals))] * (n_expected - len(vals))
        return np.array(vals[:n_expected], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def download_tianyan287_spec(machine: str = "tianyan-287",
                             token: str | None = None,
                             calibration_seed: int = 0):
    """Build a DeviceSpec from the live platform config.

    Returns None on any failure (no token / network / schema drift) so the
    caller falls back to the synthetic spec.
    """
    try:
        from cqlib import TianYanPlatform  # guarded import
        token = token or os.environ.get("TIANYAN_LOGIN_KEY", "") or \
            os.environ.get("CQLIB_LOGIN_KEY", "")
        if not token:
            log.warning("no Tianyan login key; falling back to synthetic device spec")
            return None
        platform = TianYanPlatform(login_key=token, machine_name=machine)
        cfg = platform.download_config(machine=machine)
        if not isinstance(cfg, dict):
            return None
    except Exception as e:
        log.warning("Tianyan download_config failed (%s); synthetic fallback", e)
        return None

    try:
        from qtrail.devices.calibration import CalibrationData
        from qtrail.devices.spec import build_grid_spec
        from qtrail.config import NoiseConfig

        overview = cfg.get("overview", {}) or {}
        coupler_map = overview.get("coupler_map", {}) or {}
        disabled_q = [int(x.lstrip("Q")) for x in
                      str(cfg.get("disabledQubits", "")).split(",") if x.strip()]
        disabled_c = [str(x).strip() for x in
                      str(cfg.get("disabledCouplers", "")).split(",") if x.strip()]

        # build qubit label set and edges
        labels = set()
        edges = []
        absent = []
        for c_name, pair in coupler_map.items():
            q0, q1 = str(pair[0]).lstrip("Q"), str(pair[1]).lstrip("Q")
            labels.add(int(q0))
            labels.add(int(q1))
            if str(c_name) in disabled_c or int(q0) in disabled_q or int(q1) in disabled_q:
                absent.append((int(q0), int(q1)))
            else:
                edges.append((min(int(q0), int(q1)), max(int(q0), int(q1))))
        labels = sorted(labels)
        n = len(labels)
        # remap platform labels -> internal indices
        index = {lab: i for i, lab in enumerate(labels)}
        edges = [(index[a], index[b]) for a, b in edges]
        absent = [(index[a], index[b]) for a, b in absent]

        if n == 0:
            return None

        # calibration fields (tolerant parse; synthetic fallback per field)
        t1 = _parse_float_list(_scan_value(cfg, "t1"), n)
        t2 = _parse_float_list(_scan_value(cfg, "t2"), n)
        err_1q = _parse_float_list(_scan_value(cfg, "gate_error", "single"), n)
        err_ro = _parse_float_list(_scan_value(cfg, "readout"), n)
        err_2q_raw = _scan_value(cfg, "two_gate_error", "2gate", "cz")
        err_2q = {}
        if isinstance(err_2q_raw, dict):
            for k, v in err_2q_raw.items():
                key = str(k).lstrip("C")
                if key.isdigit():
                    pair = coupler_map.get(k) or coupler_map.get(int(key))
                    if pair is not None:
                        q0, q1 = int(str(pair[0]).lstrip("Q")), int(str(pair[1]).lstrip("Q"))
                        err_2q[(min(q0, q1), max(q0, q1))] = float(v)
        elif isinstance(err_2q_raw, (list, str)):
            vals = _parse_float_list(err_2q_raw, len(edges))
            if vals is not None:
                for (a, b), v in zip(edges, vals):
                    err_2q[(a, b)] = float(v)
        if not err_2q:
            for (a, b) in edges:
                err_2q[(a, b)] = 3.8e-3

        # fill missing qubit-level fields with platform medians
        t1 = t1 if t1 is not None else np.full(n, 72.0)
        t2 = t2 if t2 is not None else np.full(n, 50.0)
        err_1q = err_1q if err_1q is not None else np.full(n, 1e-3)
        err_ro = err_ro if err_ro is not None else np.full(n, 8.2e-3)

        calib = CalibrationData(t1=t1, t2=t2, err_1q=err_1q, err_ro=err_ro,
                                err_2q=err_2q)

        # coordinates: qpu_coordinate if present, else grid guess from labels
        coords_raw = _scan_value(cfg, "coordinate", "position")
        coords = None
        if isinstance(coords_raw, dict):
            pts = []
            for lab in labels:
                v = coords_raw.get(str(lab), coords_raw.get(lab))
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    pts.append((float(v[0]), float(v[1])))
                else:
                    break
            if len(pts) == n:
                coords = np.array(pts, dtype=np.float32)

        from qtrail.devices.spec import DeviceSpec
        # build spec manually (labels are arbitrary; not necessarily a grid)
        import networkx as nx
        adj = np.zeros((n, n), dtype=np.int8)
        for a, b in edges:
            adj[a, b] = adj[b, a] = 1
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        dist = nx.floyd_warshall_numpy(g).astype(np.float32)
        noise_dist = dist.copy()  # topology-only noise fallback
        from qtrail.devices.spec import _featurize
        if coords is None:
            coords = np.zeros((n, 2), dtype=np.float32)
        node_features = _featurize(n, coords, adj, calib)

        return DeviceSpec(name=machine, n=n, adj=adj, coords=coords,
                          dist=dist, noise_dist=noise_dist,
                          node_features=node_features, calib=calib,
                          absent_couplers=absent,
                          qubit_labels=np.array(labels, dtype=np.int64),
                          disabled_qubits=sorted(set(index[q] for q in disabled_q
                                                     if q in index)))
    except Exception as e:
        log.warning("Tianyan config parse failed (%s); synthetic fallback", e)
        return None
