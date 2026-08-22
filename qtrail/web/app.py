"""QTrial web demo: FastAPI backend + static single-page frontend.

Run: python -m qtrail.web  (then open http://127.0.0.1:8000)
"""
from __future__ import annotations

import io
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from qtrail.config import CHECKPOINTS_DIR, Config, load_device_config
from qtrail.utils.qasm_io import (append_measurements, load_qasm2,
                                  strip_measurements, write_qasm2)

log = logging.getLogger("qtrail.web")
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="QTrial 量子线路映射演示", version="0.1.0")

_lock = threading.Lock()
_cache: dict = {}


def get_mapper(device: str, checkpoint: str, seed: int):
    key = (device, checkpoint, seed)
    with _lock:
        if key not in _cache:
            import numpy as np
            import torch
            from qtrail.devices import (build_grid3x3_spec, build_grid8x8_spec,
                                        build_tianyan287_spec)
            from qtrail.models import QAPolicy
            from qtrail.pipeline.mapper import Mapper

            dev_cfg = load_device_config()
            if device == "tianyan-287":
                spec = build_tianyan287_spec(dev_cfg)
            elif device == "grid-8x8":
                spec = build_grid8x8_spec()
            else:
                spec = build_grid3x3_spec()

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            policy = None
            if checkpoint and checkpoint != "auto":
                p = Path(checkpoint)
                if p.exists():
                    policy, ckpt = QAPolicy.load_checkpoint(str(p), device_n=spec.n,
                                                            map_location=dev)
                    policy.eval()
            elif checkpoint == "auto":
                best = _auto_checkpoint(device)
                if best:
                    policy, ckpt = QAPolicy.load_checkpoint(best, device_n=spec.n,
                                                            map_location=dev)
                    policy.eval()

            cfg = Config()
            cfg.device = dev_cfg
            mapper = Mapper(spec, policy=policy, cfg=cfg, dev=dev, seed=seed)
            _cache[key] = mapper
    return _cache[key]


def _auto_checkpoint(device: str) -> str | None:
    if not CHECKPOINTS_DIR.exists():
        return None
    cks = sorted(CHECKPOINTS_DIR.glob(f"{device}_*_best.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cks[0]) if cks else None


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/devices")
def devices():
    dev_cfg = load_device_config()
    return {
        "tianyan-287": {"qubits": dev_cfg.rows * dev_cfg.cols,
                        "rows": dev_cfg.rows, "cols": dev_cfg.cols,
                        "calibration": dev_cfg.calibration},
        "grid-8x8": {"qubits": 64, "rows": 8, "cols": 8, "calibration": "synthetic"},
        "grid-3x3": {"qubits": 9, "rows": 3, "cols": 3, "calibration": "synthetic"},
    }


@app.get("/api/checkpoints")
def checkpoints():
    out = []
    if CHECKPOINTS_DIR.exists():
        for p in sorted(CHECKPOINTS_DIR.glob("*.pt")):
            if "best" in p.name:
                out.append({"name": p.name, "path": str(p),
                            "size_mb": round(p.stat().st_size / 1e6, 1)})
    return out


@app.post("/api/map")
async def map_circuit(
    qasm_text: str = Form(""),
    file: UploadFile = File(None),
    device: str = Form("tianyan-287"),
    checkpoint: str = Form("auto"),
    seed: int = Form(42),
    compare: bool = Form(True),
):
    """Map a circuit (pasted QASM or uploaded file); optionally run baselines."""
    if file is not None and file.filename:
        qasm_text = (await file.read()).decode("utf-8", "ignore")
    if not qasm_text.strip():
        return {"error": "请粘贴 QASM 线路或上传 .qasm 文件"}

    t0 = time.time()
    try:
        qc = load_qasm2(io.StringIO(qasm_text))  # qasm2.load accepts text streams
    except Exception:
        # qasm2.load may not accept StringIO; fall back to temp file
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".qasm", delete=False,
                                         encoding="utf-8") as f:
            f.write(qasm_text)
            tmp = f.name
        try:
            qc = load_qasm2(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)

    qc_clean, removed = strip_measurements(qc)
    mapper = get_mapper(device, checkpoint, seed)

    from qtrail.pipeline.baselines import sabre_swap_count, sabre_transpile
    from qtrail.pipeline.metrics import compute_metrics

    result = mapper.map_circuit(qc_clean, circuit_id="web")
    final_qc = append_measurements(result.routed_qc, removed, result.final_layout)
    buf = io.StringIO()
    from qiskit import qasm2
    buf.write(qasm2.dumps(final_qc))
    from qtrail.utils.qcis import circuit_to_qcis
    qcis = circuit_to_qcis(final_qc)

    out = {
        "ok": True,
        "method": result.method,
        "warnings": result.warnings,
        "n_qubits": qc_clean.num_qubits,
        "device": device,
        "metrics": result.metrics,
        "layout": {str(k): v for k, v in result.layout.items()},
        "final_layout": {str(k): v for k, v in result.final_layout.items()},
        "wall_s": round(time.time() - t0, 3),
        "qasm": buf.getvalue(),
        "qcis": qcis,
    }
    if compare:
        baselines = {}
        for o in (1, 3):
            try:
                sc, _ = sabre_swap_count(qc_clean, mapper.cm, optimization_level=o,
                                         seed=seed)
                final_b = sabre_transpile(qc_clean, mapper.cm, optimization_level=o,
                                          seed=seed)
                m = compute_metrics(final_b, sc, mapper.spec.calib)
                baselines[f"qiskit-o{o}"] = {
                    "swaps": sc, "twoq": m["twoq_count"], "depth": m["depth"],
                    "twoq_depth": m["twoq_depth"], "fidelity": m["est_fidelity"],
                }
            except Exception as e:
                baselines[f"qiskit-o{o}"] = {"error": str(e)}
        out["baselines"] = baselines
    return out


@app.get("/api/topology/{device}")
def topology(device: str):
    """Device geometry for frontend visualization."""
    mapper = get_mapper(device, "auto", 42)
    spec = mapper.spec
    nodes = [{"id": i, "row": float(spec.coords[i, 0]), "col": float(spec.coords[i, 1]),
              "t1": float(spec.calib.t1[i]), "err1q": float(spec.calib.err_1q[i])}
             for i in range(spec.n)]
    edges = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
             if spec.adj[i, j]]
    return {"n": spec.n, "nodes": nodes, "edges": edges,
            "t1_bounds": [float(spec.calib.t1.min()), float(spec.calib.t1.max())]}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
