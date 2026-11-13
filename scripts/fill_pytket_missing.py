# -*- coding: utf-8 -*-
"""补齐 pytket 修正的缺失行（重型 QV/Heisenberg 分片未跑完）。

- 无 measure 的线路：读出修正为空操作，直接继承旧表门级值
  （fidelity_with_ro = fidelity，_rr 指标 = 旧指标）；
- 有 measure 的线路：按 `fix_pytket_readout.run_tket` 实跑补读出。

输出追加到 `tables/final_report_v8/large_pytket_readout_shard8.jsonl`，
聚合脚本按 shard 通配读取。
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pipeline.mapper import Mapper as QiskitMapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pipeline.routing import decompose_to_platform
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.qasm_io import sanitize_qasm

from fix_pytket_readout import (circuit_text, index_benchpress, readout_log,
                                run_tket)

OUT = Path("tables/final_report_v8")


def main():
    spec = build_tianyan287_spec(load_device_config())
    calib = spec.calib
    cm = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
          if spec.adj[i, j]]
    qcm = QiskitMapper(spec).cm
    idx = index_benchpress()
    old = [json.loads(l) for l in
           Path("tables/final_report/large_fidelity_rows.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    buckets, _ = pickle.loads((OUT / "buckets_cache.pkl").read_bytes())
    texts = {n: t for n, t in buckets["large"]}
    done = set()
    for f in OUT.glob("large_pytket_readout_shard*.jsonl"):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if "tket_fidelity_with_ro" in r:
                    done.add(r["circuit"])

    out = []
    for r in old:
        name = r["circuit"]
        if name in done or "tket_fidelity" not in r:
            continue
        circ = parse_qasm(texts[name])
        text = circuit_text(name, idx)
        if text is None:
            continue
        qc = qasm2.loads(sanitize_qasm(text))
        n_meas = sum(1 for i in qc.data if i.operation.name == "measure")
        if n_meas == 0:
            # 无测量：读出项为空操作，继承旧基线（无需重跑重型线路）
            r["tket_fidelity_with_ro"] = r["tket_fidelity"]
            r["tket_logf_with_ro"] = math.log(
                max(float(r["tket_fidelity"]), 1e-300))
            r["tket_n_measured"] = 0
            r["tket_swaps_rr"] = r["tket_swaps"]
            r["tket_depth_rr"] = r["tket_depth"]
            r["tket_twoq_depth_rr"] = r["tket_twoq_depth"]
            r["tket_fidelity_gate_rr"] = r["tket_fidelity"]
        else:
            expanded, sc, nm = run_tket(qc, cm, calib)
            expanded = decompose_to_platform(expanded, qcm,
                                             optimization_level=1, seed=42)
            from fix_pytket_readout import gate_logf
            glog = gate_logf(expanded, calib)
            rlog = readout_log(expanded, calib)
            m = compute_metrics(expanded, sc, calib)
            r["tket_fidelity_gate_rr"] = math.exp(glog)
            r["tket_logf_with_ro"] = glog + rlog
            r["tket_fidelity_with_ro"] = math.exp(glog + rlog)
            r["tket_n_measured"] = nm
            r["tket_swaps_rr"] = int(sc)
            r["tket_twoq_count_rr"] = m["twoq_count"]
            r["tket_twoq_depth_rr"] = m["twoq_depth"]
            r["tket_depth_rr"] = m["depth"]
        out.append(r)
        print(f"{name}: meas={n_meas} fid_ro={r['tket_fidelity_with_ro']:.4g}",
              flush=True)
    p = OUT / "large_pytket_readout_shard8.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False, default=float)
                         + "\n" for r in out), encoding="utf-8")
    print(f"wrote {len(out)} rows -> {p}")


if __name__ == "__main__":
    main()
