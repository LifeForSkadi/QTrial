# -*- coding: utf-8 -*-
"""v8 发布前审计（2026-11-13）：分层抽样复核真机可行性与语义正确性。

回答的问题：第 3–6 项改动（A2 自研 VW 合成、自研 VF2++ 嵌入、候选
类型化 / Inst 冻结）之后，生产管线产物是否仍然
  ① 真机合法（门集 ⊆ {rz,sx,x,cz,measure}、cz 相邻、无禁用比特、
     无 swap 残留、measure 落在 final_layout 对应物理位）；
  ② 语义正确（紧凑子空间 Statevector：U(C')|0⟩ = Π_fin·U(C)|0⟩）。

抽样：每层按等距抽样（确定性，无随机），默认 small 12 / medium 12 /
large 6，覆盖 benchpress + MQTBench + QUEKO 三源。large 层受 2^k
限制只做合法性与 measure 复核（语义由组件级证明 + 全量合法性背书）。

用法：
  python scripts/audit_v8_release.py [--small 12] [--medium 12]
      [--large 6] [--timeout 900]

输出：tables/release_audit/v8_audit_rows.jsonl + summary.md
断点续跑（按 circuit 名去重）。
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.circuit import Inst
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm

OUT = Path("tables/release_audit")
CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
MAX_SEM_K = 16  # 2^16 态矢量（含 Qiskit 开销）安全上限


def run_with_timeout(fn, timeout):
    box = {}

    def target():
        try:
            box["res"] = fn()
        except Exception as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, "timeout"
    if "err" in box:
        raise box["err"]
    return box.get("res"), None


def check_legality(final_qc, spec):
    """三重扫描：门集 / 相邻性 / 禁用比特 / 越界 / swap 残留 / measure。"""
    problems = []
    names = set()
    disabled = set(spec.disabled_qubits)
    for idx, inst in enumerate(final_qc.ops):
        names.add(inst.name)
        qs = [int(q) for q in inst.qubits]
        if inst.name == "swap":
            problems.append(f"swap残留@{idx}{qs}")
        if any(q >= spec.n for q in qs):
            problems.append(f"越界{qs}@{idx}")
        if any(q in disabled for q in qs):
            problems.append(f"禁用{qs}@{idx}")
        if inst.name == "cz" and not spec.adj[qs[0], qs[1]]:
            problems.append(f"cz非相邻{qs}@{idx}")
    if not names <= {"rz", "sx", "x", "cz", "measure"}:
        problems.append(f"门集{names - {'rz', 'sx', 'x', 'cz', 'measure'}}")
    for idx, m in enumerate(final_qc.measures):
        q = int(m.qubits[0])
        if q >= spec.n:
            problems.append(f"measure{idx}越界{q}")
        if q in disabled:
            problems.append(f"measure{idx}禁用{q}")
    return problems


def to_qc(ops, n):
    qc = QuantumCircuit(n)
    for inst in ops:
        qs = [int(q) for q in inst.qubits]
        if inst.name == "rz":
            qc.rz(inst.params[0], qs[0])
        elif inst.name == "sx":
            qc.sx(qs[0])
        elif inst.name == "x":
            qc.x(qs[0])
        elif inst.name == "cz":
            qc.cz(qs[0], qs[1])
        elif inst.name == "swap":
            qc.swap(qs[0], qs[1])
        elif inst.name == "cx":
            qc.cx(qs[0], qs[1])
        elif inst.name == "u3":
            qc.u(*inst.params, qs[0])
        elif inst.name == "u1":
            qc.p(inst.params[0], qs[0])
        elif inst.name == "u2":
            qc.u(np.pi / 2, inst.params[0], inst.params[1], qs[0])
        elif inst.name == "h":
            qc.h(qs[0])
        elif inst.name == "id":
            continue
        else:
            raise ValueError(f"非预期门 {inst.name}")
    return qc


def pick_even(items, k):
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return list(range(len(items)))
    step = len(items) / float(k)
    return [int(i * step) for i in range(k)]


def build_index():
    """题面名 -> (来源, 路径)。只扫文件名，不解析，秒级完成。"""
    idx = {}
    for p in Path("data/benchpress/qasm").rglob("*.qasm"):
        idx[f"bp_{p.parent.name}_{p.stem}"] = ("bp", p)
    for p in Path("data/mqtbench/stratified").glob("*.qasm"):
        idx[p.stem] = ("mqt", p)
    for p in iter_queko_files("BIGD"):
        idx[p.stem] = ("queko", p)
    return idx


def load_text(kind: str, p: Path) -> str:
    text = p.read_text(encoding="utf-8", errors="replace")
    return text if kind == "bp" else sanitize_qasm(text)


def v8_names(bucket: str) -> list[str]:
    """v8 已完成线路名（排除 4 条已知超时），按名排序保证可复现。"""
    p = Path(f"tables/final_report_v8/{bucket}_fidelity_rows.jsonl")
    if not p.exists():
        return []
    names = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "qt_fidelity" in r:
                names.append(r["circuit"])
    return sorted(names)


def main():
    import argparse
    t_start = time.time()
    print("stage: start", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", type=int, default=12)
    ap.add_argument("--medium", type=int, default=12)
    ap.add_argument("--large", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"stage: spec (dev={dev})", flush=True)
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    print(f"stage: index ({time.time() - t_start:.1f}s)", flush=True)
    index = build_index()
    want = {"small": args.small, "medium": args.medium, "large": args.large}
    picks = []
    for b in ("small", "medium", "large"):
        names = [n for n in v8_names(b) if n in index]
        for i in pick_even(names, want[b]):
            name = names[i]
            kind, p = index[name]
            picks.append((b, (name, load_text(kind, p))))
    print(f"审计样本: " + " / ".join(
        f"{b} {sum(1 for x in picks if x[0] == b)}" for b in want), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / "v8_audit_rows.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["circuit"])

    print(f"stage: run ({time.time() - t_start:.1f}s)", flush=True)
    for bucket, (name, ptext) in picks:
        if name in done:
            continue
        print(f"  -> {name}", flush=True)
        circ = parse_qasm(ptext)
        row = {"circuit": name, "bucket": bucket, "n": circ.n}
        t0 = time.time()
        try:
            torch.manual_seed(42)
            mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev,
                                seed=42, selection_rule="fidelity",
                                use_post=True, pipeline="v5", routing_seeds=16)
            res, terr = run_with_timeout(
                lambda: mapper.map_circuit(circ, circuit_id=name),
                args.timeout)
            if terr is not None:
                row["error"] = f"timeout_{args.timeout:.0f}s"
            else:
                final_qc = res["routed_circuit"]
                m = res["metrics"]
                fin = {int(k): int(v) for k, v in res["final_layout"].items()}
                probs = check_legality(final_qc, spec)
                row["legal"] = not probs
                if probs:
                    row["violations"] = probs[:8]
                # measure 语义：逻辑 q 的测量必须落在 fin[q]
                want_m = {(int(mm.qubits[0]), tuple(mm.cbits))
                          for mm in circ.measures}
                got_m = {(int(mm.qubits[0]), tuple(mm.cbits))
                         for mm in final_qc.measures}
                exp_m = {(fin.get(int(mm.qubits[0]), int(mm.qubits[0])),
                          tuple(mm.cbits)) for mm in circ.measures}
                row["measure"] = "ok" if (not want_m or exp_m == got_m) else "bad"
                if not want_m:
                    row["measure"] = "none"
                # 端到端语义（紧凑子空间）
                touched = sorted({int(q) for i in final_qc.ops
                                  for q in i.qubits})
                all_q = sorted(set(touched) | {fin.get(i, i)
                                               for i in range(circ.n)})
                k = len(all_q)
                if k <= MAX_SEM_K:
                    cmap = {q: j for j, q in enumerate(all_q)}
                    emb = [Inst(i.name,
                                tuple(cmap[fin.get(int(q), int(q))]
                                      for q in i.qubits), i.params)
                           for i in circ.ops]
                    fin2 = [Inst(i.name, tuple(cmap[int(q)] for q in i.qubits),
                                 i.params) for i in final_qc.ops]
                    sv1 = np.array(Statevector(to_qc(emb, k)).data)
                    sv2 = np.array(Statevector(to_qc(fin2, k)).data)
                    row["sem_fid"] = float(abs(np.vdot(sv1, sv2)) ** 2)
                    row["sem_k"] = k
                else:
                    row["sem_fid"] = None
                    row["sem_k"] = k
                row["twoq"] = m["twoq_count"]
                row["depth"] = m["depth"]
                row["fid"] = m["est_fidelity"]
                row["wall"] = round(time.time() - t0, 1)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:80]}"
        with out_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        ok = row.get("legal") and row.get("measure") in ("ok", "none") \
            and (row.get("sem_fid") is None or row["sem_fid"] > 1 - 1e-8) \
            and not row.get("error")
        print(f"[{bucket}] {name}: {'OK' if ok else 'FAIL'} "
              f"legal={row.get('legal')} measure={row.get('measure')} "
              f"sem={row.get('sem_fid')} ({row.get('wall', '-')}s)", flush=True)

    # ---- 汇总
    rows = [json.loads(l) for l in out_file.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    ok_rows = [r for r in rows if r.get("legal") and not r.get("error")]
    bad = [r for r in rows if not (r.get("legal") and not r.get("error"))]
    sem = [r["sem_fid"] for r in rows if r.get("sem_fid") is not None]
    lines = ["# v8 发布前审计（真机可行性 + 语义）\n\n",
             f"- 样本：{len(rows)} 条（small "
             f"{sum(r['bucket'] == 'small' for r in rows)} / medium "
             f"{sum(r['bucket'] == 'medium' for r in rows)} / large "
             f"{sum(r['bucket'] == 'large' for r in rows)}）\n",
             f"- 合法：{len(ok_rows)}/{len(rows)}；违规/错误：{len(bad)}\n",
             f"- measure 语义："
             f"{sum(r.get('measure') in ('ok', 'none') for r in rows)}/"
             f"{len(rows)} 通过\n"]
    if sem:
        lines.append(f"- 端到端语义（k≤{MAX_SEM_K}）：{len(sem)} 条，"
                     f"最小 fid {min(sem):.9f}，中位 "
                     f"{statistics.median(sem):.9f}\n")
    if bad:
        lines.append("\n## 失败明细\n\n")
        for r in bad:
            lines.append(f"- `{r['circuit']}`: {r.get('violations', r.get('error'))}\n")
    (OUT / "summary.md").write_text("".join(lines), encoding="utf-8")
    print("written", OUT / "summary.md", flush=True)


if __name__ == "__main__":
    main()
