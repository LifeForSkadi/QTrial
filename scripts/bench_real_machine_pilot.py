# -*- coding: utf-8 -*-
"""真机保真度验证试点：一条 QUEKO BIGD 20 比特线路 × 2 方法。

QUEKO 线路只含 x/cx——从全零初态出发的理想输出是**单一基态**（经典
可算），因此实验保真度 = 实测计数落在理想基态的占比，无采样噪声问题，
且跨方法直接可比。

方法 A：本方案布局 + O1 路由（QiskitMapper，fidelity 规则）
方法 B：噪声感知 O3（qiskit 原生管线，同种子）
各提交真机 12000 shots，取回计数与理想基态比对：
  实验保真度 = counts[理想基态] / 总 shots；另有 TVD 与 top-5 计数诊断。
  位串位序（MSB/LSB 先行）两种约定都试，取命中的较好者并记录所用约定。
输出：tables/real_machine/pilot.jsonl

用法：python scripts/bench_real_machine_pilot.py [--machine tianyan-287]
      （287 不可用时自动回退 tianyan-176 66 比特备选机）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit import transpile

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.baselines import build_target
from qtrail.pipeline.mapper import Mapper
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import PLATFORM_BASIS, load_qasm2, strip_measurements
from qtrail.utils.qcis import circuit_to_qcis

DEFAULT_CKPT = ("checkpoints/demo_laurel/"
                "tianyan-287_gat_combined_calib_dep0.1_t0.5_best.pt")
SHOTS = 12000


def pick_circuit():
    """选一条 20 比特中等密度的 QUEKO BIGD 线路。"""
    cands = list(iter_queko_files("BIGD"))
    for pref in (".5D1_.5D2_", ".3D1_.3D2_", ".7D1_.1D2_"):
        for p in cands:
            if p.name.startswith("20QBT_") and pref in p.name:
                return p
    return next(p for p in cands if p.name.startswith("20QBT_"))


def ideal_bitstring(ops) -> tuple:
    """x/cx 线路在 |0...0> 下的理想输出基态（经典模拟）。返回 (bits, n)。"""
    bits, n = 0, 0
    for name, qs in ops:
        if name == "x":
            bits ^= 1 << qs[0]
            n = max(n, qs[0] + 1)
        elif name == "cx":
            if (bits >> qs[0]) & 1:
                bits ^= 1 << qs[1]
            n = max(n, qs[0] + 1, qs[1] + 1)
        elif name == "barrier":
            continue
        else:
            raise ValueError(f"unexpected gate {name}")
    return bits, n


def add_measures(qc, phys: list) -> list:
    """按升序物理位序回挂测量（每条一位），返回测量顺序。"""
    from qiskit.circuit import ClassicalRegister
    order = sorted(phys)
    creg = ClassicalRegister(len(order), "m")
    qc.add_register(creg)
    for k, p in enumerate(order):
        qc.measure(qc.qubits[p], creg[k])
    return order


def qcis_with_labels(qc, spec) -> str:
    """平台基线路 → QCIS 文本，使用设备真实比特标签（qubit_labels）。"""
    labels = spec.qubit_labels
    lines = []
    for inst in qc.data:
        name = inst.operation.name
        qs = [qc.find_bit(q).index for q in inst.qubits]
        lab = lambda i: (f"Q{int(labels[i])}" if labels is not None
                         else f"Q{i}")
        if name == "x":
            lines.append(f"X {lab(qs[0])}")
        elif name == "sx":
            lines.append(f"RZ {lab(qs[0])} 1.5707963267948966")
            lines.append(f"X {lab(qs[0])}")
            lines.append(f"RZ {lab(qs[0])} -1.5707963267948966")
        elif name == "rz":
            lines.append(f"RZ {lab(qs[0])} {float(inst.operation.params[0])}")
        elif name == "cz":
            lines.append(f"CZ {lab(qs[0])} {lab(qs[1])}")
        elif name == "measure":
            # 平台要求的测量指令为 M（2026-08-22 真机校验修正口径）
            lines.append(f"M {lab(qs[0])}")
        elif name in ("barrier", "reset"):
            continue
        else:
            raise ValueError(f"gate '{name}' has no QCIS form")
    return "\n".join(lines)


def logical_counts(raw: dict, order: list, inv_layout: dict, n: int) -> dict:
    """实测计数（按测量位序）→ 逻辑计数（q0 首位字符串键）。"""
    out = {}
    for bitstr, c in raw.items():
        s = bitstr.zfill(len(order))
        bits = {}
        for k, p in enumerate(order):
            if k < len(s):
                bits[inv_layout.get(p, p)] = s[k]
        key = "".join(bits.get(i, "0") for i in range(n))
        out[key] = out.get(key, 0) + c
    return out


def hit_fidelity(raw: dict, order: list, inv_layout: dict, n: int,
                 target: str) -> tuple:
    """两种位序约定（MSB/LSB 先行）都试，返回 (fidelity, 约定, 逻辑计数)。"""
    best = (0.0, None, None)
    for flip in (False, True):
        norm = {}
        for bitstr, c in raw.items():
            k = bitstr[::-1] if flip else bitstr
            norm[k] = norm.get(k, 0) + c
        lc = logical_counts(norm, order, inv_layout, n)
        total = sum(lc.values()) or 1
        fid = lc.get(target, 0) / total
        if fid > best[0]:
            best = (fid, "lsb_first" if flip else "msb_first", lc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="tianyan-287")
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shots", type=int, default=SHOTS)
    args = ap.parse_args()

    qpath = pick_circuit()
    print(f"circuit: {qpath.name}", flush=True)
    text = qpath.read_text(encoding="utf-8")
    # 理想基态从原始 QASM 文本解析（x/cx 经典可算；项目加载器会把
    # x 归一化为 u 门，此处不用加载后的线路做理想模拟）
    ops_list = []
    for m in re.finditer(r"^(x|cx) q\[(\d+)\](?:,q\[(\d+)\])?;",
                         text, re.MULTILINE):
        if m.group(1) == "x":
            ops_list.append(("x", [int(m.group(2))]))
        else:
            ops_list.append(("cx", [int(m.group(2)), int(m.group(3))]))
    qc, _ = strip_measurements(load_qasm2(qpath))
    target_bits, n = ideal_bitstring(ops_list)
    target_str = "".join(str((target_bits >> i) & 1) for i in range(n))
    print(f"ideal target (q0-first): {target_str}", flush=True)

    # 实测拓扑（优先目标机器的 live 配置；失败回退合成 287 并告警）
    from qtrail.devices.adapter import download_tianyan287_spec
    from qtrail.devices.spec import usable_spec
    spec = download_tianyan287_spec(machine=args.machine)
    if spec is None:
        print(f"[warn] {args.machine} live 配置不可用，回退合成 287 —— "
              f"该拓扑无法在 {args.machine} 执行", flush=True)
        spec = build_tianyan287_spec()
    n_before, bad = spec.n, spec.disabled_qubits
    spec = usable_spec(spec)
    print(f"device: n={n_before} disabled={bad} -> usable n={spec.n} "
          f"edges={len(spec.edges)}", flush=True)
    target_obj = build_target(spec)

    policy, ckpt = QAPolicy.load_checkpoint(
        DEFAULT_CKPT, device_n=spec.n, map_location=args.dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    mapper = Mapper(spec, policy=policy, cfg=cfg, dev=args.dev,
                    seed=args.seed, selection_rule="fidelity")

    # ---- 方法 A：本方案布局 + O1 路由
    res = mapper.map_circuit(qc, circuit_id=qpath.stem,
                             optimization_level=1)
    mapped_a = res.routed_qc.copy()
    inv_a = {v: k for k, v in res.final_layout.items()}
    order_a = add_measures(mapped_a, list(res.final_layout.values()))
    qcis_a = qcis_with_labels(mapped_a, spec)
    print(f"method A: swaps={res.swap_count} "
          f"twoq={res.metrics.get('twoq_count')} "
          f"depth={res.metrics.get('depth')}", flush=True)

    # ---- 方法 B：噪声感知 O3
    routed_b = transpile(qc, target=target_obj, basis_gates=PLATFORM_BASIS,
                         optimization_level=3, seed_transpiler=args.seed)
    # 回映射：转译后线路的最终布局 → 原线路逻辑比特（输出虚拟比特
    # 是转译线路自己的比特对象，需经 input_qubit_mapping 还原）
    inv_b = {}
    try:
        in_map = routed_b.layout.input_qubit_mapping
        final_idx = routed_b.layout.final_index_layout(filter_ancillas=True)
        out_of = {qc.find_bit(v_in).index: routed_b.find_bit(v_out).index
                  for v_in, v_out in in_map.items()
                  if v_in in qc.qubits}
        for v_out_idx, p in final_idx.items():
            for lin, vidx in out_of.items():
                if vidx == v_out_idx:
                    inv_b[int(p)] = lin
    except Exception:
        init_layout = routed_b.layout.initial_layout
        inv_b = {int(p): qc.find_bit(v).index
                 for v, p in init_layout.get_virtual_bits().items()
                 if v in qc.qubits}
    order_b = add_measures(routed_b, sorted(inv_b.keys()))
    qcis_b = qcis_with_labels(routed_b, spec)
    print(f"method B: depth={routed_b.depth()} 测量比特数={len(inv_b)}",
          flush=True)

    # ---- 真机提交（287 → 176 回退）
    from qtrail.verify.platform import TianyanVerifier
    results = []
    for name, qcis, inv, order in (("ours_o1", qcis_a, inv_a, order_a),
                                   ("aware_o3", qcis_b, inv_b, order_b)):
        qid, ver = None, None
        for machine in (args.machine, "tianyan176"):
            try:
                ver = TianyanVerifier(machine=machine)
                plat = ver._get_platform()
                try:
                    plat.qcis_check_regular(qcis)
                    print(f"[{name}] qcis_check_regular OK on {machine}",
                          flush=True)
                except Exception as e:
                    print(f"[{name}] {machine} 规则校验失败: "
                          f"{type(e).__name__}: {str(e)[:100]}", flush=True)
                    continue
                qid = ver.submit(qcis, f"pilot_{name}", args.shots)
                print(f"[{name}] submitted to {machine}: {qid}", flush=True)
                break
            except Exception as e:
                print(f"[{name}] {machine} submit failed: "
                      f"{type(e).__name__}: {str(e)[:80]}", flush=True)
        if qid is None:
            results.append({"method": name, "error": "submit failed"})
            continue
        try:
            counts = ver.fetch_results(qid, max_wait_time=3600)
        except Exception as e:
            results.append({"method": name, "error":
                            f"fetch failed: {type(e).__name__}: "
                            f"{str(e)[:80]}"})
            continue
        total = sum(counts.values()) or 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        fid, conv, lc = hit_fidelity(counts, order, inv, n, target_str)
        keys = sorted(set(lc) | {target_str})
        p = np.array([float(lc.get(k, 0)) for k in keys])
        p = p / max(p.sum(), 1e-12)
        q = np.array([1.0 if k == target_str else 0.0 for k in keys])
        tvd = float(np.abs(p - q).sum() / 2.0)
        results.append({
            "method": name, "machine": ver.machine, "query_id": qid,
            "shots_total": total, "experimental_fidelity": round(fid, 4),
            "tvd": round(tvd, 4), "bit_order": conv, "top5": top,
            "target": target_str,
        })
        print(f"[{name}] experimental_fidelity={fid:.4f} tvd={tvd:.4f} "
              f"(order={conv}) top5={top}", flush=True)

    out = Path("tables/real_machine/pilot.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"circuit": qpath.name,
                            "ideal_target": target_str,
                            "results": results}, ensure_ascii=False) + "\n")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
