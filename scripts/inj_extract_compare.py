# -*- coding: utf-8 -*-
"""实验 B：qiskit 自选布局（提取再注入，VF2PostLayout 同样被跳过）vs
我们的 RL 布局注入——O3 管线内"布局 vs 布局"的对称对比。

三组（同一 O3 管线，均无 VF2PostLayout 的布局差异）：
  A = 原生感知O3（自选布局 + VF2PostLayout，读 layout_inject_* 的 native_*）
  B = qiskit 自选布局提取再注入（本脚本：原生 O3 跑一次取布局 → 注入重跑）
  C = 我们的 RL 布局注入（读 inj_* 字段）

B 的机制：原生 O3 结果中提取 initial_layout（SabreLayout+VF2Layout 的
选择），作为 initial_layout 再跑一次 O3（此时 VF2PostLayout 被跳过，
与 C 同口径）→ B vs C 即为"qiskit 布局 vs 我们布局"的净对比。

输出：tables/compare_models/inj_extract_rows.jsonl + 汇总。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2, transpile

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.baselines import build_target
from qtrail.pipeline.mapper import Mapper as QiskitMapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.utils.qasm_io import PLATFORM_BASIS, strip_measurements

CKPT = "checkpoints/demo_laurel/tianyan-287_gat_combined_calib_dep0.1_t0.5_best.pt"
SRC_FILES = ["layout_inject_benchpress_rows.jsonl",
             "layout_inject_mqtbench_n15_rows.jsonl",
             "layout_inject_queko_BIGD_rows.jsonl"]
OUT = Path("tables/compare_models/inj_extract_rows.jsonl")


def find_file(name):
    if name.startswith("benchpress/"):
        parts = name.split("/")
        p = Path("data/benchpress/qasm") / parts[1] / f"{parts[2]}.qasm"
        return p if p.exists() else None
    if name.startswith("mqtbench/"):
        p = Path("data/mqtbench/stratified") / f"{name.split('/')[1]}.qasm"
        return p if p.exists() else None
    if name.startswith("queko_BIGD/"):
        p = Path("data/Queko/BIGD") / f"{name.split('/')[1]}.qasm"
        return p if p.exists() else None
    return None


def main():
    spec = build_tianyan287_spec(load_device_config())
    calib = spec.calib
    target = build_target(spec)

    policy, ckpt = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                            map_location="cpu")
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    mapper = QiskitMapper(spec, policy=policy, cfg=cfg, dev="cpu", seed=42)

    cases = []
    for fn in SRC_FILES:
        p = Path("tables/compare_models") / fn
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cases.append((r["circuit"], r))

    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["circuit"]] = r

    rows = list(done.values())
    for i, (name, ref) in enumerate(cases):
        if name in done and "b_fidelity" in done[name]:
            continue
        if name in done:
            rows = [r for r in rows if r["circuit"] != name]  # 失败行重试
        row = {"circuit": name, "n": ref.get("n_qubits"),
               "native_fidelity": ref.get("native_fidelity"),
               "inj_fidelity": ref.get("inj_fidelity")}
        src = find_file(name)
        if src is None:
            row["b_error"] = "file missing"
        else:
            try:
                qc0, _ = strip_measurements(qasm2.load(str(src)))
                # 原生 O3（提取其自选布局）
                t0 = time.time()
                native = transpile(qc0, target=target,
                                   basis_gates=PLATFORM_BASIS,
                                   optimization_level=3, seed_transpiler=42)
                # 过滤 ancilla 虚拟比特（原生 O3 的 initial_layout 含
                # 扩ancilla 的 105 项，重注入需仅保留原始虚拟比特）
                orig_bits = set(qc0.qubits)
                init = {v: p for v, p in
                        native.layout.initial_layout.get_virtual_bits().items()
                        if v in orig_bits}
                # B：qiskit 自选布局提取再注入（VF2PostLayout 被跳过）
                b = transpile(qc0, target=target, basis_gates=PLATFORM_BASIS,
                              optimization_level=3, initial_layout=init,
                              seed_transpiler=42)
                m = compute_metrics(b, b.count_ops().get("swap", 0), calib)
                row["b_fidelity"] = m["est_fidelity"]
                row["b_twoq"] = m["twoq_count"]
                row["b_depth"] = m["depth"]
                row["b_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["b_error"] = f"{type(e).__name__}: {str(e)[:60]}"
        rows.append(row)
        with open(OUT, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps({k: row[k] for k in
                          ("circuit", "b_fidelity", "native_fidelity",
                           "inj_fidelity", "b_error")
                          if k in row}, ensure_ascii=False), flush=True)
    print(f"done -> {OUT}")

    # 汇总
    ok = [r for r in rows if "b_fidelity" in r]
    if ok:
        for group in ("benchpress", "mqtbench", "queko"):
            sub = [r for r in ok if r["circuit"].startswith(group)]
            if not sub:
                continue
            b_win_c = sum(1 for r in sub if r["b_fidelity"] >
                          r.get("inj_fidelity", 0))
            b_win_a = sum(1 for r in sub if r["b_fidelity"] >
                          r.get("native_fidelity", 0))
            print(f"{group}: {len(sub)} 条 | B(qiskit布局注入) 胜 C(我们布局注入)"
                  f" {b_win_c}/{len(sub)} | B 胜 A(原生) {b_win_a}/{len(sub)}")


if __name__ == "__main__":
    main()
