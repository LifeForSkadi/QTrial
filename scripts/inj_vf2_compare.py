# -*- coding: utf-8 -*-
"""实验：我们的 RL 布局 + O3 全管线（含 VF2PostLayout）vs 原生感知O3。

机制背景：qiskit 预设管线在 initial_layout 指定时跳过两处 VF2PostLayout
（路由段 + 优化段 post_loop，get_vf2_limits 返回 None）——此前 inj 对比
是"我们布局+O3无VF2" vs "他们布局+O3有VF2"，不对称。本脚本手工复制
O3 管线（布局段可替换），补齐对称对照组 B。

自证校验：布局段传 None 时走 SabreLayout，结果应与原生 O3 高度一致
（验证复制保真度）；通过后我们的布局版才可信。

三组：
  A = 原生感知O3（layout_inject_* 的 native_* 字段）
  B = 我们布局 + O3 全管线含 VF2PostLayout（本脚本）
  C = 我们布局 + O3 无 VF2PostLayout（inj_* 字段）

输出：tables/compare_models/inj_vf2_rows.jsonl
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2, transpile
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.transpiler.passes import (ApplyLayout, BarrierBeforeFinalMeasurements,
                                      BasisTranslator,
                                      CommutativeCancellation,
                                      ContractIdleWiresInControlFlow,
                                      EnlargeWithAncilla,
                                      FilterOpNodes,
                                      FullAncillaAllocation,
                                      Optimize1qGatesDecomposition,
                                      RemoveDiagonalGatesBeforeMeasure,
                                      RemoveIdentityEquivalent,
                                      SabreLayout, SabreSwap, SetLayout,
                                      TwoQubitPeepholeOptimization,
                                      VF2PostLayout)

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
OUT = Path("tables/compare_models/inj_vf2_rows.jsonl")
BARRIER_LABEL = "qiskit.transpiler.internal.routing.protection.barrier"


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


def build_o3_pm(qc, target, cm, layout, seed=42, trials=20,
                with_vf2: bool = True):
    """手工复制 O3 预设管线（布局段=SetLayout(我们的布局)）。

    注意：qiskit 2.5 的 PassManager.run() 会清空 property set，跨 PM 无法
    传递 TranspileLayout 上下文——全部阶段必须位于同一个 PassManager 内
    （与预设一致），优化循环用固定轮数近似预设的最低点控制。
    with_vf2=False 时跳过两处 VF2PostLayout（用于对照 C 校验复制保真度）。
    """
    from qiskit.passmanager import ConditionalController
    pm = PassManager()
    # ---- layout 段
    pm.append(SetLayout(Layout({qc.qubits[i]: int(layout[i])
                                for i in range(qc.num_qubits)})))
    pm.append(FullAncillaAllocation(cm))
    pm.append(EnlargeWithAncilla())
    pm.append(ApplyLayout())
    # ---- routing 段（含第一处 VF2PostLayout；ApplyLayout 条件执行，
    # 与预设 ConditionalController 语义一致）
    pm.append(BarrierBeforeFinalMeasurements(label=BARRIER_LABEL))
    pm.append(SabreSwap(target, heuristic="decay", seed=seed, trials=trials))
    if with_vf2:
        pm.append(VF2PostLayout(target, seed=-1, call_limit=30_000_000,
                                max_trials=250_000, strict_direction=False))
        pm.append(ConditionalController(
            ApplyLayout(), condition=lambda ps: "post_layout" in ps))
    pm.append([FilterOpNodes(lambda n: n.label != BARRIER_LABEL)])
    # ---- translation 段
    pm.append(BasisTranslator(SessionEquivalenceLibrary,
                              target_basis=PLATFORM_BASIS))
    # ---- optimization 段（预设为最低点循环，固定 3 轮近似）
    loop_passes = [
        TwoQubitPeepholeOptimization(target, approximation_degree=1.0),
        RemoveIdentityEquivalent(approximation_degree=1.0, target=target),
        Optimize1qGatesDecomposition(basis=PLATFORM_BASIS, target=target),
        CommutativeCancellation(target=target),
        ContractIdleWiresInControlFlow(),
        RemoveDiagonalGatesBeforeMeasure(),
    ]
    for _ in range(3):
        for p in loop_passes:
            pm.append(p)
    # ---- post_loop：第二处 VF2PostLayout（exact_match 参数组）
    if with_vf2:
        pm.append(VF2PostLayout(target, seed=-1, call_limit=300_000,
                                max_trials=2_500, strict_direction=True))
        pm.append(ConditionalController(
            ApplyLayout(), condition=lambda ps: "post_layout" in ps))
    return pm.run(qc)


def build_o3_with_our_layout(qc, layout, target, cm, seed=42):
    return build_o3_pm(qc, target, cm, layout, seed)


def main():
    spec = build_tianyan287_spec(load_device_config())
    calib = spec.calib
    target = build_target(spec)
    cm = CouplingMap([(i, j) for i in range(spec.n)
                      for j in range(i + 1, spec.n) if spec.adj[i, j]])

    policy, ckpt = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                            map_location="cpu")
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    mapper = QiskitMapper(spec, policy=policy, cfg=cfg, dev="cpu", seed=42)

    # ---- 自证校验：复制管线（关 VF2）应复现 C（预设 initial_layout 注入）
    print("[validate] 复制管线（关VF2）vs 预设注入（C 口径）：", flush=True)
    for name in ("benchpress/feynman/adder_8",
                 "benchpress/feynman/gf2^7_mult"):
        src = find_file(name)
        qc0, _ = strip_measurements(qasm2.load(str(src)))
        tri = {i: i for i in range(qc0.num_qubits)}
        c_ref = transpile(qc0, target=target, basis_gates=PLATFORM_BASIS,
                          optimization_level=3, initial_layout=tri,
                          seed_transpiler=42)
        replica = build_o3_pm(qc0, target, cm, tri, seed=42, with_vf2=False)
        print(f"  {name}: 预设注入={c_ref.count_ops().get('cz', 0)} "
              f"复制关VF2={replica.count_ops().get('cz', 0)}", flush=True)

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
        if name in done:
            continue
        row = {"circuit": name, "n": ref.get("n_qubits"),
               "native_fidelity": ref.get("native_fidelity"),
               "inj_fidelity": ref.get("inj_fidelity")}
        src = find_file(name)
        if src is None:
            row["b_error"] = "file missing"
        else:
            try:
                qc0, _ = strip_measurements(qasm2.load(str(src)))
                res = mapper.map_circuit(qc0, circuit_id=name,
                                         optimization_level=1)
                layout = {i: int(res.layout[i]) for i in range(qc0.num_qubits)}
                t0 = time.time()
                routed = build_o3_with_our_layout(qc0, layout, target, cm,
                                                  seed=42)
                m = compute_metrics(routed, routed.count_ops().get("swap", 0),
                                    calib)
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


if __name__ == "__main__":
    main()
