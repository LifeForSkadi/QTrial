"""功能等价性验证 CLI：python -m qtrail.verify input.qasm [选项]

流程：
  1. 本地仿真验证：原线路 vs 映射后线路（≤20 比特精确态矢量；
     21-28 比特随机化采样；更大规模提示使用平台验证）
  2. 可选 --on-platform：经天衍平台真机验证（需 TIANYAN_LOGIN_KEY）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description="QTrial 功能等价性验证")
    ap.add_argument("input", help="原始 OpenQASM 2.0 线路")
    ap.add_argument("--mapped", default=None,
                    help="映射后 QASM（缺省时用 QTrial 管线现场生成）")
    ap.add_argument("--layout-json", default=None,
                    help="映射的最终布局 metrics.json（提取 final_layout）")
    ap.add_argument("--qcis", default=None,
                    help="pure_cli 输出的 QCIS（--on-platform 真机验证用，"
                         "需 --calibration live 生成的真实标签版本）")
    ap.add_argument("--on-platform", action="store_true",
                    help="天衍平台真机验证（需 TIANYAN_LOGIN_KEY / .tianyan_key）")
    ap.add_argument("--machine", default="tianyan-287")
    ap.add_argument("--shots", type=int, default=12000)
    ap.add_argument("--threshold", type=float, default=0.999)
    ap.add_argument("--max-qubits", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--output", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args(argv)

    from qtrail.utils.qasm_io import load_qasm2, strip_measurements
    from qtrail.verify.equivalence import verify_equivalence

    original = load_qasm2(args.input)
    original_clean, _ = strip_measurements(original)

    mapped = None
    final_layout = None
    if args.mapped:
        mapped = load_qasm2(args.mapped)
        mapped, _ = strip_measurements(mapped)
    else:
        # 用 QTrial 管线现场生成映射线路
        from qtrail.cli.map_cli import run as map_run
        from types import SimpleNamespace
        ns = SimpleNamespace(
            input=args.input, device=args.machine, output="out/verify_tmp",
            format="qasm,json", checkpoint=None,
            calibration="synthetic", decode="multistart", starts=10,
            postprocess=True, opt=1, baseline="", seed=42,
            dev=None, fast=False, ensemble=False, rule="fidelity",
            routing="sabre", cqlib_objective="size", cqlib_timeout=300,
            target_post=False, target_post_opt=3, target_post_seeds=8,
            target_post_top=3, submit=False, verbose=False)
        metrics = map_run(ns)
        out_name = Path(args.input).stem
        mapped = load_qasm2(Path("out/verify_tmp") / f"{out_name}_mapped.qasm")
        mapped, _ = strip_measurements(mapped)
        mpath = Path("out/verify_tmp") / f"{out_name}_metrics.json"
        if mpath.exists():
            with open(mpath, encoding="utf-8") as f:
                mj = json.load(f)
            fl = mj.get("final_layout")
            if isinstance(fl, dict):
                final_layout = {int(k): v for k, v in fl.items()}
    if args.layout_json:
        with open(args.layout_json, encoding="utf-8") as f:
            mj = json.load(f)
        fl = mj.get("final_layout")
        if isinstance(fl, dict):
            final_layout = {int(k): v for k, v in fl.items()}

    if mapped is None:
        print("错误：无法获得映射线路")
        return 2

    report = verify_equivalence(original_clean, mapped, layout=final_layout,
                                max_qubits=args.max_qubits,
                                n_samples=args.n_samples,
                                threshold=args.threshold)
    print(f"\n== 本地仿真验证 ==")
    print(f"  方法: {report['method']} | 保真度: {report['fidelity']:.6f} | "
          f"阈值: {report['threshold']}")
    print(f"  结论: {'✔ 功能等价' if report['equivalent'] else '✘ 不等价'}")

    if args.on_platform:
        from qtrail.verify.platform import TianyanVerifier
        verifier = TianyanVerifier(machine=args.machine)
        if not verifier.available():
            print("\n== 天衍平台验证 ==\n  （跳过）未配置 API key：设置环境变量 "
                  "TIANYAN_LOGIN_KEY 或项目根目录 .tianyan_key 文件后重试")
        elif not args.qcis:
            print("\n== 天衍平台验证 ==\n  （跳过）需 --qcis 指定 pure_cli "
                  "输出的 QCIS（建议用 --calibration live 生成的真实标签版本）")
        else:
            try:
                pres = verify_on_platform(original, args, verifier)
                report["platform"] = pres
                print(f"\n== 天衍平台验证 ==")
                print(f"  经典保真度: {pres['classical_fidelity']} | "
                      f"TVD: {pres['tvd']} | 结论: "
                      f"{'✔ 等价' if pres['equivalent'] else '✘ 不等价'}")
            except Exception as e:
                report["platform_error"] = str(e)
                print(f"\n== 天衍平台验证 ==\n  失败: {type(e).__name__}: "
                      f"{str(e)[:200]}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=float)
        print(f"报告 -> {args.output}")
    return 0 if report["equivalent"] else 1


def verify_on_platform(original, args, verifier):
    """真机验证：提交映射 QCIS → 计数回映射到逻辑空间 → 与理想分布对照。

    协议（避免依赖平台参照线路的未知布局）：
      理想分布 = 原线路的经典模拟测量分布（逻辑空间，≤20 测量比特）；
      实测分布 = 映射 QCIS 真机计数，按 QCIS 的 M 指令位序 + 真实比特
      标签 + final_layout 回映射到逻辑空间；
      比较经典保真度与 TVD（阈值 0.98 / 0.05，与 platform.py 一致）。
    """
    import json
    import re
    from pathlib import Path as P

    from qiskit.quantum_info import Statevector
    from qtrail.verify.platform import _compare_distributions

    # 1) 理想分布（逻辑空间，按升序逻辑比特编码键；qiskit 2.x 无
    #    probabilities(qubits=...) 子集参数 → 全分布手动边缘化）
    meas_q = []
    for inst in original.data:
        if inst.operation.name == "measure":
            meas_q.append(original.find_bit(inst.qubits[0]).index)
    if not meas_q:
        raise RuntimeError("原线路无测量指令，无法做分布对照")
    if len(meas_q) > 20:
        raise RuntimeError(f"测量比特数 {len(meas_q)} > 20，经典模拟不可行")
    n_tot = original.num_qubits
    full = Statevector(original.remove_final_measurements(
        inplace=False)).probabilities_dict()
    ideal = {}
    for key, p in full.items():
        bs = str(key).zfill(n_tot)          # qubit 0 = 最高位（qiskit 惯例）
        sub = "".join(bs[n_tot - 1 - q] for q in meas_q)
        key2 = "".join(sub[meas_q.index(l)] for l in sorted(meas_q))
        ideal[key2] = ideal.get(key2, 0.0) + float(p)

    # 2) 布局元数据（pure_cli metrics.json）
    if not args.layout_json:
        raise RuntimeError("需 --layout-json 指定 pure_cli 的 metrics.json")
    mj = json.loads(P(args.layout_json).read_text(encoding="utf-8"))
    final_layout = {int(lk): int(v) for lk, v in mj["final_layout"].items()}
    labels = mj.get("qubit_labels")
    qcis_text = P(args.qcis).read_text(encoding="utf-8")

    # 3) 提交真机 + 取回计数
    qid = verifier.submit(qcis_text, f"qtrail_verify_{P(args.qcis).stem}",
                          shots=args.shots)
    counts = verifier.fetch_results(qid)

    # 4) 物理计数 → 逻辑空间
    logical_counts = _relabel_counts(counts, qcis_text, final_layout, labels)

    # 5) 与理想分布对照
    return _compare_distributions(logical_counts, ideal, shots=args.shots,
                                  query_ids=[qid], machine=args.machine)


def _relabel_counts(counts: dict, qcis_text: str, final_layout: dict,
                    labels) -> dict:
    """平台计数（物理标签比特串）→ 逻辑空间计数（键 = 升序逻辑比特）。"""
    import re
    m_lines = re.findall(r"^M Q(\d+)", qcis_text, re.M)
    if not m_lines:
        raise RuntimeError("QCIS 无 M 测量指令")
    phys_labels = [int(x) for x in m_lines]
    label_to_idx = ({int(lab): i for i, lab in enumerate(labels)}
                    if labels is not None else None)
    inv_final = {}
    for logical, idx in final_layout.items():
        inv_final[idx] = logical
    n = len(m_lines)
    out = {}
    for bitstr, c in counts.items():
        bs = str(bitstr).zfill(n)[-n:]          # 高位对齐到第一个 M
        log_bits = {}
        ok = True
        for i, ch in enumerate(bs):
            phys = phys_labels[i]
            idx = label_to_idx[phys] if label_to_idx is not None else phys
            log = inv_final.get(idx)
            if log is None:
                ok = False
                break
            log_bits[log] = ch
        if not ok or len(log_bits) != n:
            continue
        key = "".join(log_bits[l] for l in sorted(log_bits))
        out[key] = out.get(key, 0) + int(c)
    return out


if __name__ == "__main__":
    sys.exit(main())
