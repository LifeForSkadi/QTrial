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
    ap.add_argument("--on-platform", action="store_true",
                    help="天衍平台真机验证（需 TIANYAN_LOGIN_KEY）")
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
            print("\n== 天衍平台验证 ==\n  （跳过）未配置 TIANYAN_LOGIN_KEY，"
                  "设置环境变量后加 --on-platform 重试")
        else:
            from qtrail.utils.qcis import circuit_to_qcis
            mapped_qcis = circuit_to_qcis(mapped)
            # 参照：原线路经平台编译（cqlib 自身管线）
            from qtrail.utils.qasm_io import qasm2_str
            reference_qcis = _platform_reference_qcis(original_clean, args.machine)
            try:
                pres = verifier.verify(mapped_qcis, reference_qcis,
                                       shots=args.shots)
                report["platform"] = pres
                print(f"\n== 天衍平台验证 ==")
                print(f"  经典保真度: {pres['classical_fidelity']} | "
                      f"TVD: {pres['tvd']} | 结论: "
                      f"{'✔ 等价' if pres['equivalent'] else '✘ 不等价'}")
            except Exception as e:
                report["platform_error"] = str(e)
                print(f"\n== 天衍平台验证 ==\n  失败: {e}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=float)
        print(f"报告 -> {args.output}")
    return 0 if report["equivalent"] else 1


def _platform_reference_qcis(qc, machine: str) -> str:
    """原线路经平台自身编译的 QCIS（真机验证参照）。"""
    try:
        from cqlib.mapping import mapping as cmap
        from cqlib.quantum_platform.base import BasePlatform
        from cqlib.utils.qasm_to_qcis import QasmToQcis
        from qtrail.utils.qasm_io import qasm2_str
        from cqlib import TianYanPlatform
        platform = TianYanPlatform(
            login_key=__import__("os").environ.get("TIANYAN_LOGIN_KEY", ""),
            machine_name=machine)
        qcis = QasmToQcis().convert_to_qcis(qasm2_str(qc))
        return cmap.transpile_qcis(qcis, platform)
    except Exception as e:
        raise RuntimeError(f"平台参照编译失败（请检查 cqlib 与 key）：{e}") from e


if __name__ == "__main__":
    sys.exit(main())
