# -*- coding: utf-8 -*-
"""D2 验证实验：通过 API 拉取天衍真机基础信息（不下发任何任务）。

只做 download_config（机器元数据/拓扑/校准），保存到 data/tianyan_probe/（gitignored）。
打印概览：比特数、耦合器数、禁用比特/耦合器、校准统计。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "tianyan_probe"
OUT.mkdir(parents=True, exist_ok=True)


def load_key() -> str:
    for cand in (ROOT / ".tianyan_key",):
        try:
            text = cand.read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError:
            continue
    return ""


def probe(machine: str, token: str):
    from cqlib import TianYanPlatform
    print(f"\n===== {machine} =====")
    try:
        platform = TianYanPlatform(login_key=token, machine_name=machine)
        cfg = platform.download_config(machine=machine)
        if not isinstance(cfg, dict):
            print(f"  download_config 返回非 dict: {type(cfg)}")
            return
        (OUT / f"{machine}_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        print(f"  已保存 {OUT / f'{machine}_config.json'}")

        overview = cfg.get("overview", {}) or {}
        print(f"  cfg 顶层键: {sorted(cfg.keys())}")
        print(f"  overview 键: {sorted(overview.keys())}")
        cm = overview.get("coupler_map", {}) or {}
        n_couplers = len(cm)
        qubits = sorted({int(str(v0).lstrip('Q')) for pair in cm.values()
                         for v0 in (str(pair[0]), str(pair[1]))})
        print(f"  coupler_map 条目数: {n_couplers}")
        print(f"  涉及比特: {len(qubits)} 个（min Q{min(qubits)} max Q{max(qubits)}）"
              if qubits else "  涉及比特: 无")
        for k in ("disabledQubits", "disabledCouplers", "availableQubits",
                  "availableCouplers"):
            if k in cfg:
                v = cfg[k]
                s = str(v)
                print(f"  {k}: {s[:300]}")
        for k in ("status", "state", "machineStatus", "type"):
            for kk in (k, k.lower()):
                if kk in overview:
                    print(f"  overview[{kk}]: {overview[kk]}")
                    break
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {e}")


def main():
    token = load_key()
    if not token:
        print("无 .tianyan_key")
        return
    print(f"key 长度 {len(token)}（不打印内容）")
    for m in ("tianyan-287", "tianyan176"):
        probe(m, token)


if __name__ == "__main__":
    main()
