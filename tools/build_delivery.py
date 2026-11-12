# -*- coding: utf-8 -*-
"""交付包重建（2026-11-12，外来审查 P0-2 修复）。

问题：delivery/ 是约一个月前的手抄快照（缺 post_v2/relabel_vf2/
synthesis/router_kernel，且清单声称"逐字节一致"）——评审拿到的
代码与仓库主线不一致。

本脚本从**当前工作区**重建 delivery 的代码与文档部分（数据/权重/
结果表不重抄），生成逐文件 SHA256 清单并校验：
  1. 全量同步 qtrail/（含 pure/ 全部新模块）、tools/、tests/
  2. scripts/ 按白名单同步：交付包已有的脚本 + 本轮新增的验证/
     审计/基准脚本（REQUIRED_SCRIPTS）
  3. 重新生成 delivery/docs/DELIVERY_MANIFEST_SHA256.txt
  4. 自检：关键文件（post_v2.py/relabel_vf2.py/synthesis.py/
     router_kernel.py/final_report_bench_v7.py）存在且哈希与仓库
     一致；不一致则非零退出

用法：python tools/build_delivery.py [--check]
  --check 只校验不写入（CI 用）。
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DELIVERY = ROOT / "delivery"

SYNC_DIRS = ["qtrail", "tools", "tests"]

# 交付包必备的新增脚本（本轮机制/验证链的核心，缺一即失败）
REQUIRED_SCRIPTS = [
    "final_report_bench_v7.py",
    "audit_pure_pipeline.py",
    "verify_e2e_production.py",
    "verify_decompose_linear_semantics.py",
    "ab_decompose_linear.py",
    "ab_linear_pipeline.py",
    "ab_linear_semantics.py",
    "ab_router_kernel.py",
    "verify_all_legality.py",
    "ab_a2_backend.py",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path) -> list[Path]:
    copied = []
    if not src.exists():
        return copied
    for p in src.rglob("*"):
        if p.is_dir() or "__pycache__" in p.parts:
            continue
        if p.suffix in (".pyc", ".pyo"):
            continue
        rel = p.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha256(p) != sha256(target):
            shutil.copy2(p, target)
            copied.append(target)
    return copied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="只校验不写入")
    args = ap.parse_args()

    # ---- 1. 同步代码目录
    updated = []
    for d in SYNC_DIRS:
        src = ROOT / d
        dst = DELIVERY / d
        if args.check:
            missing = [p for p in src.rglob("*.py")
                       if "__pycache__" not in p.parts
                       and not (dst / p.relative_to(src)).exists()]
            stale = [p for p in src.rglob("*.py")
                     if "__pycache__" not in p.parts
                     and (dst / p.relative_to(src)).exists()
                     and sha256(p) != sha256(dst / p.relative_to(src))]
            print(f"[{d}] 缺失 {len(missing)} 陈旧 {len(stale)}")
            updated.extend(missing + stale)
        else:
            copied = copy_tree(src, dst)
            print(f"[{d}] 同步 {len(copied)} 个文件")
            updated.extend(copied)

    # ---- 2. scripts：已有同名覆盖 + 必备新增
    repo_scripts = ROOT / "scripts"
    del_scripts = DELIVERY / "scripts"
    del_scripts.mkdir(parents=True, exist_ok=True)
    to_copy = {p.name for p in del_scripts.glob("*.py")}
    to_copy |= set(REQUIRED_SCRIPTS)
    n_sc = 0
    for name in sorted(to_copy):
        src = repo_scripts / name
        if not src.exists():
            print(f"  [跳过缺失脚本] {name}")
            continue
        dst = del_scripts / name
        if args.check:
            if not dst.exists() or sha256(src) != sha256(dst):
                updated.append(dst)
        else:
            if not dst.exists() or sha256(src) != sha256(dst):
                shutil.copy2(src, dst)
                n_sc += 1
    print(f"[scripts] 同步 {n_sc} 个（白名单 {len(to_copy)}）")

    if args.check:
        ok = not updated
        print(f"\n校验结果：{'一致' if ok else f'不一致 {len(updated)} 个文件'}")
        for p in updated[:10]:
            print("  ", p.relative_to(ROOT))
        return 0 if ok else 1

    # ---- 3. 哈希清单 + 必备自检
    manifest = []
    for d in ["qtrail", "scripts", "tools", "tests", "configs"]:
        base = DELIVERY / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_dir() or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            manifest.append(f"{sha256(p)}  {p.relative_to(DELIVERY)}")
    mpath = DELIVERY / "docs" / "DELIVERY_MANIFEST_SHA256.txt"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"[manifest] {len(manifest)} 个文件哈希 → {mpath}")

    # ---- 4. 必备文件自检（评审口径）
    must = ["qtrail/pure/post_v2.py", "qtrail/pure/relabel_vf2.py",
            "qtrail/pure/synthesis.py", "qtrail/pure/router_kernel.py"]
    must += [f"scripts/{s}" for s in REQUIRED_SCRIPTS]
    bad = []
    for rel in must:
        dp = DELIVERY / rel
        rp = ROOT / rel
        if not dp.exists():
            bad.append(f"缺失 {rel}")
        elif rp.exists() and sha256(dp) != sha256(rp):
            bad.append(f"不一致 {rel}")
    if bad:
        print("\n[自检失败]", *bad, sep="\n  ")
        return 1
    print(f"\n[自检通过] {len(must)} 个必备文件与仓库逐字节一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
