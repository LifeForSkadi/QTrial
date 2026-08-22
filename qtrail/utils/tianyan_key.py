"""天衍平台 API key 注入工具（安全约定）。

优先级：环境变量 TIANYAN_LOGIN_KEY / CQLIB_LOGIN_KEY > 项目根目录
`.tianyan_key` 本地文件（一行 key）。**该文件是私有凭据，绝不进入
交付包**——打包/提交前务必删除（README 与作品报告提交提醒中已注明）。
"""
from __future__ import annotations

import os
from pathlib import Path

_LOCAL_FILE = Path(__file__).resolve().parent.parent.parent / ".tianyan_key"


def get_key() -> str:
    """返回可用的 API key（无则空串）。"""
    for var in ("TIANYAN_LOGIN_KEY", "CQLIB_LOGIN_KEY"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    if _LOCAL_FILE.exists():
        try:
            v = _LOCAL_FILE.read_text(encoding="utf-8").strip()
            if v:
                return v
        except OSError:
            pass
    return ""


def key_source() -> str:
    """当前生效 key 的来源（env var / local file / none），用于诊断输出。"""
    for var in ("TIANYAN_LOGIN_KEY", "CQLIB_LOGIN_KEY"):
        if os.environ.get(var, "").strip():
            return f"env:{var}"
    if _LOCAL_FILE.exists():
        return "file:.tianyan_key"
    return "none"
