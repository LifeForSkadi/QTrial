"""自研 OpenQASM 2.0 解析器（qiskit-free）。

支持：qreg/creg、qelib1 标准门（经 u3/u1/cx 展开）、u/u1/u2/u3/cx/cz/swap/
sx/sxdg、用户自定义 gate 定义（含多行，递归展开）、measure、barrier。
不支持：opaque、if、reset（明确报错）。
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from qtrail.pure.circuit import Circuit, Inst

ID = r"[a-zA-Z][a-zA-Z0-9_]*"
QARG_RE = re.compile(rf"({ID})\[(\d+)\]")
LINE_GATE_RE = re.compile(rf"^\s*({ID})(?:\s*\((.*?)\))?\s+([^;]+);?\s*$")
NUM_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")

# qelib1 展开（不含 cx/measure；id 特殊处理）
QELIB1 = {
    "h": ("u3", (math.pi / 2, 0.0, math.pi)),
    "x": ("u3", (math.pi, 0.0, math.pi)),
    "y": ("u3", (math.pi / 2, math.pi / 2, math.pi / 2)),
    "z": ("u1", (math.pi,)),
    "s": ("u1", (math.pi / 2,)),
    "sdg": ("u1", (-math.pi / 2,)),
    "t": ("u1", (math.pi / 4,)),
    "tdg": ("u1", (-math.pi / 4,)),
    "id": ("id", ()),
}


class QASMError(ValueError):
    pass


def _eval(expr: str, binds: dict[str, float]) -> float:
    expr = re.sub(r"\bpi\b", str(math.pi), expr).replace("^", "**")
    env = {"__builtins__": {}, **binds}
    return float(eval(expr, env))  # noqa: S307 词法受限白名单


def _split_top(text: str) -> list[str]:
    out, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


class _GateDef:
    def __init__(self, params, qnames, body):
        self.params = params
        self.qnames = qnames
        self.body = body  # [("u3"|"u1"|"cx"|"u2", [param_expr], [qref])]


# Toffoli 分解（Nielsen & Chuang 图 4.9，相对相位在块内抵消）：
# (gate, 相对 qubit 下标, params)——h/t/tdg 用 u3/u1 表示
_TOFFOLI = [
    ("u3", (2,), (math.pi / 2, 0.0, math.pi)),          # h(c)
    ("cx", (1, 2), ()),                                 # cx(b,c)
    ("u1", (2,), (-math.pi / 4,)),                      # t†(c)
    ("cx", (0, 2), ()),                                 # cx(a,c)
    ("u1", (2,), (math.pi / 4,)),                       # t(c)
    ("cx", (1, 2), ()),                                 # cx(b,c)
    ("u1", (2,), (-math.pi / 4,)),                      # t†(c)
    ("cx", (0, 2), ()),                                 # cx(a,c)
    ("u1", (1,), (math.pi / 4,)),                       # t(b)
    ("u1", (2,), (math.pi / 4,)),                       # t(c)
    ("u3", (2,), (math.pi / 2, 0.0, math.pi)),          # h(c)
    ("cx", (0, 1), ()),                                 # cx(a,b)
    ("u1", (0,), (math.pi / 4,)),                       # t(a)
    ("u1", (1,), (-math.pi / 4,)),                      # t†(b)
    ("cx", (0, 1), ()),                                 # cx(a,b)
]


def _parse_gate_body(body: str, params: list[str], qnames: list[str]):
    out = []
    for stmt in body.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        m = re.match(rf"^({ID})(?:\s*\((.*?)\))?\s+([^;]+)$", stmt)
        if not m:
            raise QASMError(f"bad gate body stmt: {stmt[:50]}")
        name, pstr, qstr = m.group(1), m.group(2) or "", m.group(3)
        pexprs = [s.strip() for s in _split_top(pstr)] if pstr else []
        qrefs = [s.strip() for s in qstr.split(",") if s.strip()]
        out.append((name, pexprs, qrefs))
    return out


def parse_qasm(text: str) -> Circuit:
    qregs, cregs = {}, {}
    qreg_order, creg_order = [], []
    gates: dict[str, _GateDef] = {}
    circuit = None

    def q_index(name, idx):
        base = 0
        for r in qreg_order:
            if r == name:
                return base + idx
            base += qregs[r]
        raise QASMError(f"unknown qreg {name}")

    def c_index(name, idx):
        base = 0
        for r in creg_order:
            if r == name:
                return base + idx
            base += cregs[r]
        raise QASMError(f"unknown creg {name}")

    def emit(inst):
        nonlocal circuit
        if circuit is None:
            n = sum(qregs.values())
            if n == 0:
                raise QASMError("no qreg declared")
            circuit = Circuit(n)
        circuit.append(inst)

    def apply(name, pexprs, qrefs, binds, depth=0):
        if depth > 32:
            raise QASMError("gate definition recursion too deep")
        if name in gates:
            gd = gates[name]
            if len(pexprs) != len(gd.params) or len(qrefs) != len(gd.qnames):
                raise QASMError(f"gate {name} arity mismatch")
            binds = dict(binds)
            for p, e in zip(gd.params, pexprs):
                binds[p] = _eval(e, binds)
            # 形式量子名 → 实际全局索引（qrefs 为 QARG 字符串）
            for qname, qref in zip(gd.qnames, qrefs):
                m3 = QARG_RE.match(qref)
                if not m3:
                    raise QASMError(f"bad qarg {qref} for gate {name}")
                binds[qname] = q_index(m3.group(1), int(m3.group(2)))
            for (gname, gpexprs, gqrefs) in gd.body:
                qs = []
                for ref in gqrefs:
                    if ref in binds:
                        qs.append(binds[ref])
                    else:
                        m2 = QARG_RE.match(ref)
                        if not m2:
                            raise QASMError(f"bad qref {ref} in gate {name}")
                        qs.append(q_index(m2.group(1), int(m2.group(2))))
                apply(gname, [_eval(e, binds) for e in gpexprs], qs,
                      binds, depth + 1)
            return
        qubits = tuple(q_index(m2.group(1), int(m2.group(2)))
                       if isinstance(r, str) and (m2 := QARG_RE.match(r))
                       else r for r in qrefs)
        pvals = [_eval(e, binds) for e in pexprs]
        if name in ("cx", "CX"):
            emit(Inst("cx", qubits))
        elif name == "cz":
            emit(Inst("cz", qubits))
        elif name == "swap":
            emit(Inst("swap", qubits))
        elif name == "u3":
            emit(Inst("u3", qubits, tuple(pvals)))
        elif name == "u2":
            emit(Inst("u3", qubits, (math.pi / 2, pvals[0], pvals[1])))
        elif name == "u1":
            emit(Inst("u1", qubits, (pvals[0],)))
        elif name == "u":
            emit(Inst("u3", qubits, tuple(pvals)))
        elif name == "sx":
            emit(Inst("u3", qubits, (math.pi / 2, -math.pi / 2, math.pi / 2)))
        elif name == "sxdg":
            emit(Inst("u3", qubits, (-math.pi / 2, -math.pi / 2, math.pi / 2)))
        elif name == "p":
            emit(Inst("u1", qubits, (pvals[0],)))
        elif name == "cp":
            # 受控相位标准分解（|11⟩ 获相位 e^{iλ}）：
            # u1(λ/2) c; cx(c,t); u1(−λ/2) t; cx(c,t); u1(λ/2) t
            ctl, tgt = qubits
            emit(Inst("u1", (ctl,), (pvals[0] / 2,)))
            emit(Inst("cx", (ctl, tgt)))
            emit(Inst("u1", (tgt,), (-pvals[0] / 2,)))
            emit(Inst("cx", (ctl, tgt)))
            emit(Inst("u1", (tgt,), (pvals[0] / 2,)))
        elif name == "rx":
            emit(Inst("u3", qubits, (pvals[0], -math.pi / 2, math.pi / 2)))
        elif name == "ry":
            emit(Inst("u3", qubits, (pvals[0], 0.0, 0.0)))
        elif name == "rz":
            emit(Inst("u1", qubits, (pvals[0],)))
        elif name == "ccx":
            # Toffoli 标准分解（N&C 4.9，6 CNOT + 单比特门）
            a, b, c = qubits
            for gname, gqs, gparams in _TOFFOLI:
                emit(Inst(gname, tuple(qubits[q] for q in gqs), gparams))
        elif name in QELIB1:
            kind, vals = QELIB1[name]
            if kind == "id":
                emit(Inst("id", qubits))
            elif kind == "u1":
                emit(Inst("u1", qubits, (vals[0],)))
            else:
                emit(Inst("u3", qubits, vals))
        else:
            raise QASMError(f"unknown gate '{name}'")

    # ---- 扫描行
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        if not s or s.startswith("//"):
            continue
        if s.startswith("OPENQASM") or s.startswith("include"):
            continue
        m = re.match(rf"qreg\s+({ID})\[(\d+)\];?", s)
        if m:
            qregs[m.group(1)] = int(m.group(2))
            qreg_order.append(m.group(1))
            continue
        m = re.match(rf"creg\s+({ID})\[(\d+)\];?", s)
        if m:
            cregs[m.group(1)] = int(m.group(2))
            creg_order.append(m.group(1))
            continue
        if s.startswith("barrier"):
            continue
        m = re.match(rf"measure\s+{QARG_RE.pattern}\s*->\s*{QARG_RE.pattern};?", s)
        if m:
            emit(Inst("measure", (q_index(m.group(1), int(m.group(2))),),
                      cbits=(c_index(m.group(3), int(m.group(4))),)))
            continue
        if s.startswith("opaque") or s.startswith("if") or s.startswith("reset"):
            raise QASMError(f"unsupported instruction: {s[:50]}")
        m = re.match(rf"gate\s+({ID})(?:\s*\(([^)]*)\))?\s+([a-zA-Z0-9\s,]+)\s*\{{", s)
        if m:
            gname, pstr, qstr = m.group(1), m.group(2) or "", m.group(3)
            body = ""
            while i < len(lines) and "}" not in lines[i - 1]:
                body += " " + lines[i].split("//")[0]
                i += 1
            body += " " + s.split("{", 1)[1]
            body = body.split("}")[0]  # 去掉闭括号
            params = [p.strip() for p in _split_top(pstr)] if pstr else []
            qnames = [q.strip() for q in qstr.split(",") if q.strip()]
            gates[gname] = _GateDef(params, qnames,
                                    _parse_gate_body(body, params, qnames))
            continue
        m = LINE_GATE_RE.match(s)
        if m:
            gname, pstr, qstr = m.group(1), m.group(2) or "", m.group(3)
            if gname in ("measure", "qreg", "creg"):
                continue
            pexprs = [x.strip() for x in _split_top(pstr)] if pstr else []
            qrefs = [x.strip() for x in qstr.split(",") if x.strip()]
            apply(gname, pexprs, qrefs, {})
            continue
        raise QASMError(f"unparseable line: {s[:60]}")

    if circuit is None:
        raise QASMError("no instructions found")
    return circuit


def load_qasm_file(path: str | Path) -> Circuit:
    return parse_qasm(Path(path).read_text(encoding="utf-8"))
