# -*- coding: utf-8 -*-
"""OMML 数学对象定义（Word/MathType 原生公式格式）。

行内数学符号与显示公式统一以 m:oMath 对象输出——Word 内置公式编辑器
与 MathType 插件均可双击编辑。
"""
from __future__ import annotations

import re

from lxml import etree

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _m(tag):
    return f"{{{M}}}{tag}"


def _r(parent, text):
    r = etree.SubElement(parent, _m("r"))
    # Word 原生公式的每个 run 都带样式属性（默认文本型）
    rPr = etree.SubElement(r, _m("rPr"))
    sty = etree.SubElement(rPr, _m("sty"))
    sty.set(_m("val"), "p")
    t = etree.SubElement(r, _m("t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return r


def _ssub(parent, base, sub):
    s = etree.SubElement(parent, _m("sSub"))
    e = etree.SubElement(s, _m("e"))
    _r(e, base)
    sb = etree.SubElement(s, _m("sub"))
    _r(sb, sub)
    return s


def _ssup(parent, base, sup):
    s = etree.SubElement(parent, _m("sSup"))
    e = etree.SubElement(s, _m("e"))
    _r(e, base)
    sb = etree.SubElement(s, _m("sup"))
    _r(sb, sup)
    return s


def _sum(parent, sub_text, e_items):
    """求和符号：以 Σ 带下限（sSub）实现，替代 m:nary——渲染稳定、
    与 nary 的 undOvr 形式视觉等效，避免个别 Word 版本对 nary 的兼容问题。"""
    s = etree.SubElement(parent, _m("sSub"))
    e = etree.SubElement(s, _m("e"))
    _r(e, "Σ")
    sb = etree.SubElement(s, _m("sub"))
    _r(sb, sub_text)
    for it in e_items:
        _items(parent, [it])
    return s


def _items(parent, items):
    for it in items:
        kind = it[0]
        if kind == "r":
            _r(parent, it[1])
        elif kind == "ssub":
            _ssub(parent, it[1], it[2])
        elif kind == "ssup":
            _ssup(parent, it[1], it[2])
        elif kind == "nary":
            _sum(parent, it[1], it[2])


def build_omath(items) -> etree._Element:
    """构造一个 m:oMath 元素（Word 原生公式对象）。"""
    om = etree.Element(_m("oMath"))
    _items(om, items)
    return om


def append_inline_math(p_xml, items):
    """把 m:oMath 追加到 w:p 元素（行内公式）。"""
    p_xml.append(build_omath(items))


# ---------------------------------------------------------------- 显示公式
# 键 → (标签, 公式 items)
DISPLAY = {
    "qap_cost": ("式(2-1)　", [
        ("r", "C("), ("r", "π"), ("r", ") = "),
        ("nary", "i<j", [("ssub", "w", "ij"), ("r", " · 2 · d("),
                         ("r", "π"), ("r", "(i), "), ("r", "π"),
                         ("r", "(j))")])]),
    "d_eff": ("式(2-2)　", [
        ("ssub", "D", "eff"), ("r", " = "), ("ssub", "D", "topo"),
        ("r", " + λ("), ("ssub", "D", "noise"), ("r", " − "),
        ("ssub", "D", "topo"), ("r", ")")]),
    "score": ("式(4-2)　", [
        ("r", "score(a,b) = "),
        ("nary", "g", [("ssup", "0.5", "t"), ("r", " · d′(g)")]),
        ("r", " + 0.5 · "),
        ("nary", "e", [("ssup", "0.5", "t"), ("r", " · d′")]),
        ("r", " + 0.5 · hist[a,b]")]),
    "reward": ("式(4-1)　", [
        ("r", "R = −("),
        ("nary", "i<j", [("ssub", "w", "ij"), ("r", " · 2 · "),
                         ("ssub", "D", "eff"), ("r", "(π(i), π(j))")]),
        ("r", ") / Σw")]),
    "u3": ("式(4-3)　", [
        ("r", "U3("), ("r", "θ,φ,λ"), ("r", ") = RZ("), ("r", "λ"),
        ("r", ")·SX·RZ("), ("r", "θ"), ("r", "+"), ("r", "π"),
        ("r", ")·SX·RZ("), ("r", "φ"), ("r", "+3"), ("r", "π"),
        ("r", ")")]),
    "loss": ("式(4-4)　", [
        ("r", "L = −(1/B)·"),
        ("nary", "b", [("r", "[("), ("ssub", "R", "s"), ("r", " − "),
                       ("ssub", "R", "b"), ("r", ") · log "),
                       ("ssub", "p", "s"), ("r", ")]")])]),
    "clamps": ("式(4-5)　", [
        ("ssub", "φ", "e"), ("r", " = clip("), ("ssub", "ε", "e"),
        ("r", "/ē, 0.2, 5.0)；"), ("ssub", "τ", "e"),
        ("r", " = clip(1 − min("), ("ssub", "T1", "i"), ("r", ", "),
        ("ssub", "T1", "j"), ("r", ")/max("), ("ssub", "T1", "i"),
        ("r", ", "), ("ssub", "T1", "j"), ("r", "), 0, 2.0)")]),
    "laurel": ("式(4-6)　", [
        ("ssub", "x", "i+1"), ("r", " = "), ("ssub", "α", "i"),
        ("r", "·f("), ("ssub", "x", "i"), ("r", ") + "), ("ssub", "β", "i"),
        ("r", "·"), ("ssub", "x", "i"), ("r", " + ("), ("ssub", "A", "i"),
        ("r", "·"), ("ssub", "B", "i"), ("r", ")·"), ("ssub", "x", "i")]),
    "decay": ("式(4-7)　", [
        ("ssub", "w", "g"), ("r", " = "), ("ssup", "0.5", "t_g"),
        ("r", "；"), ("ssub", "t", "g"), ("r", " ← "), ("ssub", "t", "g"),
        ("r", " + 1")]),
    "conjug1": ("式(4-8)　", [
        ("r", "SWAP(a,b)·CNOT(a,c)·SWAP(a,b) ≡ CNOT(b,c)")]),
    "conjug2": ("式(4-9)　", [
        ("r", "SWAP(a,b)·CNOT(a,b)·SWAP(a,b) ≡ CNOT(b,a)")]),
}

# ---------------------------------------------------------------- 行内符号
# 正则 → items（长模式在前，避免子串误匹配）
INLINE_PATTERNS = [
    (r"ε_1q", [("ssub", "ε", "1q")]),
    (r"ε_2q", [("ssub", "ε", "2q")]),
    (r"ε_ro", [("ssub", "ε", "ro")]),
    (r"ε_e", [("ssub", "ε", "e")]),
    (r"D_eff", [("ssub", "D", "eff")]),
    (r"D_topo", [("ssub", "D", "topo")]),
    (r"D_noise", [("ssub", "D", "noise")]),
    (r"w_ij", [("ssub", "w", "ij")]),
    (r"w_e", [("ssub", "w", "e")]),
    (r"w_g", [("ssub", "w", "g")]),
    (r"φ_e", [("ssub", "φ", "e")]),
    (r"τ_e", [("ssub", "τ", "e")]),
    (r"λ_n", [("ssub", "λ", "n")]),
    (r"λ_ms", [("ssub", "λ", "ms")]),
    (r"α_i", [("ssub", "α", "i")]),
    (r"β_i", [("ssub", "β", "i")]),
    (r"α_t", [("ssub", "α", "t")]),
    (r"A_i", [("ssub", "A", "i")]),
    (r"B_i", [("ssub", "B", "i")]),
    (r"R_s", [("ssub", "R", "s")]),
    (r"R_b", [("ssub", "R", "b")]),
    (r"T1_i", [("ssub", "T1", "i")]),
    (r"T1_j", [("ssub", "T1", "j")]),
    (r"x_{i\+1}", [("ssub", "x", "i+1")]),
    (r"x_i", [("ssub", "x", "i")]),
    (r"0\.5\^t_g", [("ssup", "0.5", "t_g")]),
    (r"0\.5\^t_e", [("ssup", "0.5", "t_e")]),
    (r"0\.5\^t", [("ssup", "0.5", "t")]),
    (r"t_g", [("ssub", "t", "g")]),
    (r"C\(π\)", [("r", "C(π)")]),
    (r"π_θ", [("ssub", "π", "θ")]),
    (r"π", [("r", "π")]),
    (r"λ", [("r", "λ")]),
    (r"α", [("r", "α")]),
    (r"β", [("r", "β")]),
    (r"θ", [("r", "θ")]),
    (r"φ", [("r", "φ")]),
    (r"ε", [("r", "ε")]),
]


def convert_inline(text):
    """段落文本 → [(type, payload)]，type ∈ {"t", "m"}。

    按 INLINE_PATTERNS 顺序贪婪匹配；未命中的字符归入文本段。
    """
    segments = []
    buf = []
    i = 0
    while i < len(text):
        matched = False
        for pat, items in INLINE_PATTERNS:
            m = re.match(pat, text[i:])
            if m:
                if buf:
                    segments.append(("t", "".join(buf)))
                    buf = []
                segments.append(("m", items))
                i += len(m.group(0))
                matched = True
                break
        if not matched:
            buf.append(text[i])
            i += 1
    if buf:
        segments.append(("t", "".join(buf)))
    return segments
