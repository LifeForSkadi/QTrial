# -*- coding: utf-8 -*-
"""后处理候选的状态类型（2026-11-13 执行顺序第 4 项）。

背景（学术审查 §3.2）：v5 管线的三类候选——fold_only+全局重标记、
fold_and_relink、fold_keep、裸路由——曾以裸三元组
``(circuit, final_layout, input_layout)`` 混在同一个列表里，标签的
坐标帧只写在注释里。2026-11-12 的 fold "保留 swap" P0 正是帧混用：
内容坐标下又发射物理 swap ⇒ 同一置换应用两次。

本模块把帧变成类型化的构造契约：
  - frame="content"：线路标签是**内容/初始坐标**（折叠类产物）。消费者
    必须用候选自身的 final_layout 作为输入嵌入——因此构造时强制
    ``input_layout == final_layout``。
  - frame="physical"：线路标签是**物理坐标**（裸路由）。输入嵌入是
    路由的初始布局 layout（可与 final_layout 不同）。
构造时复制布局字典，防止候选之间/调用方后续原地修改造成别名污染。
"""
from __future__ import annotations

from dataclasses import dataclass

from qtrail.pure.circuit import Circuit

FRAMES = ("physical", "content")


@dataclass(frozen=True)
class CandidateResult:
    circuit: Circuit
    frame: str
    final_layout: dict
    input_layout: dict | None
    tag: str = ""

    def __post_init__(self):
        if self.frame not in FRAMES:
            raise ValueError(
                f"CandidateResult: 未知帧 {self.frame!r}（允许 {FRAMES}）")
        # 复制布局：候选是不可变值，不得与调用方共享可变 dict
        object.__setattr__(self, "final_layout", dict(self.final_layout))
        if self.input_layout is not None:
            object.__setattr__(self, "input_layout",
                               dict(self.input_layout))
        if self.frame == "content":
            if self.input_layout is None:
                raise ValueError(
                    "CandidateResult: content 帧必须提供 input_layout")
            if dict(self.input_layout) != dict(self.final_layout):
                raise ValueError(
                    "CandidateResult: content 帧的输入嵌入必须等于候选"
                    " final_layout（内容坐标约定，2026-11-12 P0 防复发）")
        # physical 帧：input_layout 允许不同于 final_layout（裸路由：
        # 初始布局 vs 最终布局）；None 表示恒等嵌入。

