# -*- coding: utf-8 -*-
"""作品报告书章节重构：按用户要求重组第 2/3/4/5 章与摘要，输出 JSON 内容文件。

原则：第 1 章、附录 A/C/D 的块保持原样（逐字不变）；重构仅涉及
摘要、第 2-5 章与附录 B 图表清单。运行后由 build.py 读取 JSON 生成 docx。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import content1
import content2
import content3

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 交叉引用替换
# 仅作用于第 2-5 章与附录 B（第 1 章与附录 A/C/D 不经过本替换）。
# 顺序敏感：先删指向已删除内容的引用，再做大范围重编号；
# 同前缀条目长串在前，避免短串先行破坏长串。
REFMAP = [
    # —— 消融实验（原 5.9）删除，其引用一并清除
    ("，并在 5.9 节的消融讨论中将其与深度决胜规则一并评估", ""),
    ("在 5.9 节的消融讨论", ""),
    ("，消融结果见 5.9 节", ""),
    ("见 5.9 节", ""),
    ("消融与缺陷扫描表明", "缺陷扫描表明"),
    ("创新性以第 4 章三处方法组件及其消融验收",
     "创新性以第 4 章三处方法组件的设计与评测验收"),
    ("（5.9 节的 oracle 负结果佐证）", ""),
    # —— 第 3 章（上一轮已重构，编号不变：3.3 为关键设计决策）
    ("见 3.4 节", "见 5.1 节"),  # 数据集与评测体系
    ("见 3.5 节", "见 3.3 节"),
    ("本文 3.5 节“语义等价", "本文 3.3 节“语义等价"),
    ("3.5 节记录的", "3.3 节记录的"),
    ("3.5 节与 4.3.4 节", "3.3 节与 4.1.6 节"),
    ("是 3.5 节口径一致原则", "是 3.3 节口径一致原则"),
    ("保证了 3.3 节所述模块可替换性的成立", "保证了模块可替换性的成立"),
    # —— 第 4 章（上一轮已重构，引用目标不变）
    ("见 4.7 节", "见 4.1.8 节"),
    ("见 4.8 节", "见 4.1.9 节"),
    ("见 4.3.2 节", "见 4.1.4 节"),
    ("见 4.3.4 节", "见 4.1.6 节"),
    ("见 4.9 节", "见 4.3.2 节"),
    ("见 4.10 节", "见 4.3.3 节"),
    ("在 4.3 节详述", "在 4.1 节详述"),
    ("并在 4.9 节的验证链中", "并在 4.3.2 节的验证链中"),
    # —— 第 5 章：消融删除后 5.4/5.5 顺延为 5.3/5.4；
    #    结果与分析次节重排：运行时间 5.4.3→5.4.4，噪声证据 5.4.4→5.4.3
    ("见 5.2 节", "见 5.4.1 节"),
    ("见 5.3 节", "见 5.4.2 节"),
    ("见 5.4 节", "见 5.4.4 节"),
    ("见 5.5 节", "见 5.4.3 节"),
    ("见 5.6 节", "见 5.4.5 节"),
    ("见 5.7 节", "见 5.3 节"),
    ("见 5.8 节", "见 5.4.6 节"),
    ("5.8 节分析", "5.4.6 节分析"),
    ("与 5.3 节的分层结论一致", "与 5.4.2 节的分层结论一致"),
    ("与 5.3 节评测同源", "与 5.4.2 节评测同源"),
    ("与 5.3 节评测完全一致", "与 5.4.2 节评测完全一致"),
    ("5.2 节报告的收敛轮数对比", "5.4.1 节报告的收敛轮数对比"),
    ("即 5.5 节的三拓扑", "即 5.4.3 节的三拓扑"),
    # —— 图号重排（第 5 章删除旧图5.2/图5.5 后顺延；条目精确匹配旧文本
    #    原文，避免误伤新增文本中的最终编号）：
    #    旧图5.3→图5.2、旧图5.4→图5.3、旧图5.6→图5.4、旧图5.7→图5.5；
    #    图5.8/图5.9/图5.10 编号不变
    ("图5.3给出", "图5.2给出"),
    ("图5.4给出", "图5.3给出"),
    ("结果如图5.6所示", "结果如图5.4所示"),
    ("图5.7给出", "图5.5给出"),
    # 布局贡献移入 5.4.3 后的图号顺延（精确匹配旧文本原文）：
    # 旧图5.10→图5.6、旧图5.8→图5.9、旧图5.9→图5.10
    ("结果如图5.10所示", "结果如图5.6所示"),
    ("图5.9给出 LAUREL", "图5.10给出 LAUREL"),
    ("图5.8给出", "图5.9给出"),
    ("5.1 节基线设置", "5.2 节基线设置"),
    ("本文 5.6 节的布局贡献实验", "本文 5.4.3 节的布局贡献实验"),
    ("5.6 节的布局贡献实验", "5.4.3 节的布局贡献实验"),
    ("为 5.6 节布局贡献实验的解读", "为 5.4.3 节布局贡献实验的解读"),
    ("5.3 至 5.6 节的分层与对比评测验收", "5.4 节的分层与对比评测验收"),
    # —— 公式编号（上一轮已重排，本轮不变）
    ("（式(4-2)）", "（式(4-1)）"),  # 源文中式(4-2) 均指训练奖励（现式(4-1)）
    # —— 其余（上一轮沿用）
    ("其一是 2.5 节的噪声加权距离", "其一是 2.3.1 节的噪声加权距离"),
    ("将 2.6 节介绍的 LAUREL 残差", "将 2.2.3 节介绍的 LAUREL 残差"),
]


def fix_refs(text):
    for a, b in REFMAP:
        text = text.replace(a, b)
    return text


# ---------------------------------------------------------------- 第 1 章微调
# 第 1 章不参与 REFMAP；仅修正因消融实验删除、第 3 章扩写与 SWIFT 更名而失效
# 的节号引用与术语（不改动任何论述）。
CH1_PATCH = [
    ("优势收窄（见 5.3 节）", "优势收窄（见 5.4.2 节）"),
    ("分别见 3.5 节与 4.3.4 节", "分别见 3.3 节与 4.1.6 节"),
    ("，并以消融实验逐项量化各自的贡献（见 5.9 节）", ""),
    ("自研 SABRE 路由、置换重写后处理", "SWIFT 路由内核、置换重写后处理"),
    ("（3）自主复现并工程加速 SABRE 路由器", "（3）自主复现并工程加速 SWIFT 路由内核"),
    ("工作（1）对应第 3 章总体设计与第 4 章的 4.1、4.4、4.5 节；工作（2）对应 "
     "4.2、4.3、4.6 节与第 5 章的 5.5、5.6 节实验；工作（3）对应 4.4 节与 "
     "5.4 节实验；工作（4）对应 4.5、4.9 节与 5.7 节实验；工作（5）对应 4.7、"
     "4.8 节与 5.2、5.9 节实验。",
     "工作（1）对应第 3 章总体设计与第 4 章的 4.1、4.2、4.3 节；工作（2）对应 "
     "4.1 节与第 5 章的 5.4.2、5.4.3、5.4.5 节实验；工作（3）对应 4.2 节与 "
     "5.4.4 节实验；工作（4）对应 4.3 节与 5.3 节实验；工作（5）对应 4.1.8、"
     "4.1.9 节与 5.4.1 节实验。"),
    ("模块划分、数据集与三项关键设计决策", "模块划分与三项关键设计决策"),
    ("第 5 章按训练、性能、时间、噪声、对比、真机、消融七个维度给出实验",
     "第 5 章按数据集与评测体系、实验设置、真机验证与结果分析四部分组织实验"),
]


def patch_ch1(blocks):
    """第 1 章节号/术语微调（正文其他部分经 fix_refs 处理）。"""
    out = []
    for b in blocks:
        if b[0] == "p":
            t = b[1]
            for a, c in CH1_PATCH:
                t = t.replace(a, c)
            out.append(("p", t))
        else:
            out.append(b)
    return out


def split_by_h2(blocks):
    """把块序列按 h2 切分为 {h2标题: [后续块]}（保持顺序）。"""
    secs = {}
    order = []
    cur = None
    for b in blocks:
        if b[0] == "h2":
            cur = b[1]
            secs[cur] = []
            order.append(cur)
        elif cur is not None:
            secs[cur].append(b)
    return secs, order


# ================================================================ 摘要
ABSTRACT_CN = (
    "量子线路映射是含噪声中等规模量子（NISQ）时代量子计算编译链路中的核心环节，"
    "其目标是为给定的逻辑线路与超导芯片拓扑确定比特布局并插入尽可能少的 SWAP 门，"
    "使映射后的线路可直接在真机上执行。本文面向天衍-287 超导量子处理器（105 比特、"
    "15×7 网格），设计并实现了量子线路映射优化方案 QTrial。方案按强化学习布局、"
    "SWIFT 路由内核与置换重写后处理三个阶段组织，输出可直接提交平台的映射后"
    "线路。")
ABSTRACT_CN_P2 = (
    "第一阶段为强化学习布局。布局被建模为二次分配问题，以图注意力网络与指针网络"
    "构成的策略求解：程序交互图与设备耦合图分别编码后逐比特分配物理位置，以噪声"
    "加权距离构造的 QAP 代价作为稀疏终局奖励训练，解码阶段辅以多起点采样与自适应"
    "局部搜索；设备特征按单台设备自身统计量归一化，使同一策略权重可跨拓扑、跨设备"
    "零重训迁移。")
ABSTRACT_CN_P3 = (
    "第二阶段为SWIFT路由内核。SWIFT 为 Swap-based Incremental Front-layered rouTer 的缩写，即基于交换的增量前沿路由；该内核自主复现 SABRE 路由机制——前沿层并行"
    "执行、扩展集前瞻、逐门衰减权重与交换历史惩罚，并通过前沿与扩展集增量维护、"
    "融合内核直接产出候选评分矩阵等工程改造将百比特级线路的路由时间压缩至秒级，"
    "逐位 A/B 验证保证工程加速不改变算法行为。")
ABSTRACT_CN_P4 = (
    "第三阶段为置换重写后处理。该阶段以共轭推挤、对消解、尾部吸收与平台基分解"
    "四层操作消化路由产生的 SWAP 门，全程限定为保语义等价变换，交付由态矢量等价"
    "测试、平台规则性校验与真机实测对照三层验证链守护。方案引入 LAUREL 可学习残差"
    "将训练收敛轮数由 400 轮压缩至 89 轮，并构建 QuEst 风格保真度预测器以捕获 T1/T2 "
    "解相干效应；在 Benchpress、MQTBench 与 QUEKO 共 587 条线路的分层评测中，总体"
    "估计保真度超过 pytket 与盲目基线，跨拓扑实验与缺陷敏感性扫描验证了噪声感知"
    "机制的有效性。")
ABSTRACT_CN_FULL = "\n".join([ABSTRACT_CN, ABSTRACT_CN_P2,
                              ABSTRACT_CN_P3, ABSTRACT_CN_P4])

ABSTRACT_EN = (
    "Quantum circuit mapping is a core step in the compilation flow of the noisy "
    "intermediate-scale quantum (NISQ) era, aiming to determine qubit layouts and "
    "insert as few SWAP gates as possible so that mapped circuits can execute "
    "directly on real hardware. This thesis designs and implements QTrial, a "
    "circuit mapping optimization scheme targeting the Tianyan-287 superconducting "
    "processor (105 qubits, 15x7 grid). The scheme is organized into three stages, "
    "namely reinforcement-learning-based layout, the SWIFT routing kernel, "
    "and permutation-rewriting post-processing, and outputs platform-ready circuits.")
ABSTRACT_EN_P2 = (
    "The first stage is reinforcement-learning-based layout. The layout is modeled "
    "as a quadratic assignment problem and solved by a policy composed of graph "
    "attention networks and pointer networks: the program interaction graph and the "
    "device coupling graph are encoded separately, physical positions are assigned "
    "qubit by qubit, a noise-weighted QAP cost serves as the sparse terminal reward "
    "during training, and multi-start sampling with adaptive local search refines "
    "decoding. Device features are normalized by each device's own statistics, so "
    "one policy weight transfers across topologies and devices without retraining.")
ABSTRACT_EN_P3 = (
    "The second stage is the SWIFT routing kernel (Swap-based Incremental Front-layered rouTer). The kernel reproduces "
    "the SABRE routing mechanisms - parallel execution of the front layer, extended "
    "set lookahead, per-gate decay weights and swap history penalties - and compresses "
    "the routing time of hundred-qubit circuits to seconds through incremental "
    "frontier maintenance and fused kernels that produce candidate scoring matrices "
    "directly. Bitwise A/B verification guarantees that the engineering acceleration "
    "does not alter algorithm behavior.")
ABSTRACT_EN_P4 = (
    "The third stage is permutation-rewriting post-processing. This stage digests "
    "the SWAP gates produced by routing through four layers of operations - "
    "conjugation pushing, pair cancellation, trailing absorption and platform-basis "
    "decomposition - restricted to semantics-preserving transformations, and the "
    "delivery is guarded by a three-layer verification chain of statevector "
    "equivalence, platform rule checking and on-device comparison. The scheme also "
    "adopts LAUREL learned residual layers to compress training from 400 epochs to "
    "89 epochs, and builds a QuEst-style fidelity predictor to capture T1/T2 "
    "decoherence effects. On the stratified evaluation of 587 circuits from "
    "Benchpress, MQTBench and QUEKO, the overall estimated fidelity exceeds pytket "
    "and noise-blind baselines, and cross-topology experiments as well as defect "
    "sensitivity scans verify the effectiveness of the noise-aware mechanism.")
ABSTRACT_EN_FULL = "\n".join([ABSTRACT_EN, ABSTRACT_EN_P2,
                              ABSTRACT_EN_P3, ABSTRACT_EN_P4])


# ================================================================ 第 2 章
def build_ch2(blocks):
    secs, order = split_by_h2(blocks)
    lead = ("本章建立后续章节所需的理论基础：先给出映射问题的形式化、SABRE 路由"
            "算法与评价指标体系，再介绍布局策略所依赖的强化学习与图神经网络方法，"
            "最后说明噪声模型与目标硬件平台的特性。")
    out = [("h1", "第2章 相关理论与技术基础"), ("p", lead)]
    out.append(("h2", "2.1 映射问题形式化与路由算法基础"))
    out.append(("h3", "2.1.1 量子线路映射问题的形式化"))
    out += secs["2.1 量子线路映射问题的形式化"]
    out.append(("h3", "2.1.2 SABRE 路由算法"))
    out += secs["2.2 SABRE 路由算法"]
    out.append(("h3", "2.1.3 评价指标体系"))
    out += secs["2.8 评价指标体系"]
    out.append(("h2", "2.2 强化学习与图神经网络方法"))
    out.append(("h3", "2.2.1 强化学习与 REINFORCE 算法"))
    out += secs["2.3 强化学习与 REINFORCE 算法"]
    out.append(("h3", "2.2.2 图注意力网络与指针网络"))
    out += secs["2.4 图注意力网络与指针网络"]
    out.append(("h3", "2.2.3 LAUREL 可学习残差层"))
    out += secs["2.6 LAUREL 可学习残差层"]
    out.append(("h2", "2.3 噪声模型与目标硬件"))
    out.append(("h3", "2.3.1 噪声模型与保真度估计"))
    out += secs["2.5 噪声模型与保真度估计"]
    out.append(("h3", "2.3.2 天衍-287 硬件平台"))
    out += secs["2.7 天衍-287 硬件平台"]
    # 2.9 本章小结删除；其余块（无）
    return [(fix_refs if False else b[0],) for b in [] ]  # placeholder


def apply(blocks):
    """对块列表中的 p 文本统一做交叉引用替换。"""
    out = []
    for b in blocks:
        if b[0] == "p":
            out.append(("p", fix_refs(b[1])))
        else:
            out.append(b)
    return out


def build_ch2_final(blocks):
    secs, order = split_by_h2(blocks)
    lead = ("本章建立后续章节所需的理论基础：先给出映射问题的形式化、SABRE 路由"
            "算法与评价指标体系，再介绍布局策略所依赖的强化学习与图神经网络方法，"
            "最后说明噪声模型与目标硬件平台的特性。")
    out = [("h1", "第2章 相关理论与技术基础"), ("p", lead)]
    out.append(("h2", "2.1 映射问题形式化与路由算法基础"))
    out.append(("h3", "2.1.1 量子线路映射问题的形式化"))
    out += secs["2.1 量子线路映射问题的形式化"]
    out.append(("h3", "2.1.2 SABRE 路由算法"))
    out += secs["2.2 SABRE 路由算法"]
    out.append(("h3", "2.1.3 评价指标体系"))
    out += secs["2.8 评价指标体系"]
    out.append(("h2", "2.2 强化学习与图神经网络方法"))
    out.append(("h3", "2.2.1 强化学习与 REINFORCE 算法"))
    out += secs["2.3 强化学习与 REINFORCE 算法"]
    out.append(("h3", "2.2.2 图注意力网络与指针网络"))
    out += secs["2.4 图注意力网络与指针网络"]
    out.append(("h3", "2.2.3 LAUREL 可学习残差层"))
    out += secs["2.6 LAUREL 可学习残差层"]
    out.append(("h2", "2.3 噪声模型与目标硬件"))
    out.append(("h3", "2.3.1 噪声模型与保真度估计"))
    out += secs["2.5 噪声模型与保真度估计"]
    out.append(("h3", "2.3.2 天衍-287 硬件平台"))
    out += secs["2.7 天衍-287 硬件平台"]
    return apply(out)


# ================================================================ 第 3 章
def build_ch3(blocks):
    secs, order = split_by_h2(blocks)
    out = [("h1", "第3章 总体方案设计")]
    out.append(("h2", "3.1 设计需求与总体架构"))
    out += secs["3.1 设计需求与目标"]
    out += secs["3.2 系统总体架构"]
    out.append(("h2", "3.2 模块功能划分与接口"))
    out += secs["3.3 模块功能划分与接口"]
    # 原 3.4 中的接口契约段并入 3.2
    s34 = secs["3.4 数据集与评测体系设计"]
    out.append(s34[2])  # 契约段（首段/分层段/Benchpress段/原生池段之外）
    out.append(("h2", "3.3 关键设计决策"))
    out += secs["3.5 关键设计决策"]
    return apply(out)


def dataset_blocks(blocks):
    """从原 3.4 提取移入 5.1 的块（契约段为模块接口内容，属第 3 章，不再移入）。"""
    secs, _ = split_by_h2(blocks)
    s34 = secs["3.4 数据集与评测体系设计"]
    # s34 顺序：首段(三基准+287池)、分层口径段、Benchpress 8类段、契约段、287原生池生成器段
    return [s34[0], s34[1], s34[4]] if len(s34) == 5 else s34


# ================================================================ 第 4 章
# 4.2 SWIFT 路由内核新增次节的扩写内容：全部取自原报告已有事实
# （零依赖红线、SABRE 机制、候选评分场景、接口契约与折叠事故教训等），
# 无新实验数据、新对比或新结论。
SWIFT_IMPL = {
    "lead": "本节给出 SWIFT 路由内核的设计动机、工作机制与关键参数，并说明其工程加速、"
            "行为一致性保障与相邻模块衔接。",
    "motivation": [
        ("p", "自研路由内核的决策源自三个约束。其一，交付管线的零依赖红线：3.1 节的功能"
              "完整性目标要求全链路不依赖 qiskit，直接调用 qiskit 的 SABRE 实现违背该"
              "红线，路由算法必须自主实现。其二，工业编译器中的 SABRE 虽被广泛采用，其"
              "工程实现通常与特定编译器栈绑定，且加速改造容易改变算法行为；自主实现将"
              "算法与工程的控制权一并收回，使工程加速与算法行为可以分开治理。其三，布局"
              "候选评分需要反复调用路由：候选决胜口径以“路由加后处理的真实结果”为据，"
              "每个候选布局都要完整路由一次，路由器的单次耗时直接进入布局决策的总耗时，"
              "必须做到百比特级线路秒级完成。"),
        ("p", "SWIFT 的命名取自其三项工程特征：Swap-based 表明以交换门为核心动作，"
              "Incremental 表明状态全程增量维护，Front-layered 表明以前沿层驱动执行与"
              "评分。整体定位上，SWIFT 完整复现 SABRE[2] 的路由机制——前沿层并行执行、"
              "扩展集前瞻、逐门衰减权重与交换历史惩罚，并以工程改造换取常数级加速，走"
              "“机制忠实、工程自研”的路线。"),
    ],
    "exec_model": ("p", "执行与交换两阶段的交替细节为：前沿层内的门并行执行，同一量子比特上的"
                        "多个门依序排队；前沿层无可执行的双比特门时进入交换阶段；扩展集由依赖"
                        "接近满足的门构成并参与前瞻评分，使交换决策不仅服务于眼前可执行的门，"
                        "也为即将就绪的门预留邻近位置。"),
    "params": ("p", "关键参数均沿袭 SABRE 的经典取值：衰减底数 0.5、扩展集为依赖满足前的次前沿"
                    "集合、历史惩罚随交换次数线性累积。随机种子固定的意义在于：同一布局候选在"
                    "同一配置下必得同一路由结果——候选评分对多个布局反复路由，确定性是候选间"
                    "比较无歧义的前提。"),
    "speed_role": ("p", "加速改造在管线中的定位：路由是布局候选评分的重复性开销，每个候选布局都要"
                        "完整路由一次，单次路由耗时直接进入布局决策的总耗时；Numba 作为可选加速"
                        "依赖引入（5.2 节的环境清单列其为可选），不改变路由结果、仅改变耗时。"
                        "路由内核的进一步融合与布局解码的批处理化共同构成 3.1 节所述性能优化"
                        "优先级的两条主线。"),
    "ab_rationale": ("p", "选择“输出逐字节一致”而非“指标相同”作为 A/B 判定的口径，原因在于指标"
                          "相同可能掩盖标签重标记层面的差异，而逐字节一致意味着输出线路、交换数与"
                          "最终布局在比特层面完全一致——这是加速改造与算法改动隔离的最强形式。"),
    "interface": ("p", "SWIFT 的输入输出契约与 3.2 节的三条接口约定一致：上游布局模块交付逻辑比特"
                       "到物理比特的映射字典，SWIFT 在该初始映射上路由，输出物理空间的 Circuit 与"
                       "最终布局；下游后处理模块消费路由产物，消化 SWAP 门并更新最终布局。路由门"
                       "存储的是执行时刻的物理位置，该约定是后处理栈连通性红线的数据基础——3.3 节"
                       "记录的“折叠事故”表明，混淆“输出标签的置换”与“执行时刻的位置”会在真机"
                       "校验处被拦截。"),
    "interface2": ("p", "SWIFT 的调用契约还包括输入防御：路由器对输入 Circuit 的依赖集做防御性拷贝，"
                        "不原地修改调用方数据。该设计源于候选评分场景——同一 Circuit 对象会被多个"
                        "候选布局反复路由，若路由器就地消耗依赖集，后续调用将拿到残缺的依赖 DAG"
                        "（4.1.1 节记录了该缺陷导致评测失真的事故）。此外，为隔离布局算法的贡献，"
                        "路由模块可整体替换为 qiskit 的 SABRE 实现（5.4.5 节的布局贡献实验即采用"
                        "该方式），可替换性是接口抽象粒度刻意保留的设计。"),
}


def build_ch4(blocks):
    secs, order = split_by_h2(blocks)
    out = [("h1", "第4章 关键模块的设计与实现"),
           ("p", "本章按方案三个阶段的顺序组织各模块的实现。第一阶段为强化学习布局，"
                 "覆盖从线路解析、程序图构建到布局决策的完整链条，并承载噪声感知与两项"
                 "模型侧改进；第二阶段为SWIFT路由内核；第三阶段为置换重写后处理，"
                 "并在其后给出验证与质量保障体系。")]
    out.append(("h2", "4.1 强化学习布局"))
    out.append(("h3", "4.1.1 电路解析器与电路中间表示"))
    out += secs["4.1 电路解析器与电路中间表示"]
    out.append(("h3", "4.1.2 程序图构建与节点特征"))
    out += secs["4.2 程序图构建与节点特征"]
    s43 = secs["4.3 基于强化学习的布局算法"]
    out.append(("h3", "4.1.3 策略网络结构"))
    out += _take_h3(s43, "4.3.1 策略网络结构")
    out.append(("h3", "4.1.4 训练算法"))
    out += _take_h3(s43, "4.3.2 训练算法")
    out.append(("h3", "4.1.5 解码与局部搜索"))
    out += _take_h3(s43, "4.3.3 解码与局部搜索")
    out.append(("h3", "4.1.6 失败阶梯"))
    out += _take_h3(s43, "4.3.4 失败阶梯")
    out.append(("h3", "4.1.7 噪声感知与跨设备零重训"))
    out += secs["4.6 噪声感知与跨设备零重训"]
    out.append(("h3", "4.1.8 LAUREL 可学习残差训练加速"))
    out += secs["4.7 LAUREL 可学习残差训练加速"]
    out.append(("h3", "4.1.9 QuEst 风格保真度预测器"))
    out += secs["4.8 QuEst 风格保真度预测器"]
    out.append(("h2", "4.2 SWIFT 路由内核"))
    s44 = secs["4.4 自研 SABRE 路由器"]
    out.append(("p", SWIFT_IMPL["lead"]))
    out.append(("h3", "4.2.1 设计动机与整体定位"))
    out += SWIFT_IMPL["motivation"]
    out.append(("h3", "4.2.2 工作机制与关键参数"))
    out += [s44[0], s44[1], s44[2]]  # 机制复现 + 式(4-2) + 评分三项说明
    out.append(("p", "评分三项中逐门衰减权重的维护规则如式(4-7)所示：式中 w_g 为门 g 的衰减"
                     "权重，t_g 为其衰减计数；选中交换 SWAP(a,b) 后，前沿层与扩展集中任一量子"
                     "比特落在 a 或 b 上的门其计数加一、权重随之减半。衰减使各门权重随路由进程"
                     "动态演化，是评分区别于静态距离求和的关键机制。"))
    out.append(("eq", "decay"))
    out.append(SWIFT_IMPL["exec_model"])
    out.append(SWIFT_IMPL["params"])
    out += [s44[7]]  # 候选边构造
    out += [s44[5]]  # 内部数据结构
    out.append(("h3", "4.2.3 增量维护与融合内核"))
    out += [s44[3], s44[6]]  # 四项工程改造 + 时间复杂度
    out.append(SWIFT_IMPL["speed_role"])
    out.append(("h3", "4.2.4 行为一致性 A/B 验证"))
    out += [s44[4]]  # A/B 方法学
    out.append(SWIFT_IMPL["ab_rationale"])
    out.append(("h3", "4.2.5 与相邻模块的衔接"))
    out += [s44[8]]  # 候选评分衔接
    out.append(SWIFT_IMPL["interface"])
    out.append(SWIFT_IMPL["interface2"])
    out.append(("h2", "4.3 置换重写后处理"))
    out.append(("h3", "4.3.1 置换重写后处理栈"))
    out += secs["4.5 后处理栈"]
    out.append(("h3", "4.3.2 真机验证链"))
    out += secs["4.9 真机验证链"]
    out.append(("h3", "4.3.3 测试体系与质量保障"))
    out += secs["4.10 测试体系与质量保障"]
    return apply(out)


def _take_h3(blocks, title):
    """取 h3 标题及所属块。"""
    out = []
    inside = False
    for b in blocks:
        if b[0] == "h3":
            inside = b[1] == title
            if inside:
                continue
        if inside:
            out.append(b)
    return out


# ================================================================ 第 5 章
# 运行时间测试集描述（5.1 数据集与评测体系新增）
RUNTIME_SET_BLOCKS = [
    ("p", "此外，为支撑运行时间扩展性的评测，单独构建了一个规模-难度网格测试集（下称运行"
          "时间测试集）。该测试集以天衍-287 的耦合图为生成源，按比特数 {10, 20, 40, 60, "
          "80, 105} 六档与固有路由难度三档构成 6×3×5 = 90 条线路：低难度档为牛耕式蛇形"
          "路径结构（交互图可完全嵌入网格，优化前 SWAP 门数最小）；中难度档取网格距离 2-3 "
          "的随机交互对；高难度档为远距交互对优先的高密度结构。每条线路的逻辑比特标签经"
          "随机置换，避免恒等布局即最优的退化情形。"),
    ("p", "单独构建该测试集的原因在于：既有三个基准均面向映射质量评测，其比特数与难度的"
          "分布并不均匀（如 Benchpress 的三十条中中小规模线路占多数），不适合刻画运行时间"
          "随规模与难度的变化规律；运行时间评测需要的是“比特数大中小均有、每档内难度高低"
          "均有”的受控网格。该测试集与本节前述 287 原生池在生成源上同源（均以 287 耦合图"
          "生成），但用途不同：287 原生池用于训练图池扩充，运行时间测试集用于扩展性评测，"
          "二者互不混用。"),
    ("p", "运行时间评测的指标为端到端运行时间：对单条线路从解析开始到映射产物（平台基"
          "线路线路）落盘结束的真实墙钟耗时，逐线路单次运行记录、不取平均，并按比特数"
          "（图5.7）与优化前 SWAP 门数（图5.8）两种排序展示扩展性。其中优化前 SWAP 门数"
          "定义为平凡布局（不做布局优化）加 SABRE 路由的交换门数，用以度量线路的固有路由"
          "难度。"),
]

# 5.1 测评基准汇总（Word 三线表）
BENCH_TABLE = (
    "tbl3", "表5.1  测评基准汇总",
    ["基准", "规模与构成", "用途"],
    [["Benchpress（公开）", "656 条，取设备规模内全部线路，覆盖 8 类基准结构"
                            "（大整数运算、Clifford、QAOA、QFT 等）",
      "主评测集，与 Qiskit 官方口径一致"],
     ["MQTBench（公开）", "36 类程序化生成基准", "多类线路结构的横向覆盖"],
     ["QUEKO（公开）", "BIGD / BNTF / BSS 三个子集", "高密度映射问题的极端场景"],
     ["天衍-287 原生池（自建）", "2 万条：queko287 子集（20/40/60/80/105 比特 × 深度 10-80"
                                  " × 密度 3 档 × 种子网格，附带已知最优映射）与 native287 子集"
                                  "（层化随机线路）", "训练针对 287 拓扑的布局策略"],
     ["运行时间测试集（自建）", "90 条 = 6 比特数档 {10, 20, 40, 60, 80, 105} × 3 难度档"
                                " × 5 种子", "运行时间扩展性评测"]],
    None,
    ["l", "l", "l"],
)

# 5.2 实验硬件与软件环境（Word 三线表）
ENV_TABLE = (
    "tbl3", "表5.2  实验硬件与软件环境",
    ["项目", "配置"],
    [["处理器与内存", "桌面级十六核 CPU；64 GB 内存"],
     ["GPU", "NVIDIA RTX 5090（24 GB 显存，CUDA 12.8）"],
     ["操作系统与 Python", "Windows 11；Python 3.13.7"],
     ["交付管线依赖", "torch 2.11（cu128）、numpy 2.x、networkx 3.6、pyyaml；"
                       "numba 0.67（可选加速）"],
     ["对照实验依赖", "qiskit 2.5、qiskit-aer、pytket 1.x、pytket-qiskit、mqt.bench 2.2"]],
    None,
    ["c", "l"],
)

# 结果与分析各板块的过渡句（lead）与阶段性结论句（closing）
SECTION_LEADS = {
    "train": "本章开篇首先回答“模型能否训练、训练是否可复现”的问题——这是后续一切结论的"
             "前提：若布局策略无法稳定收敛，分层性能与对比结论均无从谈起。",
    "strata": "训练可行之后，下一个问题是“方案在各类规模的线路上表现如何”。本节按 ≤10、"
              "11-50、51-105 三层展示保真度与深度，重点观察性能随规模的变化结构与短板所在。",
    "noise": "分层性能中本方案相对盲目基线的优势，可能来自噪声感知，也可能来自布局算法本身。"
             "本节以三组受控对照——缺陷敏感性扫描、跨拓扑零重训迁移与布局贡献实验——直接检验"
             "噪声感知机制的有效性，并把优势归因到布局。",
    "speed": "质量之外，效率同样是可用性的前提。本节在 5.1 节构建的运行时间测试集上，回答"
             "“各方案的端到端运行时间如何随规模与难度变化”的问题。",
    "compare": "前述各节分别考察了训练、分层、机制与效率。本节把视角拉远，回答“相对成熟"
               "方案，本方案整体处于什么位置”——先看 587 条线路的总体账，再看模型侧改进的"
               "逐线路细节。",
}
SECTION_CLOSINGS = {
    "train": "综上，两次独立训练的一致收敛曲线与单轮秒级耗时表明训练过程可行且可复现，"
             "LAUREL 残差进一步将收敛轮数压缩至 89 轮而不牺牲质量——模型侧的前提已经成立，"
             "下面看其在各类场景下的实际表现。",
    "strata": "分层结果表明本方案在小中规模上的保真度优势稳定，大线路层的深度差距是当前"
              "主要短板；而这一优势的来源是否为噪声感知机制本身，还需专门的对照证据——"
              "下一节回答该问题。",
    "noise": "三组对照分别从“规避热区”“跨设备迁移”与“布局归因”三个角度确认了噪声感知机制"
             "的有效性。质量层面的证据链到此闭合，下一节转向效率问题：方案的运行时间是否可接受。",
    "speed": "综上，自研纯管线在全部 90 条线路上保持秒级端到端耗时且随难度平稳，本方案布局"
             "加 O3 路由的混合模式以分钟级预算换取质量增益，两者形成速度与质量的两档配置。"
             "效率问题回答完毕，下一节回到质量维度，与成熟方案做总账式的综合对比。",
    "compare": "综合对比表明：在噪声现实的芯片上，本方案的保真度整体超过 pytket 与全部盲目"
               "基线、接近噪声感知 O3。各维度的结论将在下一节归纳。",
}

# 5.4.4 运行时间（全新小节：图5.6 按比特数、图5.7 按优化前 SWAP、表5.2 三线表）
RUNTIME_SECTION = [
    ("p", "按比特数升序排列的运行时间对比如图5.7所示。可以看到三个层次的现象：其一，全部"
          "六个方案中，自研纯管线（本方案的交付形态）在 10 至 105 比特全程保持秒级端到端"
          "耗时（0.5-6.8 秒），且随比特数上升的斜率最缓，即使在优化前 SWAP 门数超过五百的"
          "最难线路上也在十秒以内；其二，本方案布局加 O1 路由的混合模式更快（0.7-4.3 秒）；"
          "其三，噪声感知 O3 与 pytket 在多数线路上为亚秒到数秒级，但在 105 比特低难度线路"
          "上出现明显耗时尖峰（噪声感知 O3 达 16.8 秒），该现象的原因见下文的特殊现象分析。"),
    ("fig", "图5.7  运行时间对比（90 条规模-难度网格，按比特数升序；纵轴为端到端"
            "运行时间，逐线路单次记录、不取平均）",
     "tables/report_figures/thesis/图5.7_运行时间对比_按比特数.png", 15.0),
    ("p", "按优化前 SWAP 门数升序排列的运行时间对比如图5.8所示，它直接展示耗时与固有路由"
          "难度的关系。多数方案的耗时随难度上升而增加（自研纯管线与优化前 SWAP 门数的相关"
          "系数为 +0.76，本方案布局加 O3 路由的筛选模式为 +0.94）；另有两个例外值得注意："
          "噪声感知 O3 的耗时与难度呈负相关（-0.72）——低难度端的蛇形路径线路反而最慢；"
          "本方案布局加 O1 路由的耗时对难度几乎不敏感（0.7-4.3 秒的窄带）。"),
    ("fig", "图5.8  运行时间对比（90 条规模-难度网格，按优化前 SWAP 门数升序；计时口径"
            "同图5.7）",
     "tables/report_figures/thesis/图5.8_运行时间对比_按优化前SWAP.png", 15.0),
    ("p", "噪声感知 O3 在低难度线路上反而更慢的原因分析如下。已验证的部分：同规模的 105 "
          "比特线路上，低难度档（蛇形路径）的编译耗时达 8-17 秒，而高难度档仅 0.04-0.1 秒，"
          "该差距稳定存在于全部五个种子。推测性的机制解释是：蛇形路径线路整体构成一个连通的"
          "线性函数块，qiskit 高级优化档的线性函数收集与再合成在该规模（105×105 矩阵）上"
          "耗时秒级，而散乱远距交互只能形成小块、合成几乎免费——即“结构越规整，可合并的"
          "优化块越大，再合成越贵”。该解释未做逐遍（pass）级剖析，故标注为推测。"),
]


def build_ch5(blocks, dataset):
    secs, order = split_by_h2(blocks)
    out = [("h1", "第5章 实验与结果分析")]
    # 5.1 数据集与评测体系（自 3.4 移入 + 原 5.1 统计概貌段 + 运行时间测试集）
    out.append(("h2", "5.1 数据集与评测体系"))
    out += dataset
    s51 = secs["5.1 实验环境与设置"]
    # 统计概貌段原在“5.2 训练过程”源节中，按内容归属移入 5.1
    stat = [b for b in secs["5.2 训练过程"]
            if b[0] == "p" and b[1].startswith("评测数据集的统计概貌")]
    out += stat
    out += RUNTIME_SET_BLOCKS
    out.append(("p", "上述五个数据来源的规模、构成与用途汇总见表5.1。"))
    out.append(BENCH_TABLE)
    # 5.2 实验设置（原 5.1 其余）
    out.append(("h2", "5.2 实验设置"))
    s51_keep = [b for b in s51]
    out.append(s51_keep[0])  # 环境段
    out.append(ENV_TABLE)
    out += s51_keep[1:]
    # 原 5.3 消融实验整节删除（含表5.1 消融实验汇总），后续节号顺延
    # 5.3 真机验证（原 5.4 / 源 5.7）
    out.append(("h2", "5.3 真机验证"))
    out += secs["5.7 真机验证"]
    # 5.4 结果与分析（次节顺序：训练过程→分层性能→噪声感知证据→运行时间→
    # 与成熟方案的综合对比→结果归纳与讨论；每板块以过渡句开篇、阶段性结论收束）
    out.append(("h2", "5.4 结果与分析"))
    out.append(("h3", "5.4.1 训练过程"))
    out.append(("p", SECTION_LEADS["train"]))
    # 训练设置表删除：其内容已在 4.1.3/4.1.4/4.1.8 节详述，此处不再重复；
    # 统计概貌段已移入 5.1
    s52 = [b for b in secs["5.2 训练过程"]
           if not (b[0] == "tab" and "训练设置" in b[1])
           and not (b[0] == "p" and b[1].startswith("评测数据集的统计概貌"))]
    out += s52
    out.append(("p", SECTION_CLOSINGS["train"]))
    out.append(("h3", "5.4.2 分层性能"))
    out.append(("p", SECTION_LEADS["strata"]))
    out += secs["5.3 分层性能"]
    out.append(("p", SECTION_CLOSINGS["strata"]))
    out.append(("h3", "5.4.3 噪声感知的有效性证据"))
    out.append(("p", SECTION_LEADS["noise"]))
    out += secs["5.5 噪声感知的有效性证据"]
    # 布局贡献实验自 5.4.5 移入（噪声感知优势的布局归因证据）
    s56 = secs["5.6 与成熟方案的综合对比"]
    idx = next(i for i, b in enumerate(s56)
               if b[0] == "p" and b[1].startswith("为隔离布局算法的贡献"))
    out.append(("p", "前两组对照分别证明策略会规避热区、且该能力可跨设备迁移；第三组对照"
                     "回答“相对基线的优势究竟来自布局还是来自其余环节”——布局贡献实验把路由器"
                     "固定为 qiskit SABRE，仅替换布局来源。"))
    out += s56[idx:]
    out.append(("p", SECTION_CLOSINGS["noise"]))
    out.append(("h3", "5.4.4 运行时间"))
    out.append(("p", SECTION_LEADS["speed"]))
    out += RUNTIME_SECTION
    # 保留旧运行时间小节中与具体实验无关的通用分析（耗时构成、规模外推）
    s54_keep = [b for b in secs["5.4 运行时间"] if b[0] == "p" and (
        b[1].startswith("耗时的构成") or
        b[1].startswith("200 比特以内的线性外推"))]
    out += s54_keep
    out.append(("p", SECTION_CLOSINGS["speed"]))
    out.append(("h3", "5.4.5 与成熟方案的综合对比"))
    out.append(("p", SECTION_LEADS["compare"]))
    out += s56[:idx]
    out.append(("p", SECTION_CLOSINGS["compare"]))
    out.append(("h3", "5.4.6 结果归纳与讨论"))
    out += build_synthesis(secs["5.8 结果分析与讨论"], secs["5.10 本章小结"])
    return apply(out)


def build_synthesis(s58, s510):
    """原 5.8 + 5.10 重构为归纳式收束。"""
    lead = ("本节将前述实验现象归纳为规律并收束全章。先看三组结构性结论，"
            "再作整体讨论与展望。")
    out = [("p", lead)]
    for b in s58:
        out.append(b)
    # 收束段（基于原 5.10 改写，反映新章节结构；不提及已删除的消融实验）
    out.append(("p", "综观全章，各实验环节形成一条完整的证据链：数据集与评测体系规定了"
                     "比较的公平边界，实验设置固定了可复现的操作口径，真机验证确认了交付的"
                     "可信，而分层性能、运行时间、噪声证据与成熟方案对比则从四个维度刻画了"
                     "方案的能力与边界。所有原始数据均按逐线路JSONL 落盘，任何结论均可回溯复核。"))
    return out


# ================================================================ 结论（用户修改版）
CONCLUSION = [
    ("h1x", "结论"),
    ("p", "本文面向天衍-287 超导处理器设计并实现了 QTrial 量子线路映射优化方案，主要成果如下。"),
    ("p", "在方案设计上，构建了“解析—图构建—强化学习布局—自研 SABRE 路由—置换重写后处理—"
          "导出”的完整自研管线，交付系统零 qiskit 依赖，覆盖天衍平台 QCIS 指令集与真机提交全链路；"
          "布局环节将 QAP 建模为图注意力网络加指针网络的强化学习策略，以噪声加权距离构造稀疏"
          "终局奖励，以“路由加后处理的真实结果”作为候选决胜口径，并设计三级失败阶梯保证鲁棒性。"),
    ("p", "在方法创新上，本文的主要创新可归纳为以下四个方面。"),
    ("p", "第一，提出噪声感知的强化学习布局算法，将 QAP 求解从静态拓扑优化推进到噪声现实下可迁移"
          "的构造式决策。现有学习型布局工作多依赖逐规模训练、噪声信息利用不足，且训练目标与最终"
          "评测口径之间存在偏差；VQM 等噪声感知方法则以静态排序规则一次性构造布局。本文以噪声"
          "加权有效距离构造 QAP 代价作为稀疏终局奖励，使策略在构造布局时即感知不同物理比特与"
          "耦合器的噪声差异；设备节点特征按单台设备自身统计量归一化，网络学到的是“相对本设备"
          "其他比特”的质量差异而非绝对刻度，从而实现同一策略在多拓扑、多设备间的零重训迁移；"
          "解码阶段采用多起点采样与自适应局部搜索，并以“路由加后处理的真实结果”作为候选决胜"
          "口径，使训练目标与评测口径完全一致。实验证据方面，跨拓扑实验以同一策略权重在 6×6 "
          "网格、heavy-hex 与 Sycamore 三种几何上完成零重训迁移；缺陷敏感性扫描表明，随着缺陷"
          "误差放大，盲目 O1 的 2Q 误差加权均值从 0.0035 恶化到 0.0110，而本方案稳定在 "
          "0.0030—0.0033；布局贡献实验在路由器同为 qiskit SABRE 的前提下，本方案布局在 "
          "Benchpress、MQTBench 与 QUEKO 三个基准上保真度全面第一。把噪声信息同时注入布局奖励、"
          "候选评分与保真度估计，并以“训练口径等于评测口径”消除了学习型布局常见的系统性偏差。"),
    ("p", "第二，自主复现并工程加速 SABRE 路由器，构建保语义的置换重写后处理栈，实现百比特级秒级"
          "路由与硬件可执行性守护。工业编译器中的 SABRE 虽被广泛采用，但其工程实现通常与特定"
          "编译器栈绑定，且加速改造容易改变算法行为。本文完整复现 SABRE 的前沿层并行执行、扩展集"
          "前瞻、逐门衰减权重与交换历史惩罚等机制，并通过前沿层与扩展集增量维护、候选边由邻接表"
          "构建并排序、交换历史由字典改为 numpy 矩阵、候选×前沿距离矩阵由 Numba 融合内核直接产出"
          "等四项改造，将百比特级线路的路由时间压缩至秒级；上述改造均与改造前实现做了逐位 A/B "
          "验证，在相同输入、相同随机种子下输出逐字节一致，确保纯工程加速不改变算法行为。路由之后，"
          "后处理栈以共轭推挤、对消解、尾部吸收与平台基分解四层操作消化 SWAP 门，每一层均为保语义"
          "的等价变换，全程由态矢量等价测试常驻守护，并设置连通性红线，防止“全局置换重标记”破坏"
          "执行时刻的邻近约束。既实现了百比特级线路的秒级路由，又把“语义等价”与“硬件可执行”"
          "分开验证，使自研管线在不依赖 qiskit 的前提下具备生产级可执行性。"),
    ("p", "第三，引入 LAUREL 可学习残差与 QuEst 风格保真度预测器，以即插即用方式同时改善训练效率"
          "与保真度估计口径。在训练效率方面，本文将 LAUREL 可学习增广残差层接入 GAT 编码器与图 "
          "Transformer 编码器的残差位置，采用 RW+LR 组合，低秩维为 4，初始化近零，旧检查点可零"
          "修改加载。其机理在于：RW 变体为残差分支提供逐层独立的信号强度控制，LR 变体在近乎零"
          "额外参数下为残差流增加自由度，二者共同改善深层 GAT 训练早期的梯度传播。实测收敛轮数"
          "由 400 轮压缩至 89 轮，训练效率约 4.5 倍，且在 QUEKO BIGD 30 条线路上保真度均值由 "
          "0.4091 提升至 0.4216，提升 3.1%，逐线路 21 胜 9 负，说明效率提升未牺牲质量，反而略有"
          "增益。在保真度估计方面，针对乘积模型完全忽略 T1/T2 退相干随门时长累积效应的盲区，"
          "本文实现 QuEst 风格图 Transformer 预测器：将线路转换为门图，节点为线路全部指令，边为"
          "依赖 DAG，节点 15 维特征包括门类型 one-hot、比特归一化索引、T1/T2、门错误率与读出误差；"
          "预测器为 2 层 4 头图 Transformer 加均值池化与三层全连接回归，输出经 sigmoid 压至 [0,1]；"
          "训练标签由 PST 方法生成，门时长按 1Q 20ns、2Q 200ns 折算解相干率，保证标签确实携带"
          "退相干信息。预测器以 CLI 参数可选接入，未提供权重时自动回退乘积模型，不改变交付管线"
          "的零 qiskit 依赖。以极低侵入的方式同时改善训练效率与保真度估计口径，弥补了乘积模型对"
          "退相干的盲区。"),
    ("p", "第四，建立口径一致的评测体系与三层真机验证链，使方案从“能映射”推进到“可复核、可交付”。"
          "映射方案的学术比较长期存在口径分叉：若将 CNOT 保留为复合门，深度与门数天然占优；若以"
          "不同噪声模型计算保真度，跨方法比较即失真。本文强制所有方法（含外部编译器）的输出统一"
          "经过平台基分解后再计算指标，随机种子固定，逐线路原始数据落盘 JSONL，分层给出均值、"
          "截尾均值与中位数三种统计口径，并支持断点续跑，使 587 条线路的分层评测、布局贡献实验"
          "与胜场统计具有可复核性。在交付可信层面，本文建立三层真机验证链：第一层，态矢量等价"
          "测试，映射前后线路作用于相同初态的态矢量一致，随 pytest 回归套件常驻运行；第二层，"
          "平台规则性校验，QCIS 提交前调用平台校验流程，曾真实拦截“折叠实验”事故；第三层，真机"
          "实测对照，映射线路提交真机以 12000 shots 执行，将实测计数按测量位序、真实标签与最终"
          "布局回映射到逻辑空间，与经典模拟分布比较，判定口径为经典保真度不低于 0.98 且总变差"
          "距离不超过 0.05。以语义等价与硬件可执行分开验证的原则，分别拦截数学错误、指令非法与"
          "真机行为偏差三类事故，形成从训练、评测到真机验证的完整工程闭环。"),
    ("p", "在实验验证上，587 条线路的分层评测表明方案总体估计保真度超过 pytket 与盲目基线，布局"
          "贡献实验在 Qiskit 官方测评集 Benchpress 上验证了保真度优势；缺陷敏感性扫描与跨拓扑实验"
          "验证了噪声感知机制的有效性；真机对照确认了映射线路的硬件可执行性与语义等价。各项指标"
          "数据均以逐线路原始记录留档，可供评审任意抽查复核。"),
    ("p", "本方案的不足与后续方向：大线路层的深度指标与 qiskit 高级优化档位仍有约一个数量级的"
          "差距，根源为门级再合成机制的缺失，拟结合学习化路由与近邻链展开补齐；QuEst 预测器与"
          "乘积模型的适用边界尚需更大规模的标注数据界定；200 比特级线路的管线压力测试与图池扩充"
          "训练的实验仍在进行中。"),
    ("p", "面向更长远的部署场景，方案还预留了两条扩展路线：其一是天衍平台家族的多机型适配——"
          "天衍云平台除 287 外还提供 176（66 比特）、294 与 504 等多款处理器，本方案的设备抽象已"
          "支持以 live 模式切换目标机器，仅需切换设备配置即可完成映射迁移；其二是与平台原生编译器"
          "的协作——布局注入接口已打通，可将本方案的布局结果作为初始映射交给平台原生管线二次优化，"
          "实现“学习布局 + 原生路由”的混合模式。"),
]


# ================================================================ 附录 B（图表清单更新）
APPENDIX_B_NEW = [
    ("p", "图3.1（系统总体架构图，见 3.1 节）：内容为解析、图构建、布局、路由、后处理五段主链路"
          "与噪声感知、真机验证两条支撑链路的数据流向，已随文档生成。"),
    ("p", "表5.1（测评基准汇总，见 5.1 节）：Benchpress、MQTBench、QUEKO 三个公开基准与天衍-287 "
          "原生池、运行时间测试集两个自建数据集的规模、构成与用途汇总。"),
    ("p", "表5.2（实验硬件与软件环境，见 5.2 节）：工作站硬件、操作系统与 Python 版本、交付管线"
          "依赖与对照实验依赖清单。"),
    ("p", "图5.1（训练学习曲线，见 5.4.1 节）：两次完整训练的验证集平均代价随轮次收敛曲线与单轮"
          "耗时曲线，展示训练可行性与可复现性。"),
    ("p", "图5.2（分层保真度，见 5.4.2 节）：≤10/11-50/51-105 三层估计保真度均值对比（6 方法）。"),
    ("p", "图5.3（分层深度，见 5.4.2 节）：三层线路深度截尾均值对比，大层为对数刻度。"),
    ("p", "图5.4（缺陷敏感性扫描，见 5.4.3 节）：2Q 误差加权均值随缺陷放大倍数的变化（4 方法）。"),
    ("p", "图5.5（跨拓扑噪声规避，见 5.4.3 节）：三拓扑上 2Q 误差加权均值对比（同一策略零重训）。"),
    ("p", "图5.6（布局贡献对比，见 5.4.3 节）：本方案布局 + O1/O3 路由 vs pytket 与盲目/"
          "噪声感知 O1/O3，三基准 × 保真度/深度（截尾均值）。"),
    ("p", "图5.7（运行时间对比·按比特数，见 5.4.4 节）：90 条规模-难度网格线路的端到端运行"
          "时间折线（6 方案，对数刻度）。"),
    ("p", "图5.8（运行时间对比·按优化前 SWAP 门数，见 5.4.4 节）：同一 90 条线路按固有路由"
          "难度升序的运行时间折线（6 方案，对数刻度）。"),
    ("p", "图5.9（总体对比，见 5.4.5 节）：587 条线路总体保真度（加权均值）与逐线路胜场数。"),
    ("p", "图5.10（LAUREL 逐线路对比，见 5.4.5 节）：30 条线路保真度差排序柱状图。"),
    ("p", "以上图表均已在正文对应小节中被先行引用（“如图5.x所示”“见表5.x”），图序按章独立编号，"
          "图题置于图下方居中，表题置于表上方居中。"),
]


# ================================================================ 主流程
def main():
    # 第 1 章：保留原文，仅做节号/术语微调（见 CH1_PATCH）
    ch1_blocks = []
    ch2_blocks = []
    in_ch2 = False
    for b in content1.CH1:
        if b[0] == "h1" and b[1].startswith("第2章"):
            in_ch2 = True
        (ch2_blocks if in_ch2 else ch1_blocks).append(b)
    ch1_blocks = patch_ch1(ch1_blocks)

    ch2_new = build_ch2_final(ch2_blocks[1:])  # 去掉第2章 h1 自身
    # build_ch2_final 已含 h1

    ch34 = content2.CH34
    ch3_blocks, ch4_blocks = [], []
    in_ch4 = False
    for b in ch34:
        if b[0] == "h1" and b[1].startswith("第4章"):
            in_ch4 = True
            ch4_blocks.append(b)
            continue
        (ch4_blocks if in_ch4 else ch3_blocks).append(b)
    ch3_new = build_ch3(ch3_blocks[1:])
    ds = dataset_blocks(ch3_blocks[1:])
    ch4_new = build_ch4(ch4_blocks[1:])

    ch5 = content3.CH5
    ch5_blocks = [b for b in ch5]
    ch5_new = build_ch5(ch5_blocks[1:], ds)

    appendix = content3.APPENDIX
    # 附录 A/C/D 原样，B 替换
    a_new = []
    cur = None
    for b in appendix:
        if b[0] == "h1x":
            cur = b[1]
        if cur == "附录B 图表清单":
            if b[0] == "h1x":
                a_new.append(b)
            continue
        a_new.append(b)
    a_new += [("h1x", "附录B 图表清单")] + APPENDIX_B_NEW

    data1 = {"abstract_cn": ABSTRACT_CN_FULL, "abstract_en": ABSTRACT_EN_FULL,
             "kw_cn": content1.ABSTRACT_KEYWORDS_CN,
             "kw_en": content1.ABSTRACT_KEYWORDS_EN,
             "refs": content1.REFS, "ch1": ch1_blocks, "ch2": ch2_new}
    data2 = {"ch3": ch3_new, "ch4": ch4_new}
    data3 = {"ch5": ch5_new, "conclusion": CONCLUSION, "appendix": a_new}

    for name, data in (("content1.json", data1), ("content2.json", data2),
                       ("content3.json", data3)):
        with open(OUT / name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"written {name}: blocks="
              f"{sum(len(v) for k, v in data.items() if isinstance(v, list))}")


if __name__ == "__main__":
    main()
