"""系统总体架构图（作品报告书图 3.1，中文标注）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
BLUE = "#2a78d6"
LIGHT = "#cde2fb"
BOX = "#f7f6f3"
EDGE = "#c3c2b7"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 12.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor(SURFACE)


def box(x, y, w, h, title, lines, fc=BOX, ec=EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.08",
                                facecolor=fc, edgecolor=ec, linewidth=1.2))
    ax.text(x + w / 2, y + h - 0.42, title, ha="center", va="center",
            fontsize=11, color=INK, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.95 - i * 0.5, ln, ha="center",
                va="center", fontsize=8, color="#52514e")


def arrow(x1, y1, x2, y2, label=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color="#52514e"))
    if label:
        ax.text((x1 + x2) / 2 + 0.18, (y1 + y2) / 2 + 0.12, label,
                fontsize=8, color="#52514e")


# 左列（主链路）
box(1.6, 12.9, 6.8, 0.95, "OpenQASM 2.0 线路输入", [])
arrow(5, 12.9, 5, 12.15)
box(1.6, 10.9, 6.8, 1.25, "自研解析器（qtrail/pure/qasm.py）",
     ["自定义 gate、ccx 分解、cp(λ) 扩展"])
arrow(5, 10.9, 5, 10.15)
box(1.6, 8.9, 6.8, 1.25, "电路 IR（qtrail/pure/circuit.py）",
     ["Inst + Circuit，惰性依赖 DAG"])
arrow(2.8, 8.9, 2.8, 7.9)
arrow(7.2, 8.9, 7.2, 7.9)
box(0.4, 6.6, 4.8, 1.3, "程序图（逻辑侧）",
    ["6 维节点特征", "交互权重 w_ij"], fc=LIGHT)
box(4.8, 6.6, 4.8, 1.3, "设备规格 DeviceSpec（物理侧）",
    ["7 维噪声特征", "D_topo / D_noise / D_eff"], fc=LIGHT)
arrow(2.8, 6.6, 3.6, 5.6)
arrow(7.2, 6.6, 6.4, 5.6)
box(1.6, 4.3, 6.8, 1.3, "布局（RL）：GAT + 指针网络 + 多起点 + 局部搜索",
     ["REINFORCE 稀疏奖励，噪声感知 QAP 代价", "失败阶梯：RL → 谱启发式 → 平凡布局"], fc="#b7d3f6")
arrow(5, 4.3, 5, 3.55, "初始映射 π")
box(1.6, 2.3, 6.8, 1.25, "路由（自研 SABRE）",
     ["front layer + 衰减 + 历史惩罚", "增量维护 + Numba 融合内核"])
arrow(5, 2.3, 5, 1.55, "含 SWAP 的物理线路")
box(1.6, 0.4, 6.8, 1.15, "后处理栈 + 平台基分解",
     ["推挤 / 消解 / 尾部吸收 / U3 再合成 → [rz, sx, x, cz]"])
# 输出
arrow(5, 0.4, 5, 0.0)

# 右列（输出与验证）
arrow(8.6, 0.95, 9.5, 0.95)
box(8.9, 0.35, 1.1, 1.2, "", ["QASM", "QCIS", "指标"], fc=LIGHT)
ax.text(8.5, 2.6, "真机验证链\n（态矢量等价 /\n平台校验 /\n真机对照）",
        ha="center", va="center", fontsize=8.5, color="#52514e",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0efec",
                  edgecolor=EDGE, linewidth=0.8))
ax.text(8.5, 7.2, "噪声感知：\nD_eff = D_topo\n+ λ(D_noise − D_topo)",
        ha="center", va="center", fontsize=8.5, color="#52514e",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0efec",
                  edgecolor=EDGE, linewidth=0.8))

fig.savefig("tables/report_figures/thesis/图3.1_系统架构图.png", dpi=200,
            bbox_inches="tight", facecolor=SURFACE)
print("架构图已生成")
