# -*- coding: utf-8 -*-
"""验证：布局挂到线路对象上后，O3 是否尊重该布局且 VF2PostLayout 仍运行。

对照三组（queko 20QBT 样例）：
  A: 原生 O3（自选布局 + VF2PostLayout）
  B: qc.layout 挂我们的布局 → O3（目标：布局被采用 + VF2PostLayout 运行）
  C: initial_layout 注入 → O3（已知：VF2PostLayout 被跳过）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2, transpile
from qiskit.transpiler import TranspileLayout, Layout

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pipeline.baselines import build_target
from qtrail.utils.qasm_io import PLATFORM_BASIS, strip_measurements

spec = build_tianyan287_spec(load_device_config())
target = build_target(spec)

SRC = "data/Queko/BIGD/20QBT_45CYC_.0D1_.1D2_0.qasm"
qc0, _ = strip_measurements(qasm2.load(SRC))
n = qc0.num_qubits

# 简单布局：平凡（测试用）
layout_ours = {qc0.qubits[i]: i for i in range(n)}

# A: 原生 O3
a = transpile(qc0, target=target, basis_gates=PLATFORM_BASIS,
              optimization_level=3, seed_transpiler=42)
print("A 原生O3: twoq=", a.count_ops().get("cz", 0))

# B: 布局挂到线路对象
init_lay = Layout(layout_ours)
tl = TranspileLayout(initial_layout=init_lay,
                     input_qubit_mapping=layout_ours,
                     final_layout=init_lay)
qc1 = qc0.copy()
qc1._layout = tl
b = transpile(qc1, target=target, basis_gates=PLATFORM_BASIS,
              optimization_level=3, seed_transpiler=42)
print("B 挂布局+O3: twoq=", b.count_ops().get("cz", 0))

# C: initial_layout 注入
c = transpile(qc0, target=target, basis_gates=PLATFORM_BASIS,
              optimization_level=3, initial_layout=layout_ours,
              seed_transpiler=42)
print("C 注入+O3: twoq=", c.count_ops().get("cz", 0))
