# -*- coding: utf-8 -*-
"""验证挂载布局是否真的被 O3 尊重（用故意很差的布局对照）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit import qasm2, transpile
from qiskit.transpiler import TranspileLayout, Layout

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pipeline.baselines import build_target
from qtrail.utils.qasm_io import PLATFORM_BASIS, strip_measurements

spec = build_tianyan287_spec(load_device_config())
target = build_target(spec)
qc0, _ = strip_measurements(qasm2.load(
    "data/Queko/BIGD/20QBT_45CYC_.0D1_.1D2_0.qasm"))
n = qc0.num_qubits
coords = spec.coords
usable = list(range(spec.n))

# 贪心最远点采样：选相互距离最大的 20 个物理位（故意最差的布局）
sel = [int(np.random.default_rng(0).choice(usable))]
for _ in range(n - 1):
    sel.append(max((p for p in usable if p not in sel),
                   key=lambda p: min(np.linalg.norm(coords[p] - coords[q])
                                     for q in sel)))
bad = {qc0.qubits[i]: int(sel[i]) for i in range(n)}

for name, lyt in [("坏布局挂载", bad),
                  ("平凡挂载", {qc0.qubits[i]: i for i in range(n)})]:
    init_lay = Layout(lyt)
    tl = TranspileLayout(initial_layout=init_lay,
                         input_qubit_mapping=lyt, final_layout=init_lay)
    qc1 = qc0.copy()
    qc1._layout = tl
    b = transpile(qc1, target=target, basis_gates=PLATFORM_BASIS,
                  optimization_level=3, seed_transpiler=42)
    print(f"{name}: twoq={b.count_ops().get('cz', 0)}")
