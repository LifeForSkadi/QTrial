"""训练 QuEst 图 Transformer 保真度预测器（arXiv:2210.16724）。

用小型网格设备（默认 3×3 = 9 比特）生成随机平台基线路（rz/sx/x/cz），
在**含 T1/T2 解相干 + 门 depolarizing + 读出误差**的噪声模型下用
qiskit AerSimulator 跑 PST（拼接逆线路后全零输出占比）作标签，训练
``qtrail/models/fidelity.py`` 的 ``FidelityPredictor``。

关键点：噪声模型必须含 T1/T2 项——否则 PST 退化为乘积模型，预测器
学不到比现有 ``estimate_fidelity``（只用 ε_1q/ε_2q/ε_ro，忽略 T1/T2）
更多的信息。训练用 qiskit 仅在实验侧（scripts/），交付管线零 qiskit。

用法：
    python scripts/train_fidelity.py --n 200 --epochs 10 --out checkpoints
依赖：qiskit + qiskit-aer（噪声模拟）。
输出：checkpoints/fidelity_<device>_best.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 门时长（ns），仅用于把 T1/T2 转成解相干率
GATE_TIME_1Q_NS = 20.0
GATE_TIME_2Q_NS = 200.0


def random_platform_circuit(spec, rng, depth=8):
    """随机平台基线路，同时产出纯 Circuit 与 qiskit QuantumCircuit。"""
    from qtrail.pure.circuit import Circuit, Inst
    from qiskit import QuantumCircuit

    n = spec.n
    circ = Circuit(n, name="rand_fid")
    qc = QuantumCircuit(n)
    edges = spec.edges
    for _ in range(depth):
        for q in range(n):
            r = float(rng.random())
            if r < 0.4:
                ang = float(rng.uniform(0.0, 2.0 * np.pi))
                circ.append(Inst("rz", (q,), (ang,)))
                qc.rz(ang, q)
            elif r < 0.6:
                circ.append(Inst("sx", (q,)))
                qc.sx(q)
            elif r < 0.7:
                circ.append(Inst("x", (q,)))
                qc.x(q)
        for _ in range(max(1, n // 2)):
            a, b = edges[int(rng.integers(0, len(edges)))]
            circ.append(Inst("cz", (a, b)))
            qc.cz(a, b)
    return circ, qc


def build_noise_model(calib, n):
    """由 CalibrationData 构造噪声模型：T1/T2 解相干 + 门 depolarizing + 读出。"""
    from qiskit_aer.noise import (NoiseModel, ReadoutError,
                                  amplitude_damping_error,
                                  depolarizing_error, phase_damping_error)

    noise = NoiseModel()
    t1 = np.asarray(calib.t1, dtype=np.float64) * 1000.0  # us -> ns
    t2 = np.asarray(calib.t2, dtype=np.float64) * 1000.0
    err_1q = np.asarray(calib.err_1q, dtype=np.float64)
    err_ro = np.asarray(calib.err_ro, dtype=np.float64)
    err_2q = calib.err_2q

    for q in range(n):
        gamma = float(1.0 - np.exp(-GATE_TIME_1Q_NS / max(t1[q], 1e-9)))
        lam = float(1.0 - np.exp(-GATE_TIME_1Q_NS / max(t2[q], 1e-9)))
        e1 = amplitude_damping_error(gamma).compose(phase_damping_error(lam))
        e1 = e1.compose(depolarizing_error(float(err_1q[q]), 1))
        for g in ("rz", "sx", "x"):
            noise.add_quantum_error(e1, g, [q])
    for (a, b), e in err_2q.items():
        noise.add_quantum_error(depolarizing_error(float(e), 2), "cz", [a, b])
    for q in range(n):
        p = float(err_ro[q])
        noise.add_readout_error(ReadoutError([[1.0 - p, p], [p, 1.0 - p]]), [q])
    return noise


def pst_label(qc, calib, n, shots=1024):
    """拼接逆线路后跑 AerSimulator，返回全零输出占比（PST）。"""
    from qiskit_aer import AerSimulator

    full = qc.copy()
    full.barrier()
    full.compose(qc.inverse(), inplace=True)
    full.measure_all()

    sim = AerSimulator(noise_model=build_noise_model(calib, n))
    counts = sim.run(full, shots=shots).result().get_counts()
    return float(counts.get("0" * n, 0)) / float(shots)


def main():
    ap = argparse.ArgumentParser(description="训练 QuEst 保真度预测器")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--n", type=int, default=500, help="训练样本数")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--depth", type=int, default=8, help="随机线路层数")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    import torch
    from qtrail.devices import build_grid_family_spec
    from qtrail.models.fidelity import FidelityPredictor, build_gate_graph

    spec = build_grid_family_spec(args.rows, args.cols, seed=args.seed)
    calib = spec.calib
    rng = np.random.default_rng(args.seed)

    print(f"[train_fidelity] device={spec.name} n={spec.n} samples={args.n}")
    t0 = time.time()
    xs, ys = [], []
    for _ in range(args.n):
        circ, qc = random_platform_circuit(spec, rng, depth=args.depth)
        x, adj, mask = build_gate_graph(circ, calib)
        xs.append((x, adj, mask))
        ys.append(pst_label(qc, calib, spec.n))
    print(f"[train_fidelity] data done in {time.time() - t0:.1f}s "
          f"mean PST={float(np.mean(ys)):.4f}")

    dev = args.dev
    model = FidelityPredictor().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for (x, adj, mask), y in zip(xs, ys):
            xt = torch.from_numpy(x).to(dev)
            at = torch.from_numpy(adj).to(dev)
            mt = torch.from_numpy(mask).to(dev)
            yt = torch.tensor([y], dtype=torch.float32, device=dev)
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward(xt, at, mt).unsqueeze(0)
            loss = loss_fn(torch.sigmoid(pred), yt)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        print(f"  epoch {epoch:3d}/{args.epochs} loss={total / len(xs):.6f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fidelity_{spec.name}_best.pt"
    model.save_checkpoint(path, epoch=args.epochs, val_loss=total / len(xs))
    print(f"[train_fidelity] saved -> {path}")


if __name__ == "__main__":
    main()
