"""临时分析：qaoa N69 全管线耗时分解（cProfile）+ 路由器裸吞吐。"""
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
P = Path("data/benchpress/qasm/qaoa/qaoa_barabasi_albert_N69_3reps.qasm")

dev_cfg = load_device_config()
cfg = Config()
cfg.device = dev_cfg
spec = build_tianyan287_spec(dev_cfg)
policy, ck = QAPolicy.load_checkpoint(
    CKPT, device_n=spec.n,
    map_location="cuda" if torch.cuda.is_available() else "cpu")
policy.eval()
cfg.model = ck.get("model_cfg", cfg.model)
cfg.graph = policy.graph_cfg

text = P.read_text(encoding="utf-8")
circ = parse_qasm(text)
print(f"circuit: {P.name} n={circ.n} ops={len(circ.ops)} 2q={circ.count_2q()}")

# 1) 路由器裸吞吐：平凡布局单次全路由
import numpy as np
layout = {i: int(np.arange(circ.n)[i] % spec.n) for i in range(circ.n)}
t0 = time.perf_counter()
routed, swaps, _ = sabre_route(circ, spec, layout, seed=42)
t_router = time.perf_counter() - t0
print(f"router alone (trivial layout): {swaps} swaps in {t_router:.2f}s "
      f"({t_router / max(swaps, 1) * 1e6:.1f} us/swap)")

# 2) 全管线 cProfile
pr = cProfile.Profile()
pr.enable()
mapper = PureMapper(spec, policy=policy, cfg=cfg, dev="cuda",
                    seed=42, selection_rule="fidelity", use_post=True)
res = mapper.map_circuit(parse_qasm(text), circuit_id="qaoa_n69")
pr.disable()
print(f"map_circuit: swaps={res['swap_count']} wall={res['metrics']['wall_s']}s")
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(28)
print(s.getvalue())
