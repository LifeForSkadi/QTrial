# -*- coding: utf-8 -*-
"""graphstate_50 全管线分阶段计时（2026-11-12）：100 门 50 比特却
900s 超时——定位超多项式阶段。逐阶段时间戳（无 profiler 开销）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faulthandler
faulthandler.dump_traceback_later(120, repeat=True)  # 每 120s 栈转储

import numpy as np
import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"

T0 = time.time()


def mark(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    circ = parse_qasm(Path("data/mqtbench/stratified/graphstate_50.qasm")
                      .read_text(encoding="utf-8", errors="replace"))
    print(f"graphstate_50: n={circ.n} 门={len(circ.ops)}", flush=True)
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)

    # 阶段 1：RL 布局（compute_layout 内部）
    from qtrail.problems import build_program_graph
    from qtrail.pure.mapper import _ops_from_circuit
    mark("开始 compute_layout")
    graph = build_program_graph(circ.n, _ops_from_circuit(circ),
                                circuit_id="gs50",
                                temporal_alpha=cfg.graph.temporal_alpha)
    pi, _ = mapper.compute_layout(graph, circ, 0.0)
    mark(f"compute_layout 完成")
    layout = {i: int(pi[i]) for i in range(graph.n)}

    # 阶段 2：最终路由（16 种子）
    from qtrail.pure.router import sabre_route
    mark("开始 sabre_route")
    routed, swaps, fl = sabre_route(circ, spec, layout, seed=42,
                                    trials=16, mix_rollout=True,
                                    max_swaps=100_000)
    mark(f"sabre_route 完成 swaps={swaps} 门={len(routed.ops)}")

    # 阶段 3：post_route
    from qtrail.pure.post import post_route
    mark("开始 post_route")
    routed, removed = post_route(routed, fl, spec.adj)
    mark(f"post_route 完成 removed={removed} 门={len(routed.ops)}")

    # 阶段 4：_v5_pipeline
    mark("开始 _v5_pipeline")
    routed, fl2, qc = mapper._v5_pipeline(routed, fl, circ, 0.0,
                                          with_learned=True,
                                          with_resynth=True,
                                          relabel_budget=25.0)
    mark(f"_v5_pipeline 完成 门={len(qc.ops)}")

    # 阶段 5：metrics
    from qtrail.pure.metrics import compute_metrics
    m = compute_metrics(qc, routed.count("swap"), spec.calib)
    mark(f"完成 fid={m['est_fidelity']:.3e} depth={m['depth']}")


if __name__ == "__main__":
    main()
