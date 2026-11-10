# -*- coding: utf-8 -*-
"""A2 重合成频率扫描：heisenberg N16 身份布局路由，resynth_every ∈
{1,2,3,4,6,12} 的 1Q/cz/深度/墙钟对比（lock 关——生产口径）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route
from qtrail.pure.post_v2 import decompose_to_platform_v2


def count_1q(ops):
    return sum(1 for i in ops if i.nq == 1 and i.name in ("rz", "sx", "x"))


def main():
    spec = build_tianyan287_spec(load_device_config())
    circ = parse_qasm(open(
        'data/benchpress/qasm/square-heisenberg/'
        'square_heisenberg_N16.qasm', encoding='utf-8').read())
    layout = {i: i for i in range(circ.n)}
    routed, swaps, fin = sabre_route(circ, spec, layout, seed=42, trials=1,
                                     lookahead_w=0.0, w_ext=0.0,
                                     w_layer=None, cand_scope='front')
    print(f'路由: {len(routed.ops)} 门 swap={swaps}', flush=True)
    for re in (1, 2, 3, 4, 6, 12):
        t0 = time.time()
        out = decompose_to_platform_v2(routed, spec=spec,
                                       resynth_every=re)
        print(f're={re}: 1Q={count_1q(out.ops)} cz={out.count("cz")} '
              f'total={len(out.ops)} depth={out.layer_depth()} '
              f'({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
