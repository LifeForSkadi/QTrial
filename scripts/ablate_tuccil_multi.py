# -*- coding: utf-8 -*-
"""Tuccil 端到端消融的普适性检查：h25 / qv34 × (a) 全组件 vs (b) 无 Tuccil。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route
from qtrail.pure.post_v2 import decompose_to_platform_v2

CASES = [
    ('h25', 'data/benchpress/qasm/square-heisenberg/'
     'square_heisenberg_N25.qasm'),
    ('qv34', 'data/benchpress/qasm/qv/qv_N034_12345.qasm'),
]


def count_1q(ops):
    return sum(1 for i in ops if i.nq == 1 and i.name in ("rz", "sx", "x"))


def main():
    import time
    spec = build_tianyan287_spec(load_device_config())
    for tag, path in CASES:
        circ = parse_qasm(open(path, encoding='utf-8').read())
        layout = {i: i for i in range(circ.n)}
        routed, swaps, fin = sabre_route(circ, spec, layout, seed=42,
                                         trials=1, lookahead_w=0.0,
                                         w_ext=0.0, w_layer=None,
                                         cand_scope='front')
        t0 = time.time()
        out_a = decompose_to_platform_v2(routed, spec=spec, resynth_every=1)
        ta = time.time() - t0
        print(f'{tag} (a) 全组件: 1Q={count_1q(out_a.ops)} '
              f'cz={out_a.count("cz")} depth={out_a.layer_depth()} '
              f'({ta:.0f}s)', flush=True)

        import qtrail.pure.synthesis as syn
        orig = syn.fit_block_tuccil
        syn.fit_block_tuccil = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("disabled"))
        t0 = time.time()
        out_b = decompose_to_platform_v2(routed, spec=spec, resynth_every=1)
        tb = time.time() - t0
        syn.fit_block_tuccil = orig
        print(f'{tag} (b) 无Tuccil: 1Q={count_1q(out_b.ops)} '
              f'cz={out_b.count("cz")} depth={out_b.layer_depth()} '
              f'({tb:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
