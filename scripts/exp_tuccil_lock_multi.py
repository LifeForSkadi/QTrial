# -*- coding: utf-8 -*-
"""Tuccil lock 多线路 probe：身份布局路由后 re=3 lock on/off 对比。

线路：square_heisenberg_N25、qv_N034（1Q 差距重灾区）、qaoa_barabasi_N29。
每线路 2 配置（lock off/on），输出 1Q/cz/depth/tu_locked。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route
from qtrail.pure.post_v2 import _decompose_ablation

CASES = [
    ('data/benchpress/qasm/square-heisenberg/'
     'square_heisenberg_N25.qasm', 'h25'),
    ('data/benchpress/qasm/qv/qv_N034_12345.qasm', 'qv34'),
]


def count_1q(ops):
    return sum(1 for i in ops if i.nq == 1 and i.name in ("rz", "sx", "x"))


def main():
    spec = build_tianyan287_spec(load_device_config())
    for path, tag in CASES:
        p = Path(path)
        if not p.exists():
            # qv 文件名探测
            cands = list(Path(path).parent.glob('*N034*.qasm'))
            if cands:
                p = cands[0]
            else:
                print(f'{tag}: 找不到 {path}，跳过', flush=True)
                continue
        circ = parse_qasm(open(p, encoding='utf-8').read())
        layout = {i: i for i in range(circ.n)}
        t0 = time.time()
        routed, swaps, fin = sabre_route(circ, spec, layout, seed=42,
                                         trials=1, lookahead_w=0.0,
                                         w_ext=0.0, w_layer=None,
                                         cand_scope='front')
        print(f'{tag}: n={circ.n} 路由 {len(routed.ops)} 门 '
              f'swap={swaps} ({time.time()-t0:.0f}s)', flush=True)
        for lock in (False, True):
            t0 = time.time()
            out = _decompose_ablation(routed, spec=spec,
                                           resynth_every=3,
                                           lock_emissions=lock)
            nlock = sum(1 for i in out.ops if getattr(i, 'tu_locked', False))
            print(f'{tag} lock={int(lock)}: 1Q={count_1q(out.ops)} '
                  f'cz={out.count("cz")} total={len(out.ops)} '
                  f'depth={out.layer_depth()} tu_locked={nlock} '
                  f'({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
