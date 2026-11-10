# -*- coding: utf-8 -*-
"""定位 qubit>=16 越界门的产生源：包装 decompose 循环内全部 pass，
每步扫描产物，首个越界门出现即报出 pass 名 + 上下文并退出。

复现路径 = exp_tuccil_lock 的 re=3 lock=True（语义验证崩溃配置）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route

import qtrail.pure.post_v2 as pv2

NQ = 16
_round = [0]


def scan_list(name, ops):
    for idx, inst in enumerate(ops):
        qs = [int(q) for q in inst.qubits]
        if any(q >= NQ for q in qs):
            ctx = [(x.name, tuple(int(q) for q in x.qubits))
                   for x in ops[max(0, idx - 3):idx + 4]]
            print(f'[Q16] round={_round[0]} 产生于 pass={name} '
                  f'idx={idx} inst={inst.name}{qs} ctx={ctx}',
                  flush=True)
            raise SystemExit(1)


def wrap(name, fn):
    def w(*a, **k):
        r = fn(*a, **k)
        if a and isinstance(a[0], list):
            scan_list(name, a[0])
        elif a and hasattr(a[0], 'ops'):
            scan_list(name, a[0].ops)
        return r
    w.__name__ = name
    setattr(pv2, name, w)


for _n in ('_sort_segment_1q', '_merge_sx_pairs', 'slide_x_through_cz',
           'merge_1q_runs', 'merge_adjacent_rz', 'slide_1q_through',
           'commute_rz_through_cz', 'commute_rz_through_cz_rev',
           '_absorb_h', '_cancel_cz_global', '_cancel_adjacent_h',
           '_strip_identity_1q', '_cluster_cz_segments',
           '_platformize_insts', 'collect_resynthesize_2q_blocks'):
    wrap(_n, getattr(pv2, _n))


def main():
    spec = build_tianyan287_spec(load_device_config())
    circ = parse_qasm(open(
        'data/benchpress/qasm/square-heisenberg/'
        'square_heisenberg_N16.qasm', encoding='utf-8').read())
    layout = {i: i for i in range(circ.n)}
    routed, swaps, fin = sabre_route(circ, spec, layout, seed=42, trials=1,
                                     lookahead_w=0.0, w_ext=0.0,
                                     w_layer=None, cand_scope='front')
    scan_list('router', routed.ops)
    print(f'routed 干净: {len(routed.ops)} 门 swap={swaps}', flush=True)

    # 包装 decompose 循环：每轮打标记 + 轮末扫描（循环体内 pass 已包装）
    orig_decompose = pv2.decompose_to_platform_v2

    def w_decompose(c, **kw):
        # 直接复用原函数逻辑但打轮标记：改写为逐 pass 包装不够细——
        # 改为包装 _v4_decompose 后手动展开循环太繁琐，这里直接调用
        # 原函数（其内部 pass 已被包装），轮号由 collect 调用次数推断
        print(f'[run] decompose start lock={kw.get("lock_emissions")}', flush=True)
        r = orig_decompose(c, **kw)
        scan_list('decompose_final', r.ops)
        return r

    pv2.decompose_to_platform_v2 = w_decompose

    t0 = time.time()
    out = pv2.decompose_to_platform_v2(routed, spec=spec,
                                       resynth_every=3,
                                       lock_emissions=True)
    print(f'完成: 1Q={sum(1 for i in out.ops if i.nq == 1)} '
          f'cz={out.count("cz")} total={len(out.ops)} '
          f'({time.time()-t0:.0f}s)', flush=True)
    scan_list('final', out.ops)
    print('无越界门', flush=True)


if __name__ == '__main__':
    main()
