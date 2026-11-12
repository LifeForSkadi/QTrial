# -*- coding: utf-8 -*-
"""裁决 qubit-18 之谜：
1) 同一种子路由两次 → 对比 ops 是否逐门一致（确定性检查）
2) 对路由产物扫越界门
3) decompose re=3 lock=True 后：out.n、越界门扫描、fold_only 模拟
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route
from qtrail.pure.post_v2 import fold_only, _decompose_ablation


def scan(ops, tag, nq=16):
    bad = []
    for idx, inst in enumerate(ops):
        qs = [int(q) for q in inst.qubits]
        if any(q >= nq for q in qs):
            bad.append((idx, inst.name, qs))
    if bad:
        print(f'[{tag}] 越界门 {len(bad)} 个，首个: idx={bad[0][0]} '
              f'{bad[0][1]}{bad[0][2]}')
        idx = bad[0][0]
        ctx = [(x.name, tuple(int(q) for q in x.qubits))
               for x in ops[max(0, idx - 2):idx + 3]]
        print(f'   ctx={ctx}')
    else:
        print(f'[{tag}] 干净（qubits 全 < {nq}）')


def main():
    spec = build_tianyan287_spec(load_device_config())
    circ = parse_qasm(open(
        'data/benchpress/qasm/square-heisenberg/'
        'square_heisenberg_N16.qasm', encoding='utf-8').read())
    layout = {i: i for i in range(circ.n)}

    def route():
        return sabre_route(circ, spec, layout, seed=42, trials=1,
                           lookahead_w=0.0, w_ext=0.0,
                           w_layer=None, cand_scope='front')

    r1, s1, f1 = route()
    print(f'route#1: {len(r1.ops)} 门 swap={s1} n={r1.n}')
    scan(r1.ops, 'route#1')
    r2, s2, f2 = route()
    print(f'route#2: {len(r2.ops)} 门 swap={s2} n={r2.n}')
    scan(r2.ops, 'route#2')
    same = (len(r1.ops) == len(r2.ops)
            and all(a.name == b.name and a.qubits == b.qubits
                    for a, b in zip(r1.ops, r2.ops)))
    print(f'两次路由逐门一致: {same}')

    t0 = time.time()
    out = _decompose_ablation(r2, spec=spec, resynth_every=3,
                                   lock_emissions=True)
    print(f'decompose: {len(out.ops)} 门 n={out.n} '
          f'({time.time()-t0:.0f}s)')
    scan(out.ops, 'decompose')

    # fold_only 模拟：n=16 时含 18 必然 IndexError
    try:
        f3, arr3 = fold_only(out)
        print(f'fold_only(out) 成功: {len(f3.ops)} 门 arr_n={len(arr3)}')
    except IndexError as e:
        print(f'fold_only(out) IndexError: {e}')
    print(f'路由产物 fold_only 结果（对照）:')
    try:
        f4, arr4 = fold_only(r2)
        print(f'fold_only(routed) 成功: {len(f4.ops)} 门')
    except IndexError as e:
        print(f'fold_only(routed) IndexError: {e}')


if __name__ == '__main__':
    main()
