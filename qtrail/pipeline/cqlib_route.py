"""Cqlib 注入接口：QTrial RL 布局 → 天衍平台原生映射管线（transpile_qcis）。

统一口径：Cqlib 输出经 QCIS→QASM 回转后，用与 QTrial/Qiskit/tket 相同的
compute_metrics 度量（SWAP 数从 QCIS 文本计数，深度用 qiskit 并行深度）。
平台对象：有 login_key 时用真实 TianYanPlatform（download_config 真机配置），
无 key 时用合成配置的 MockPlatform——接口一致，评审现场可无缝切换。
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, qasm2

from qtrail.devices.spec import DeviceSpec


def _build_platform(spec: DeviceSpec, login_key: str | None = None,
                    machine: str = "tianyan-287"):
    """构造 Cqlib 平台对象（真实 token 优先，合成配置回退）。"""
    import os
    key = login_key or os.environ.get("TIANYAN_LOGIN_KEY", "") \
        or os.environ.get("CQLIB_LOGIN_KEY", "")
    if key:
        try:
            from cqlib import TianYanPlatform
            return TianYanPlatform(login_key=key, machine_name=machine)
        except Exception:
            pass

    class MockPlatform:
        def __init__(self, spec):
            self.machine_name = machine
            self._config = None

        def download_config(self, read_time=None, machine=None):
            if self._config is None:
                cm = {}
                idx = 0
                for i in range(spec.n):
                    for j in range(i + 1, spec.n):
                        if spec.adj[i, j]:
                            cm[f"C{idx}"] = [f"Q{i}", f"Q{j}"]
                            idx += 1
                self._config = {
                    "overview": {"coupler_map": cm},
                    "disabledQubits": "",
                    "disabledCouplers": "",
                }
            return self._config

    return MockPlatform(spec)


def cqlib_route(qc: QuantumCircuit, spec: DeviceSpec, layout: dict | None = None,
                objective: str = "depth", seed: int = 0,
                login_key: str | None = None, timeout_guard: float = 300.0):
    """QTrial 布局（或 None=平台原生映射）经 Cqlib 管线执行。

    Returns: (qiskit_circuit, swap_count, final_layout) —— 统一口径。
    """
    import signal
    from cqlib.mapping import mapping as cmap
    from cqlib.utils.qasm_to_qcis import QasmToQcis
    from cqlib.utils.qcis_to_qasm import QcisToQasm

    platform = _build_platform(spec, login_key)

    # 恒等门补全到全部平台比特（仅注入模式需要）：cqlib 的 transpile_qcis 要求
    # (a) initial_layout 的键都能在 qubit_mapping 中找到（只认 QCIS 中
    #     出现过的比特）——未使用比特必须用 id 门引入 QCIS；
    # (b) mapping_virtual_to_final 迭代 range(len(ag))（ag = 全部平台
    #     比特的架构图）——initial_layout 必须覆盖每一个虚拟索引，
    #     否则 KeyError。
    # 原生模式（layout=None）由 cqlib 自选初始映射，无需补全。
    # 测量必须剥离后最后回挂：cqlib 的 generate_qubit_mapping 在遇到
    # measure 行即停止收集比特，若 id 门排在 measure 之后则被忽略。
    measures = [inst for inst in qc.data if inst.operation.name == "measure"]
    qc_padded = qc.copy()
    qc_padded.remove_final_measurements(inplace=True)
    used = {qc.find_bit(q).index for inst in qc.data for q in inst.qubits}
    for qi in range(qc.num_qubits):
        if qi not in used:
            qc_padded.id(qi)
    if layout is not None and qc_padded.num_qubits < spec.n:
        n_before = qc_padded.num_qubits
        qc_padded.add_register(QuantumRegister(spec.n - n_before, "anc"))
        for qi in range(n_before, spec.n):
            qc_padded.id(qi)
    # remove_final_measurements 会连带删除经典寄存器，先补回
    for creg in qc.cregs:
        if creg.name not in {c.name for c in qc_padded.cregs}:
            qc_padded.add_register(creg)
    for inst in measures:
        qc_padded.append(inst)

    qcis = QasmToQcis().convert_to_qcis(qasm2.dumps(qc_padded))

    # 布局补全为全平台满射：真实布局覆盖已用比特，其余虚拟比特
    # 分配剩余空闲物理比特（无门的虚拟比特不影响 MCTS 搜索语义）
    init = None
    if layout is not None:
        init = {int(k): int(v) for k, v in layout.items()}
        occupied = set(init.values())
        free = [p for p in range(spec.n) if p not in occupied]
        for i in range(spec.n):
            if i not in init:
                init[i] = free.pop(0)

    # 超时保护（Cqlib MCTS 在大线路上慢，且内部可能死循环——
    # 用子进程硬终止，线程池超时无法杀掉失控线程）
    platform_config = platform.download_config()
    raw_qcis, final_map = _run_with_timeout(
        qcis, platform_config, init, objective, seed, timeout_guard)
    swap_count = raw_qcis.count("SWAP")
    qc_back = _qcis_to_qiskit_with_swaps(raw_qcis, spec.n)
    # 过滤补全用的 id 门（恒等，不影响语义；保留会使深度虚增 1）
    qc_back.data = [inst for inst in qc_back.data
                    if inst.operation.name != "id"]
    return qc_back, swap_count, dict(final_map)


def _qcis_to_qiskit_with_swaps(qcis_text: str, n_qubits: int) -> QuantumCircuit:
    """QCIS → qiskit，支持 SWAP：拆成无 SWAP 段逐段转换，段间插入
    qiskit swap（QcisToQasm 不支持 SWAP 门）；末段 M 行按原顺序回挂。"""
    import re
    from qiskit import ClassicalRegister
    from cqlib.utils.qcis_to_qasm import QcisToQasm

    conv = QcisToQasm()
    ops = []   # ("seg", text) | ("swap", (a, b))
    buf = []
    meas = []
    for line in qcis_text.splitlines():
        s = line.strip()
        if not s or s.startswith("BARRIER"):
            continue
        m = re.match(r"SWAP\s+Q(\d+)\s+Q(\d+)", s)
        if m:
            if buf:
                ops.append(("seg", "\n".join(buf)))
                buf = []
            ops.append(("swap", (int(m.group(1)), int(m.group(2)))))
            continue
        m2 = re.match(r"M\s+Q(\d+)", s)
        if m2:
            if buf:
                ops.append(("seg", "\n".join(buf)))
                buf = []
            meas.append(int(m2.group(1)))
            continue
        buf.append(s)
    if buf:
        ops.append(("seg", "\n".join(buf)))

    from qtrail.utils.qasm_io import sanitize_qasm
    out = QuantumCircuit(n_qubits)
    for kind, payload in ops:
        if kind == "swap":
            out.swap(*payload)
        else:
            # QcisToQasm 会输出 sx 等 qelib1 之外的门，先统一改写
            seg_qc = qasm2.loads(
                sanitize_qasm(conv.convert_qcis_to_qasm(payload)))
            for inst in seg_qc.data:
                qs = [seg_qc.find_bit(q).index for q in inst.qubits]
                out.append(inst.operation, [out.qubits[i] for i in qs], [])
    if meas:
        creg = ClassicalRegister(len(meas), "m")
        out.add_register(creg)
        for i, q in enumerate(meas):
            out.measure(q, creg[i])
    return out


def _mcts_worker(qcis, config, init, objective, seed):
    """子进程工作函数：从平台配置重建最小平台对象并执行 transpile_qcis。"""
    from cqlib.mapping import mapping as cmap

    class _SubPlatform:
        def __init__(self, cfg):
            self.machine_name = "tianyan-287"
            self._config = cfg

        def download_config(self, read_time=None, machine=None):
            return self._config

    res = cmap.transpile_qcis(qcis, _SubPlatform(config),
                              initial_layout=init, objective=objective,
                              seed=seed)
    return res[0].as_str(), dict(res[3])  # QCIS 文本 + virtual→final 映射


def _queue_worker(q, *args):
    try:
        q.put(_mcts_worker(*args))
    except Exception as e:  # 子进程异常回传（"__ERR__" 哨兵）
        q.put(("__ERR__", f"{type(e).__name__}: {e}"))


def _run_with_timeout(qcis, config, init, objective, seed, timeout_s):
    """MCTS 子进程硬超时：超时即 terminate（Windows spawn，参数全部可
    pickle；线程池方案在超时后 shutdown 会等待失控线程，无法真正杀掉）。"""
    import multiprocessing as mp
    import queue

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_queue_worker,
                    args=(q, qcis, config, init, objective, seed))
    p.start()
    try:
        res = q.get(timeout=timeout_s)
    except queue.Empty:
        p.terminate()
        p.join(5)
        raise TimeoutError(f"Cqlib transpile 超时（>{timeout_s}s，已硬终止）")
    p.join(5)
    if isinstance(res, tuple) and res and res[0] == "__ERR__":
        raise RuntimeError(f"Cqlib transpile 子进程失败：{res[1]}")
    return res
