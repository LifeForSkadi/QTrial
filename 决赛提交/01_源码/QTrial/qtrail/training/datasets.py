"""Episode sources: ER random graphs + circuit-derived graphs, 多拓扑支持.

每个 episode 是 (ProgramGraph, DeviceSpec) 对；设备从 device_pool 中随机
选取（程序图规模必须 ≤ 设备规模，采样时按规模筛选合法设备）。
"""
from __future__ import annotations

import numpy as np

from qtrail.config import TrainConfig
from qtrail.devices.spec import DeviceSpec
from qtrail.problems import ProgramGraph, random_program_graph


class EpisodeSource:
    """Generates training/validation episodes over a device pool.

    单设备（device_pool 长度 1）时行为与旧版完全一致。
    """

    def __init__(self, cfg: TrainConfig, device_pool: list[DeviceSpec],
                 graph_pool: list[ProgramGraph] | None = None):
        self.cfg = cfg
        self.device_pool = device_pool
        self.pool = graph_pool or []
        self._rng = np.random.default_rng(cfg.seed)
        self._sorted_devices = sorted(device_pool, key=lambda s: s.n)
        self._device_sizes = np.array([s.n for s in self._sorted_devices])

    # ------------------------------------------------------------ sampling
    def _sample_device(self, max_n: int | None = None) -> DeviceSpec:
        """随机选一台设备；max_n 给定时只选规模 ≥ max_n 的设备。"""
        sizes = self._device_sizes
        lo = int(np.searchsorted(sizes, max_n, side="left")) if max_n else 0
        hi = len(sizes) - 1
        if lo > hi:
            # 没有足够大的设备：取最大的（程序图将被截断适配）
            lo = hi
        idx = int(self._rng.integers(lo, hi + 1))
        return self._sorted_devices[idx]

    def _sample_size(self, spec: DeviceSpec) -> int:
        r = self._rng.random()
        mix = self.cfg.size_mix
        ranges = self.cfg.size_ranges
        if r < mix["small"]:
            lo, hi = ranges["small"]
        elif r < mix["small"] + mix["medium"]:
            lo, hi = ranges["medium"]
        elif r < mix["small"] + mix["medium"] + mix.get("huge", 0.0):
            lo, hi = ranges.get("huge", ranges["large"])
        else:
            lo, hi = ranges["large"]
        lo = min(lo, spec.n)
        hi = min(hi, spec.n)
        return int(self._rng.integers(lo, hi + 1))

    def sample_random(self, k: int, weighted: bool = False):
        """随机图 episodes：(graph, spec) 对。"""
        eps = []
        for _ in range(k):
            spec = self._sample_device()
            n = self._sample_size(spec)
            g = random_program_graph(n, p=self.cfg.random_p, rng=self._rng,
                                     weighted=weighted)
            eps.append((g, spec))
        return eps

    def sample_circuit(self, k: int):
        """线路图 episodes：图按规模匹配到足够大的设备。"""
        if not self.pool:
            raise ValueError("circuit graph pool is empty")
        eps = []
        for _ in range(k):
            g = self.pool[int(self._rng.integers(0, len(self.pool)))]
            spec = self._sample_device(max_n=g.n)
            eps.append((g, spec))
        return eps

    def sample_episodes(self, k: int):
        """随机图与线路图混合（多设备）。"""
        mix = self.cfg.dataset_mix
        n_circuit = int(k * mix.get("circuit", 0.0))
        eps = self.sample_circuit(n_circuit) if n_circuit else []
        eps += self.sample_random(k - n_circuit)
        self._rng.shuffle(eps)
        return eps

    def val_episodes(self):
        """验证集：跨设备池均匀分布。"""
        k = self.cfg.val_episodes
        per_device = max(k // len(self.device_pool), 1)
        eps = []
        for spec in self.device_pool:
            for _ in range(per_device):
                if self.pool and self._rng.random() < 1 / 3:
                    picked = None
                    for _ in range(50):  # 拒绝采样：找 n ≤ spec.n 的池图
                        g = self.pool[int(self._rng.integers(0, len(self.pool)))]
                        if g.n <= spec.n:
                            picked = g
                            break
                    if picked is not None:
                        eps.append((picked, spec))
                        continue
                n = self._sample_size(spec)
                eps.append((random_program_graph(n, p=self.cfg.random_p,
                                                 rng=self._rng), spec))
        return eps[:k]


def bucket_batches(episodes, batch_size: int):
    """按 (设备, 程序规模) 分桶，yield (indices, spec)。

    episodes: list of (graph, spec)；同批实例共享同一台设备。
    """
    uniq = {id(s): s for _, s in episodes}
    dev_list = sorted(uniq.values(), key=lambda d: d.n)
    dev_ids = {id(s): i for i, s in enumerate(dev_list)}
    order = sorted(range(len(episodes)),
                   key=lambda i: (dev_ids[id(episodes[i][1])], episodes[i][0].n))
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        sub = {}
        for i in chunk:
            sub.setdefault(dev_ids[id(episodes[i][1])], []).append(i)
        for dev_id, idxs in sub.items():
            yield idxs, dev_list[dev_id]
