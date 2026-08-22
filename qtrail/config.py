"""Configuration system: dataclasses + YAML load/merge.

The device geometry lives in configs/device_tianyan287.yaml (single source of
truth); model/training/decode/postprocess settings live in configs/train_*.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
TABLES_DIR = PROJECT_ROOT / "tables"


@dataclass
class NoiseConfig:
    """Noise-aware mapping parameters (QTrail)."""
    alpha: float = 0.5          # weight of 2Q error factor in edge cost
    beta: float = 0.1           # weight of T1 penalty in edge cost
    lambda_n: float = 0.5       # D_eff = D_topo + lambda_n * (D_noise - D_topo)
    err2q_clip: tuple = (0.2, 5.0)   # clip range of relative 2Q error factor
    t1_penalty_clip: float = 2.0


@dataclass
class DeviceConfig:
    name: str = "tianyan-287"
    rows: int = 15
    cols: int = 7
    absent_couplers: list = field(default_factory=list)  # [[q0, q1], ...] internal idx
    disabled_qubits: list = field(default_factory=list)  # internal idx
    calibration: str = "synthetic"   # synthetic | live | none
    calibration_seed: int = 0
    correlated_defects: bool = True  # spatial defect centers in synthetic calib
    noise: NoiseConfig = field(default_factory=NoiseConfig)


@dataclass
class ModelConfig:
    encoder: str = "gat"             # gat | gt
    d: int = 128
    gat_layers: int = 4
    gat_heads: int = 8
    gat_dropout: float = 0.1
    norm: str = "graph"              # graph | layer
    gt_layers: int = 4
    gt_heads: int = 8
    gt_ff: int = 512
    gt_dropout: float = 0.1
    gt_dist_bias: bool = False       # relative-distance bias for device graph
    gt_tau: float = 8.0
    decoder_heads: int = 16
    clamp: float = 10.0
    context: str = "concat_project"  # concat_project | project_concat | stack_project
    rich_context: bool = False       # QTrail ext: partial-allocation distance context
    program_features: str = "six"    # six | onehot
    device_features: str = "calib"   # calib (7-dim) | onehot


@dataclass
class RewardConfig:
    mode: str = "combined"           # topology | noise | combined
    dist_mult: float = 2.0           # CO-MAP SWAP proxy factor
    normalize: bool = True           # reward = -cost / sum(w)
    depth_lambda: float = 0.0        # 深度感知项：惩罚交互热点比特
                                     # cost += depth_lambda * sum_q h(q)^2 / sum_q h(q)
                                     # （保持终局一次性稀疏结构）
    compactness_lambda: float = 0.0  # 紧凑性项：惩罚布局物理直径（路径 D）
    oracle_mode: str = "static"      # static | routing：路由 oracle 终局奖励
                                     # （研究级训练目标升级：候选布局的真实路由 SWAP）
    oracle_max_n: int = 25           # oracle 适用的最大程序规模（控制训练开销）
    oracle_weight_cap: int = 5       # 合成线路的边权上限（多重边展开封顶）
    oracle_noise_lambda: float = 0.0 # 联合 oracle 的噪声项权重（>0 保持噪声感知）


@dataclass
class GraphConfig:
    """程序图构建选项（数据侧，不影响模型结构/检查点兼容性）。"""
    temporal_alpha: float = 0.0      # 时序感知加权：边权 += alpha × 交互层跨度数
                                     # （强调跨时间层的持续耦合，QFT 链式交互受益）


@dataclass
class DecodeConfig:
    multistart_k: int = 10
    samples: int = 5
    shuffled_greedy: int = 4


@dataclass
class PostProcessConfig:
    enabled: bool = True
    starts: int = 10                 # number of local-search starts (best layouts)
    patience: int = 50
    big_prob: float = 0.3            # large-perturbation probability (phase 1)
    big_moves: int = 3               # consecutive swaps in a large perturbation
    phase2_ratio: float = 0.7        # budget share of phase 1 (exploration)
    hot_frac: float = 0.25           # top fraction of "hot" logical qubits
    max_moves: int = 5000            # total move budget per start
    embedding_moves: bool = True     # 嵌入感知移动（路径 B：结构匹配借鉴）
    embed_prob: float = 0.3          # 嵌入定向交换的概率
    embedding_tol: float = 2.0       # 嵌入改善可容忍的代价上界


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 200
    batch_size: int = 512
    episodes_per_epoch: int = 4096
    lr: float = 3e-4
    lr_min: float = 3e-5
    grad_clip: float = 1.0
    entropy_beta: float = 0.0        # optional entropy bonus (off by default)
    baseline: str = "self_critic"    # self_critic | val_mean
    early_stop_patience: int = 20
    log_every: int = 1
    save_every: int = 5
    compile: bool = False            # torch.compile
    random_p: float = 0.3            # ER edge probability (paper value)
    size_mix: dict = field(default_factory=lambda: {"small": 0.5, "medium": 0.3, "large": 0.2})
    size_ranges: dict = field(default_factory=lambda: {"small": (5, 40), "medium": (41, 80), "large": (81, 105)})
    dataset_mix: dict = field(default_factory=lambda: {"random": 0.5, "circuit": 0.5})
    curriculum: bool = False
    val_episodes: int = 96           # 64 random + 32 circuit
    val_seed: int = 1234
    pool: str = "data/mqtbench/graph_pool.pkl"  # 线路图池路径（可时序加权版）
    device_pool: list = field(default_factory=list)  # 多拓扑训练设备名单
    # 例：["tianyan-287","grid-8x8","grid-6x6","grid-10x10","sycamore-53",
    #      "heavy-hex-115","grid-22x22"]；空列表 = 单设备（--device-name 指定）


@dataclass
class Config:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---------------------------------------------------------------- helpers
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return _build(cls, d)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        cfg = cls.from_dict(d)
        cfg._path = Path(path)
        return cfg

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)


def _build(dc, d: dict):
    """Recursively build a dataclass from a plain dict (unknown keys ignored).

    Note: fields declared with default_factory have no class attribute, so the
    field list must come from __dataclass_fields__ (not hasattr on the class).
    """
    import dataclasses
    kwargs = {}
    fields = dc.__dataclass_fields__
    for k, v in d.items():
        if k not in fields:
            continue
        f = fields[k]
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # pragma: no cover
            default = f.default_factory()
        else:  # pragma: no cover
            default = None
        if isinstance(v, dict) and dataclasses.is_dataclass(default):
            kwargs[k] = _build(default.__class__, v)
        else:
            kwargs[k] = v
    return dc(**kwargs)


def default_config() -> Config:
    return Config()


def load_device_config(path: str | Path | None = None) -> DeviceConfig:
    p = Path(path) if path else CONFIGS_DIR / "device_tianyan287.yaml"
    d = yaml.safe_load(open(p, encoding="utf-8")) or {}
    return _build(DeviceConfig, d)
