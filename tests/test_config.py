"""Config loading regression tests (YAML must actually override defaults)."""
from pathlib import Path

import pytest

from qtrail.config import Config, load_device_config

ROOT = Path(__file__).resolve().parent.parent


def test_yaml_overrides_defaults():
    cfg = Config.load(ROOT / "configs" / "train_gat.yaml")
    assert cfg.model.encoder == "gat"
    assert cfg.reward.mode == "topology"
    assert cfg.model.device_features == "onehot"
    assert cfg.train.epochs == 400
    assert cfg.train.early_stop_patience == 50
    assert cfg.train.batch_size == 512
    assert cfg.model.context == "concat_project"


def test_configs_are_distinct():
    g1 = Config.load(ROOT / "configs" / "train_gat.yaml")
    g2 = Config.load(ROOT / "configs" / "train_gat_noise.yaml")
    g3 = Config.load(ROOT / "configs" / "train_gt_noise.yaml")
    assert (g1.reward.mode, g1.model.device_features) == ("topology", "onehot")
    assert (g2.reward.mode, g2.model.device_features) == ("combined", "calib")
    assert g2.model.rich_context is True
    assert g3.model.encoder == "gt"
    assert g3.train.batch_size == 256
    assert g3.model.gt_dist_bias is True


def test_device_config_loads():
    d = load_device_config()
    assert d.name == "tianyan-287"
    assert d.rows == 15 and d.cols == 7
    assert d.calibration == "synthetic"
    assert d.noise.lambda_n == pytest.approx(0.5)
