"""Unit tests that the Hydra configs compose and produce usable structures.

These catch broken YAML / bad defaults early without running training.
"""
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.transforms import build_transforms

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")

pytestmark = [pytest.mark.training]

def _compose(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config", overrides=overrides or [])


def test_default_config_composes():
    cfg = _compose()
    assert cfg.model._target_ == "src.model.CNNClassifier"
    for group in ("datamodule", "optimizer", "loss", "trainer", "callbacks"):
        assert group in cfg, f"missing config group: {group}"


def test_datamodule_config_has_dataset_target():
    cfg = _compose()
    assert cfg.datamodule.dataset._target_.startswith("src.dataset.")
    assert len(cfg.datamodule.dataset.channels) > 0


def test_train_transform_config_builds_and_runs():
    cfg = _compose()
    tf = build_transforms(
        OmegaConf.to_container(cfg.datamodule.train_transform, resolve=True)
    )
    n_channels = len(cfg.datamodule.dataset.channels)
    out = tf(torch.rand(n_channels, 40, 40))
    assert out.shape[0] == n_channels
