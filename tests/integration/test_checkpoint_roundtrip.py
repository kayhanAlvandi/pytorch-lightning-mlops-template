"""Integration test: train -> save checkpoint -> load -> identical predictions.

This is the assertion-based replacement for the old diagnostic
``test_lightning_ckpt.py`` script. It guards against the PyTorch 2.6
``weights_only`` regression: hparams must round-trip cleanly and the reloaded
model must reproduce the original model's outputs exactly.
"""
from pathlib import Path

import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model import CNNClassifier

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _load_checkpoint(path):
    """Load a checkpoint, tolerant of Lightning versions that do/don't
    forward ``weights_only`` to ``torch.load``."""
    try:
        return CNNClassifier.load_from_checkpoint(path, weights_only=False)
    except TypeError:
        # Older Lightning: weights_only isn't a load_from_checkpoint kwarg
        return CNNClassifier.load_from_checkpoint(path)


def test_checkpoint_roundtrip_preserves_outputs(tiled_datamodule, tmp_path):
    dm = tiled_datamodule
    dm.setup()

    model = CNNClassifier(
        in_channels=len(dm.channels),
        num_classes=dm.label_encoder.num_classes,
        num_blocks=2,
        base_channels=8,
        hidden_dim=16,
    )

    ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path), filename="model", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[ckpt_cb],
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_train_batches=2,
        limit_val_batches=2,
    )
    trainer.fit(model, dm)

    ckpt_path = ckpt_cb.best_model_path
    assert ckpt_path and Path(ckpt_path).exists()

    loaded = _load_checkpoint(ckpt_path)

    model.eval()
    loaded.eval()
    x = torch.randn(2, len(dm.channels), 32, 32)
    with torch.no_grad():
        expected = model(x)
        actual = loaded(x)
    assert torch.allclose(expected, actual, atol=1e-5)


def test_checkpoint_hparams_are_plain_types(tiled_datamodule, tmp_path):
    """Saved hyper_parameters must be plain Python types (no OmegaConf types)
    so the checkpoint loads under torch>=2.6 weights_only defaults."""
    from omegaconf import DictConfig, ListConfig

    dm = tiled_datamodule
    dm.setup()
    model = CNNClassifier(
        in_channels=len(dm.channels),
        num_classes=dm.label_encoder.num_classes,
        num_blocks=2,
        base_channels=8,
        hidden_dim=16,
    )
    ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path), filename="m", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", devices=1, logger=False,
        callbacks=[ckpt_cb], enable_progress_bar=False, enable_model_summary=False,
        limit_train_batches=2, limit_val_batches=2,
    )
    trainer.fit(model, dm)

    ckpt = torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})

    def _assert_plain(obj, path="hparams"):
        assert not isinstance(obj, (DictConfig, ListConfig)), f"OmegaConf type at {path}"
        if isinstance(obj, dict):
            for k, v in obj.items():
                _assert_plain(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _assert_plain(v, f"{path}[{i}]")

    _assert_plain(hparams)
