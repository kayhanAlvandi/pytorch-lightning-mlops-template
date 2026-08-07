"""Integration smoke test: a full (tiny) training loop must complete.

Marked slow/integration so it can be excluded from the fast CI path with
``pytest -m "not slow"``.
"""
import pytest
import torch
import pytorch_lightning as pl

from src.model import CNNClassifier

pytestmark = [pytest.mark.slow]

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device not available"
)


def _tiny_model(dm) -> CNNClassifier:
    return CNNClassifier(
        in_channels=len(dm.channels),
        num_classes=dm.label_encoder.num_classes,
        num_blocks=2,
        base_channels=8,
        hidden_dim=16,
    )


def test_training_completes_one_epoch(tiled_datamodule):
    dm = tiled_datamodule
    dm.setup()
    model = _tiny_model(dm)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_train_batches=2,
        limit_val_batches=2,
    )
    trainer.fit(model, dm)

    assert trainer.state.finished
    # a train loss metric should have been recorded
    assert any("train" in k for k in trainer.callback_metrics)


def test_validation_and_test_run(tiled_datamodule):
    dm = tiled_datamodule
    dm.setup()
    model = _tiny_model(dm)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
    )
    trainer.fit(model, dm)
    results = trainer.test(model, dm, verbose=False)
    assert isinstance(results, list) and len(results) >= 1


class _DeviceProbe(pl.Callback):
    """Records whether the model was on a CUDA device during training.

    Checked at batch start because Lightning moves the model back to CPU
    during teardown, so inspecting the model after fit() is unreliable.
    """

    def __init__(self):
        self.trained_on_cuda = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.trained_on_cuda is None:
            self.trained_on_cuda = next(pl_module.parameters()).is_cuda


@pytest.mark.gpu
@requires_cuda
def test_training_on_gpu(tiled_datamodule):
    dm = tiled_datamodule
    dm.setup()
    model = _tiny_model(dm)

    probe = _DeviceProbe()
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[probe],
    )
    trainer.fit(model, dm)

    assert trainer.state.finished
    # the trainer selected a CUDA device
    assert trainer.strategy.root_device.type == "cuda"
    # the model was actually on CUDA while training ran
    assert probe.trained_on_cuda is True
