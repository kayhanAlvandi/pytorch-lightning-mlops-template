"""Unit tests for MultiChannelDataModule setup, splitting and dataloaders.

Uses the shared ``tiled_datamodule`` fixture (synthetic on-disk images +
stubbed labels provider) so the split is deterministic and balanced
(2 classes x 2 wells), avoiding stratify errors.
"""
import pytest


pytestmark = [pytest.mark.training]

@pytest.fixture
def datamodule(tiled_datamodule):
    return tiled_datamodule


def test_setup_builds_label_encoder(datamodule):
    datamodule.setup()
    assert datamodule.label_encoder.num_classes == 2
    assert set(datamodule.label_encoder.classes) == {"ClassA", "ClassB"}


def test_train_val_split_has_no_well_overlap(datamodule):
    datamodule.setup()
    train_wells = {(s["plate"], s["well"]) for s in datamodule.train_dataset.samples}
    val_wells = {(s["plate"], s["well"]) for s in datamodule.val_dataset.samples}
    assert train_wells
    assert val_wells
    assert train_wells.isdisjoint(val_wells)


def test_train_dataloader_batch_shapes(datamodule):
    datamodule.setup()
    loader = datamodule.train_dataloader()
    images, labels = next(iter(loader))
    assert images.ndim == 4  # (B, C, H, W)
    assert images.shape[1] == len(datamodule.channels)
    assert images.shape[-2:] == (32, 32)
    assert images.shape[0] == labels.shape[0]


def test_val_dataloader_is_not_shuffled(datamodule):
    datamodule.setup()
    loader = datamodule.val_dataloader()
    # DataLoader stores a SequentialSampler when shuffle=False
    from torch.utils.data import SequentialSampler
    assert isinstance(loader.sampler, SequentialSampler)
