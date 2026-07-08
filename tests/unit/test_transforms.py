"""Unit tests for sample-level transforms and the build_transforms registry."""
import pytest
import torch

from src.transforms import (
    build_transforms,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomCrop,
    CenterCrop,
    Normalize,
)


def test_build_transforms_returns_callable_and_preserves_channels(sample_chw):
    pipeline = build_transforms([
        {"name": "RandomHorizontalFlip", "p": 0.0},
        {"name": "Normalize"},
    ])
    out = pipeline(sample_chw)
    assert out.shape[0] == sample_chw.shape[0]
    assert out.shape[-2:] == sample_chw.shape[-2:]


def test_random_horizontal_flip_deterministic(sample_chw):
    flip = RandomHorizontalFlip(p=1.0)
    assert torch.equal(flip(sample_chw), sample_chw.flip(-1))


def test_random_horizontal_flip_identity_when_p_zero(sample_chw):
    flip = RandomHorizontalFlip(p=0.0)
    assert torch.equal(flip(sample_chw), sample_chw)


def test_random_vertical_flip_deterministic(sample_chw):
    flip = RandomVerticalFlip(p=1.0)
    assert torch.equal(flip(sample_chw), sample_chw.flip(-2))


def test_random_crop_changes_spatial_size(sample_chw):
    out = RandomCrop(size=32)(sample_chw)
    assert out.shape == (sample_chw.shape[0], 32, 32)


def test_center_crop_shape(sample_chw):
    out = CenterCrop(size=16)(sample_chw)
    assert out.shape == (sample_chw.shape[0], 16, 16)


def test_crop_pads_when_larger_than_input(sample_chw):
    # crop bigger than the 64x64 input must pad, not crash
    out = CenterCrop(size=80)(sample_chw)
    assert out.shape == (sample_chw.shape[0], 80, 80)


def test_normalize_zero_mean_unit_std(sample_chw):
    out = Normalize()(sample_chw)
    per_channel_mean = out.mean(dim=(-2, -1))
    per_channel_std = out.std(dim=(-2, -1))
    assert torch.allclose(per_channel_mean, torch.zeros_like(per_channel_mean), atol=1e-5)
    assert torch.allclose(per_channel_std, torch.ones_like(per_channel_std), atol=1e-2)


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        build_transforms([{"name": "NotARealTransform"}])
