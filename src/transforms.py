"""Image transformations for training and validation.

All transforms operate on (C, H, W) tensors. Each transform class can be
instantiated from YAML via Hydra's _target_ mechanism.

Sample-level transforms are composed into a pipeline via ``build_transforms()``.
Batch-level transforms (Mixup, CutMix) operate on (B, C, H, W) batches and
are applied inside the training loop, not per-sample.
"""
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


# ═══════════════════════════════════════════════════════════════════════════
# Spatial transforms
# ═══════════════════════════════════════════════════════════════════════════

class RandomCrop(nn.Module):
    """Random crop for multi-channel images."""
    
    def __init__(self, size: int):
        super().__init__()
        self.size = size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random crop.
        
        Args:
            x: Input tensor of shape (C, H, W)
            
        Returns:
            Cropped tensor of shape (C, size, size)
        """
        _, h, w = x.shape
        
        if h < self.size or w < self.size:
            # Pad if image is smaller than crop size
            pad_h = max(0, self.size - h)
            pad_w = max(0, self.size - w)
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            _, h, w = x.shape
        
        top = torch.randint(0, h - self.size + 1, (1,)).item()
        left = torch.randint(0, w - self.size + 1, (1,)).item()
        
        return x[:, top:top + self.size, left:left + self.size]


class CenterCrop(nn.Module):
    """Center crop for multi-channel images."""
    
    def __init__(self, size: int):
        super().__init__()
        self.size = size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply center crop.
        
        Args:
            x: Input tensor of shape (C, H, W)
            
        Returns:
            Cropped tensor of shape (C, size, size)
        """
        _, h, w = x.shape
        
        if h < self.size or w < self.size:
            # Pad if image is smaller than crop size
            pad_h = max(0, self.size - h)
            pad_w = max(0, self.size - w)
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            _, h, w = x.shape
        
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        
        return x[:, top:top + self.size, left:left + self.size]


class RandomHorizontalFlip(nn.Module):
    """Random horizontal flip."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return x.flip(-1)
        return x


class RandomVerticalFlip(nn.Module):
    """Random vertical flip."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return x.flip(-2)
        return x


class RandomRotation90(nn.Module):
    """Random 90-degree rotation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(x, k, dims=[-2, -1])


# ═══════════════════════════════════════════════════════════════════════════
# Intensity / noise transforms
# ═══════════════════════════════════════════════════════════════════════════

class Normalize(nn.Module):
    """Normalize tensor to zero mean and unit variance per channel."""
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize per channel
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + self.eps)


class GaussianNoise(nn.Module):
    """Add random Gaussian noise to the image.
    
    Applied with probability p. Noise std is relative to the image's own std
    so it works regardless of input scale.
    """
    
    def __init__(self, std: float = 0.05, p: float = 0.5):
        super().__init__()
        self.std = std
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and torch.rand(1).item() < self.p:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class GaussianBlur(nn.Module):
    """Apply Gaussian blur with a random kernel size.
    
    Kernel size is randomly chosen from the range [min_kernel, max_kernel]
    (must be odd). Works on any number of channels.
    """
    
    def __init__(self, min_kernel: int = 3, max_kernel: int = 7, sigma: float = 1.0, p: float = 0.5):
        super().__init__()
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
        self.sigma = sigma
        self.p = p
    
    def _get_gaussian_kernel_1d(self, kernel_size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        return kernel / kernel.sum()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() >= self.p:
            return x
        
        # Random odd kernel size
        possible = list(range(self.min_kernel, self.max_kernel + 1, 2))
        ks = possible[torch.randint(0, len(possible), (1,)).item()]
        
        k1d = self._get_gaussian_kernel_1d(ks, self.sigma).to(x.device)
        # Outer product -> 2D kernel, expand for depthwise conv
        k2d = k1d[:, None] * k1d[None, :]
        C = x.shape[0]
        kernel = k2d.expand(C, 1, ks, ks)
        
        # (C,H,W) -> (1,C,H,W) for grouped conv
        pad = ks // 2
        out = F.conv2d(x.unsqueeze(0), kernel, padding=pad, groups=C)
        return out.squeeze(0)


class RandomBrightnessContrast(nn.Module):
    """Randomly adjust brightness and contrast per channel.
    
    brightness_range: multiplicative factor range, e.g. (0.8, 1.2)
    contrast_range:   multiplicative factor range applied after centering
    """
    
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
    ):
        super().__init__()
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() >= self.p:
            return x
        
        C = x.shape[0]
        for c in range(C):
            # Brightness: multiply
            b = torch.empty(1).uniform_(*self.brightness_range).item()
            x[c] = x[c] * b
            # Contrast: scale around per-channel mean
            mean_c = x[c].mean()
            k = torch.empty(1).uniform_(*self.contrast_range).item()
            x[c] = (x[c] - mean_c) * k + mean_c
        return x


class RandomErasing(nn.Module):
    """Randomly erase a rectangular region, replacing with zeros or noise.
    
    Scale: fraction of image area to erase.
    Ratio: aspect ratio range of the erased region.
    """
    
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.2),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: str = "zero",
    ):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value  # "zero" or "noise"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() >= self.p:
            return x
        
        C, H, W = x.shape
        area = H * W
        
        for _ in range(10):  # max attempts
            target_area = torch.empty(1).uniform_(*self.scale).item() * area
            aspect = torch.empty(1).uniform_(*self.ratio).item()
            
            h = int(round((target_area * aspect) ** 0.5))
            w = int(round((target_area / aspect) ** 0.5))
            
            if h < H and w < W:
                top = torch.randint(0, H - h, (1,)).item()
                left = torch.randint(0, W - w, (1,)).item()
                
                if self.value == "noise":
                    x[:, top:top + h, left:left + w] = torch.randn(C, h, w, device=x.device)
                else:
                    x[:, top:top + h, left:left + w] = 0
                break
        
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Compose & build utilities
# ═══════════════════════════════════════════════════════════════════════════

class Compose(nn.Module):
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


# Registry of all available transform classes (short name -> class)
TRANSFORM_REGISTRY: Dict[str, type] = {
    "RandomCrop": RandomCrop,
    "CenterCrop": CenterCrop,
    "RandomHorizontalFlip": RandomHorizontalFlip,
    "RandomVerticalFlip": RandomVerticalFlip,
    "RandomRotation90": RandomRotation90,
    "Normalize": Normalize,
    "GaussianNoise": GaussianNoise,
    "GaussianBlur": GaussianBlur,
    "RandomBrightnessContrast": RandomBrightnessContrast,
    "RandomErasing": RandomErasing,
}


def build_transforms(transforms_cfg: List[Dict]) -> Compose:
    """Build a Compose pipeline from a list of transform configs.
    
    Each config dict has a "name" key (looked up in TRANSFORM_REGISTRY)
    plus optional keyword arguments for that transform class.
    
    Example YAML:
        train_transform:
          - name: RandomHorizontalFlip
            p: 0.5
          - name: GaussianNoise
            std: 0.05
          - name: Normalize
    
    Args:
        transforms_cfg: List of dicts, each with "name" and optional params.
    
    Returns:
        Composed transform pipeline.
    """
    transforms = []
    for cfg in transforms_cfg:
        # Convert OmegaConf to plain dict if needed
        if hasattr(cfg, "items"):
            cfg = dict(cfg)
        else:
            cfg = dict(cfg)
        
        name = cfg.pop("name")
        cls = TRANSFORM_REGISTRY.get(name)
        if cls is None:
            available = ", ".join(sorted(TRANSFORM_REGISTRY.keys()))
            raise ValueError(f"Unknown transform '{name}'. Available: {available}")
        
        transforms.append(cls(**cfg))
    
    return Compose(transforms)


# ═══════════════════════════════════════════════════════════════════════════
# Batch-level transforms (Mixup / CutMix) — uses torchvision.transforms.v2
# ═══════════════════════════════════════════════════════════════════════════

def build_batch_transform(cfg: Optional[Dict] = None):
    """Build a batch-level transform from config using torchvision v2.
    
    Uses v2.CutMix, v2.MixUp, and v2.RandomChoice from torchvision.
    Returns a callable that accepts (images, labels) and returns
    (mixed_images, soft_labels) where soft_labels is (B, num_classes).
    
    Example YAML (inside datamodule config):
        batch_transform:
          name: BatchMixCut
          mixup_alpha: 0.2
          cutmix_alpha: 0.4
          num_classes: 6
    
    Supported names:
      - BatchMixup:  v2.MixUp only
      - BatchCutMix: v2.CutMix only
      - BatchMixCut: v2.RandomChoice([CutMix, MixUp])
    
    Args:
        cfg: Dict with "name" and params, or None.
    
    Returns:
        torchvision v2 transform or None.
    """
    if cfg is None:
        return None
    
    if hasattr(cfg, "items"):
        cfg = dict(cfg)
    else:
        cfg = dict(cfg)
    
    name = cfg.pop("name")
    num_classes = cfg.get("num_classes", 2)
    
    if name == "BatchMixup":
        alpha = cfg.get("mixup_alpha", 0.2)
        return v2.MixUp(alpha=alpha, num_classes=num_classes)
    
    elif name == "BatchCutMix":
        alpha = cfg.get("cutmix_alpha", 1.0)
        return v2.CutMix(alpha=alpha, num_classes=num_classes)
    
    elif name == "BatchMixCut":
        mixup_alpha = cfg.get("mixup_alpha", 0.2)
        cutmix_alpha = cfg.get("cutmix_alpha", 1.0)
        cutmix = v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes)
        mixup = v2.MixUp(alpha=mixup_alpha, num_classes=num_classes)
        return v2.RandomChoice([cutmix, mixup])
    
    else:
        raise ValueError(
            f"Unknown batch transform '{name}'. "
            f"Available: BatchMixup, BatchCutMix, BatchMixCut"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Legacy factory functions (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════

def get_train_transforms(crop_size: int, num_channels: int) -> Callable:
    """Get training transforms.
    
    Args:
        crop_size: Size for random cropping.
        num_channels: Number of input channels.
        
    Returns:
        Composed transform function.
    """
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(),
        Normalize(),
    ])


def get_val_transforms(crop_size: int, num_channels: int) -> Callable:
    """Get validation transforms.
    
    Args:
        crop_size: Size for center cropping.
        num_channels: Number of input channels.
        
    Returns:
        Composed transform function.
    """
    return Compose([
        CenterCrop(crop_size),
        Normalize(),
    ])


def get_tile_train_transforms() -> Callable:
    """Get training transforms for pre-tiled images (no cropping needed).
    
    Returns:
        Composed transform function with augmentation and normalization.
    """
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(),
        Normalize(),
    ])


def get_tile_val_transforms() -> Callable:
    """Get validation transforms for pre-tiled images (no cropping needed).
    
    Returns:
        Composed transform function with normalization only.
    """
    return Compose([
        Normalize(),
    ])
