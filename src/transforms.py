"""Image transformations for training and validation."""
from typing import Callable

import torch
import torch.nn as nn


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


class Compose(nn.Module):
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


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
