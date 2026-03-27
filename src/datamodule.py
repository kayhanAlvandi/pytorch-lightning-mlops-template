"""PyTorch Lightning DataModule for multi-channel images."""
from typing import Dict, List, Optional, Tuple

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .dataset import MultiChannelImageDataset, TiledMultiChannelDataset, DummyLabelsProvider, MongoDBLabelsProvider, LabelEncoder
from .transforms import get_train_transforms, get_val_transforms, get_tile_train_transforms, get_tile_val_transforms


class MultiChannelDataModule(pl.LightningDataModule):
    """DataModule for multi-channel microscopy images.
    
    Handles data loading, splitting, and transformations.
    """
    
    def __init__(
        self,
        root_dir: str,
        channels: List[int],
        crop_size: int = 224,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_val_split: float = 0.66,
        use_mongodb: bool = False,
        use_tiling: bool = True,
        tile_stride: Optional[int] = None,
        cache_size: int = 16,
        max_wells_per_label: Optional[int] = None,
        max_samples_per_label: Optional[int] = None,
        verbose: bool = False,
        exclude_wells: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Args:
            root_dir: Root directory containing images.
            channels: List of channel numbers to use (1-5).
            crop_size: Size for cropping/tiling.
            batch_size: Batch size for data loaders.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            train_val_split: Fraction of data for training (rest for validation).
            use_mongodb: Whether to use MongoDB for labels.
            use_tiling: If True, use grid tiling; if False, use random cropping.
            tile_stride: Stride between tiles. If None, uses crop_size (non-overlapping).
            cache_size: Number of images to keep in LRU cache (for tiling mode).
            max_wells_per_label: Max wells per label class (None = all).
            max_samples_per_label: Max samples per label for balanced dataset (None = all).
            verbose: Print detailed sample selection info.
            exclude_wells: List of (plate, well) tuples to exclude (corrupted images).
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.root_dir = root_dir
        self.channels = channels
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_split = train_val_split
        self.use_mongodb = use_mongodb
        self.use_tiling = use_tiling
        self.tile_stride = tile_stride
        self.cache_size = cache_size
        self.max_wells_per_label = max_wells_per_label
        self.max_samples_per_label = max_samples_per_label
        self.verbose = verbose
        self.exclude_wells = exclude_wells
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.label_encoder = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""
        # Get labels (string labels)
        if self.use_mongodb:
            labels_provider = MongoDBLabelsProvider()
        else:
            labels_provider = DummyLabelsProvider()
        
        labels_dict = labels_provider.get_labels(self.root_dir, exclude_wells=self.exclude_wells)
        
        if len(labels_dict) == 0:
            raise ValueError(f"No labels found for images in {self.root_dir}")
        
        # Limit wells per label if set
        if self.max_wells_per_label is not None:
            labels_dict = self._limit_wells_per_label(labels_dict, self.max_wells_per_label)
        
        # Create and fit label encoder on all unique string labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(labels_dict.values()))
        
        print(f"Classes found: {self.label_encoder.classes}")
        print(f"Number of classes: {self.label_encoder.num_classes}")
        
        if self.use_tiling:
            self._setup_tiled(labels_dict)
        else:
            self._setup_random_crop(labels_dict)
    
    def _setup_tiled(self, labels_dict: Dict):
        """Setup using TiledMultiChannelDataset with grid tiling."""
        # Split at (plate, well) level to avoid data leakage
        # Stratified by label to ensure balanced representation
        unique_wells = list(labels_dict.keys())  # List of (plate, well) tuples
        labels = [labels_dict[w] for w in unique_wells]
        
        if len(unique_wells) < 2:
            raise ValueError(f"Not enough wells to split: {len(unique_wells)} wells")
        
        # Stratified split by label
        train_wells, val_wells = train_test_split(
            unique_wells,
            train_size=self.train_val_split,
            stratify=labels,
            random_state=42,
        )
        
        # Create separate labels dicts for train/val
        train_labels = {w: labels_dict[w] for w in train_wells}
        val_labels = {w: labels_dict[w] for w in val_wells}
        
        # Create train dataset with augmentation transforms
        self.train_dataset = TiledMultiChannelDataset(
            root_dir=self.root_dir,
            channels=self.channels,
            labels_dict=train_labels,
            label_encoder=self.label_encoder,
            crop_size=self.crop_size,
            stride=self.tile_stride,
            transform=get_tile_train_transforms(),
            cache_size=self.cache_size,
            max_samples_per_label=self.max_samples_per_label,
            verbose=self.verbose,
        )
        
        # Create val dataset with minimal transforms
        self.val_dataset = TiledMultiChannelDataset(
            root_dir=self.root_dir,
            channels=self.channels,
            labels_dict=val_labels,
            label_encoder=self.label_encoder,
            crop_size=self.crop_size,
            stride=self.tile_stride,
            transform=get_tile_val_transforms(),
            cache_size=self.cache_size,
            max_samples_per_label=self.max_samples_per_label,
            verbose=self.verbose,
        )
        
        print(f"Train: {self.train_dataset.num_samples} samples -> {len(self.train_dataset)} tiles")
        print(f"Val: {self.val_dataset.num_samples} samples -> {len(self.val_dataset)} tiles")
    
    def _setup_random_crop(self, labels_dict: Dict):
        """Setup using MultiChannelImageDataset with random cropping."""
        # Split at (plate, well) level to avoid data leakage
        # Stratified by label to ensure balanced representation
        unique_wells = list(labels_dict.keys())
        labels = [labels_dict[w] for w in unique_wells]
        
        if len(unique_wells) < 2:
            raise ValueError(f"Not enough wells to split: {len(unique_wells)} wells")
        
        # Stratified split by label
        train_wells, val_wells = train_test_split(
            unique_wells,
            train_size=self.train_val_split,
            stratify=labels,
            random_state=42,
        )
        
        train_labels = {w: labels_dict[w] for w in train_wells}
        val_labels = {w: labels_dict[w] for w in val_wells}
        
        self.train_dataset = MultiChannelImageDataset(
            root_dir=self.root_dir,
            channels=self.channels,
            labels_dict=train_labels,
            label_encoder=self.label_encoder,
            transform=get_train_transforms(
                crop_size=self.crop_size,
                num_channels=len(self.channels),
            ),
        )
        
        self.val_dataset = MultiChannelImageDataset(
            root_dir=self.root_dir,
            channels=self.channels,
            labels_dict=val_labels,
            label_encoder=self.label_encoder,
            transform=get_val_transforms(
                crop_size=self.crop_size,
                num_channels=len(self.channels),
            ),
        )
        
        print(f"Train: {len(train_labels)} wells -> {len(self.train_dataset)} samples")
        print(f"Val: {len(val_labels)} wells -> {len(self.val_dataset)} samples")
    
    def _limit_wells_per_label(self, labels_dict: Dict, n_per_label: int) -> Dict:
        """Limit to first N wells per label class for consistent subset.
        
        Args:
            labels_dict: Dict mapping (plate, well) -> label
            n_per_label: Number of wells to keep per label class
            
        Returns:
            Filtered labels_dict with first N wells per label
        """
        from collections import defaultdict
        
        # Group wells by label
        wells_by_label = defaultdict(list)
        for well_key, label in labels_dict.items():
            wells_by_label[label].append(well_key)
        
        # Sort wells within each label for consistency, take first N
        filtered_dict = {}
        for label, wells in wells_by_label.items():
            wells_sorted = sorted(wells)  # Sort by (plate, well) tuple
            selected = wells_sorted[:n_per_label]
            for well_key in selected:
                filtered_dict[well_key] = label
        
        total_before = len(labels_dict)
        total_after = len(filtered_dict)
        print(f"Limited to {n_per_label} wells per label: {total_before} -> {total_after} wells")
        
        # Print selected wells per label if verbose
        if self.verbose:
            for label in sorted(wells_by_label.keys()):
                wells_sorted = sorted(wells_by_label[label])
                selected = wells_sorted[:n_per_label]
                well_names = [f"{plate}/{well}" for plate, well in selected]
                print(f"  {label}: {well_names}")
        
        return filtered_dict
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            return self.val_dataloader()
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
