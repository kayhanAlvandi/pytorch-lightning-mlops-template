"""PyTorch Lightning DataModule for multi-channel images."""
from typing import Dict, List, Optional

import pytorch_lightning as pl
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .dataset import DummyLabelsProvider, MongoDBLabelsProvider, LabelEncoder
from .transforms import build_transforms, build_batch_transform


class MultiChannelDataModule(pl.LightningDataModule):
    """DataModule for multi-channel microscopy images.
    
    Handles data loading, splitting, and transformations.
    All components (dataset class, train/val transforms) are determined
    by Hydra config _target_ fields and created via hydra.utils.instantiate().
    
    Constructor params match the YAML keys exactly so that
    ``instantiate(cfg.datamodule, _recursive_=False)`` works directly.
    """
    
    def __init__(
        self,
        dataset: DictConfig,
        train_transform: DictConfig,
        val_transform: DictConfig,
        batch_transform: Optional[DictConfig] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_val_split: float = 0.66,
        use_mongodb: bool = False,
        max_wells_per_label: Optional[int] = None,
        exclude_wells: Optional[List] = None,
    ):
        """
        Args:
            dataset: Hydra config for the dataset (contains _target_ and dataset params).
            train_transform: List of transform dicts with "name" + params, or legacy
                            dict with _target_ pointing to a factory function.
            val_transform: Same format as train_transform.
            batch_transform: Optional batch-level transform config (Mixup/CutMix).
                            Dict with "name" + params. Applied in training_step.
            batch_size: Batch size for data loaders.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            train_val_split: Fraction of data for training (rest for validation).
            use_mongodb: Whether to use MongoDB for labels.
            max_wells_per_label: Max wells per label class (None = all).
            exclude_wells: List of [plate, well] pairs to exclude (corrupted images).
        """
        super().__init__()
        
        # Convert exclude_wells to plain Python types BEFORE save_hyperparameters
        # to prevent OmegaConf ListConfig from leaking into checkpoints
        if exclude_wells is not None:
            from omegaconf import OmegaConf
            raw = OmegaConf.to_container(exclude_wells) if hasattr(exclude_wells, '_metadata') else exclude_wells
            exclude_wells = [tuple(w) for w in raw]
        
        self.save_hyperparameters(ignore=["dataset", "train_transform", "val_transform", "batch_transform"])
        
        self.dataset_cfg = dataset
        self.train_transform_cfg = train_transform
        self.val_transform_cfg = val_transform
        self.batch_transform_cfg = batch_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_split = train_val_split
        self.use_mongodb = use_mongodb
        self.max_wells_per_label = max_wells_per_label
        self.exclude_wells = exclude_wells
        
        # Expose dataset-level params for external access
        self.root_dir = dataset.root_dir
        self.channels = list(dataset.channels)
        self.crop_size = dataset.crop_size
        
        # Build batch-level transform (Mixup/CutMix) — exposed for model to use
        self.batch_transform = build_batch_transform(
            OmegaConf.to_container(batch_transform, resolve=True) if batch_transform is not None else None
        )
        
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
        
        self._setup_datasets(labels_dict)
    
    def _setup_datasets(self, labels_dict: Dict):
        """Setup train/val datasets using hydra.utils.instantiate().
        
        The dataset class and transforms are fully determined by config _target_ fields.
        No hardcoded class names — adding a new dataset type only requires a new YAML.
        """
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
        
        # Build transforms: list-based (new) or _target_-based (legacy)
        train_transform = self._build_transform(self.train_transform_cfg)
        val_transform = self._build_transform(self.val_transform_cfg)
        
        # Instantiate datasets: use get_class() + direct constructor call because
        # labels_dict has tuple keys and label_encoder is a runtime object,
        # neither of which can pass through OmegaConf.merge inside instantiate().
        dataset_class = get_class(self.dataset_cfg._target_)
        dataset_kwargs = OmegaConf.to_container(self.dataset_cfg, resolve=True)
        dataset_kwargs.pop("_target_")
        
        self.train_dataset = dataset_class(
            **dataset_kwargs,
            labels_dict=train_labels,
            label_encoder=self.label_encoder,
            transform=train_transform,
        )
        
        self.val_dataset = dataset_class(
            **dataset_kwargs,
            labels_dict=val_labels,
            label_encoder=self.label_encoder,
            transform=val_transform,
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
        if getattr(self.dataset_cfg, 'verbose', False):
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
    
    @staticmethod
    def _build_transform(cfg):
        """Build transforms from config.
        
        Supports two formats:
          1. List of dicts (new): [{name: RandomHorizontalFlip, p: 0.5}, ...]
             -> uses build_transforms() registry-based builder
          2. Dict with _target_ (legacy): {_target_: src.transforms.get_train_transforms, ...}
             -> uses hydra.utils.instantiate()
        """
        if isinstance(cfg, (list, ListConfig)):
            # New list-based format
            return build_transforms(OmegaConf.to_container(cfg, resolve=True)
                                    if isinstance(cfg, ListConfig) else cfg)
        else:
            # Legacy _target_ format
            return instantiate(cfg)
