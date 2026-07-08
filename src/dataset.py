"""Custom Dataset for multi-channel microscopy images (JXL/TIF)."""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


try:
    import pillow_jxl  # JXL support for PIL  # noqa: F401
except ImportError:
    print("Warning: pillow-jxl not installed. JXL files may not load.")


class LabelEncoder:
    """Encode string labels to integers and decode back.
    
    Maintains a consistent mapping between string labels and integer indices.
    """
    
    def __init__(self):
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self._fitted = False
    
    def fit(self, labels: List[str]) -> "LabelEncoder":
        """Fit encoder on a list of string labels."""
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self._fitted = True
        return self
    
    def transform(self, labels: List[str]) -> List[int]:
        """Transform string labels to integers."""
        if not self._fitted:
            raise ValueError("LabelEncoder must be fitted before transform")
        return [self.label_to_idx[label] for label in labels]
    
    def fit_transform(self, labels: List[str]) -> List[int]:
        """Fit and transform in one step."""
        self.fit(labels)
        return self.transform(labels)
    
    def inverse_transform(self, indices: List[int]) -> List[str]:
        """Convert integer indices back to string labels."""
        return [self.idx_to_label[idx] for idx in indices]
    
    def encode(self, label: str) -> int:
        """Encode a single label."""
        return self.label_to_idx[label]
    
    def decode(self, idx: int) -> str:
        """Decode a single index."""
        return self.idx_to_label[idx]
    
    @property
    def num_classes(self) -> int:
        """Return number of unique classes."""
        return len(self.label_to_idx)
    
    @property
    def classes(self) -> List[str]:
        """Return list of class names in order."""
        return [self.idx_to_label[i] for i in range(len(self.idx_to_label))]


class MultiChannelImageDataset(Dataset):
    """Dataset for loading multi-channel microscopy images.
    
    Each sample consists of multiple channels (C01-C05) from the same field (F).
    Labels are retrieved based on plate and well information.
    
    Filename pattern: {plate}_{well}_T{time}F{field}L{layer}A{action}Z{z}C{channel}.jxl
    """
    
    # Pattern to parse filename components (supports .jxl and .tif)
    FILENAME_PATTERN = re.compile(
        r"(?P<plate>.+?)_(?P<well>[A-Z]\d+)_T(?P<time>\d+)F(?P<field>\d+)L(?P<layer>\d+)A(?P<action>\d+)Z(?P<z>\d+)C(?P<channel>\d+)\.(?:jxl|tif)$",
        re.IGNORECASE
    )
    
    SUPPORTED_EXTENSIONS = (".jxl", ".tif")
    
    def __init__(
        self,
        root_dir: str,
        channels: List[int],
        labels_dict: Dict[Tuple[str, str], str],
        label_encoder: LabelEncoder,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root_dir: Root directory containing images.
            channels: List of channel numbers to use (1-5).
            labels_dict: Dictionary mapping (plate, well) to string label.
            label_encoder: Fitted LabelEncoder to convert string labels to int.
            transform: Optional transform to apply to samples.
        """
        self.root_dir = Path(root_dir)
        self.channels = sorted(channels)
        self.labels_dict = labels_dict
        self.label_encoder = label_encoder
        self.transform = transform
        
        # Build list of unique samples (plate, well, field combinations)
        self.samples = self._build_sample_list()
        
    def _build_sample_list(self) -> List[Dict]:
        """Build list of unique samples based on available files."""
        samples = {}
        
        # Iterate over all supported extensions
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in self.root_dir.glob(f"*{ext}"):
                match = self.FILENAME_PATTERN.match(file_path.name)
                if not match:
                    continue
                    
                info = match.groupdict()
                plate = info["plate"]
                well = info["well"]
                field = info["field"]
                channel = int(info["channel"])
                
                # Check if this channel is one we want
                if channel not in self.channels:
                    continue
                
                # Create unique sample key (plate, well, field)
                sample_key = (plate, well, field)
                
                if sample_key not in samples:
                    # Check if we have a label for this plate/well
                    if (plate, well) not in self.labels_dict:
                        continue
                        
                    samples[sample_key] = {
                        "plate": plate,
                        "well": well,
                        "field": field,
                        "label": self.labels_dict[(plate, well)],
                        "channel_files": {},
                    }
                
                samples[sample_key]["channel_files"][channel] = file_path
        
        # Filter to only samples that have all required channels
        valid_samples = []
        for sample_key, sample in samples.items():
            if all(ch in sample["channel_files"] for ch in self.channels):
                valid_samples.append(sample)
        
        return valid_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_image(self, file_path: Path) -> np.ndarray:
        """Load a single image file."""
        suffix = file_path.suffix.lower()
        if suffix == ".tif":
            image_source = cv2.imread(str(file_path), -1)
        elif suffix == ".jxl":
            image_source = Image.open(file_path)
        else:
            raise ValueError("image path should end with .tif or .jxl")
        return np.array(image_source, dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        # Load all channels for this sample
        channel_images = []
        for ch in self.channels:
            file_path = sample["channel_files"][ch]
            img = self._load_image(file_path)
            
            # Percentile-clip to [0, 1]: robust to outliers (hot/dead pixels)
            p_lo, p_hi = np.percentile(img, [1, 99.5])
            if p_hi - p_lo > 0:
                img = np.clip(img, p_lo, p_hi)
                img = ((img - p_lo) / (p_hi - p_lo)).astype(np.float32)
            else:
                img = np.zeros_like(img)
            
            channel_images.append(img)
        
        # Stack channels: (C, H, W)
        image = np.stack(channel_images, axis=0)
        
        # Convert to tensor
        image = torch.from_numpy(image)
        
        # Encode string label to integer
        label = self.label_encoder.encode(sample["label"])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class LRUImageCache:
    """LRU cache for loaded images to avoid repeated disk reads."""
    
    def __init__(self, max_size: int = 16):
        self.max_size = max_size
        self.cache: Dict[str, np.ndarray] = {}
        self.access_order: List[str] = []
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get image from cache, returns None if not found."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, image: np.ndarray) -> None:
        """Add image to cache, evicting oldest if full."""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Evict least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = image
        self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


class TiledMultiChannelDataset(Dataset):
    """Dataset that extracts grid tiles from large multi-channel images.
    
    Dynamically calculates tile grid based on image size and crop size.
    Uses LRU cache to efficiently reuse loaded images across tile requests.
    """
    
    FILENAME_PATTERN = MultiChannelImageDataset.FILENAME_PATTERN
    SUPPORTED_EXTENSIONS = MultiChannelImageDataset.SUPPORTED_EXTENSIONS
    
    def __init__(
        self,
        root_dir: str,
        channels: List[int],
        labels_dict: Dict[Tuple[str, str], str],
        label_encoder: LabelEncoder,
        crop_size: int = 224,
        stride: Optional[int] = None,
        transform: Optional[Callable] = None,
        cache_size: int = 16,
        max_samples_per_label: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Args:
            root_dir: Root directory containing images.
            channels: List of channel numbers to use (1-5).
            labels_dict: Dictionary mapping (plate, well) to string label.
            label_encoder: Fitted LabelEncoder to convert string labels to int.
            crop_size: Size of each tile (crop_size x crop_size).
            stride: Step between tiles. If None, uses crop_size (non-overlapping).
            transform: Optional transform to apply (e.g., normalization, augmentation).
            cache_size: Number of images to keep in LRU cache.
            max_samples_per_label: Max samples per label for balanced dataset.
            verbose: Print detailed sample selection info.
        """
        self.root_dir = Path(root_dir)
        self.channels = sorted(channels)
        self.labels_dict = labels_dict
        self.label_encoder = label_encoder
        self.crop_size = crop_size
        self.stride = stride or crop_size
        self.transform = transform
        self.cache = LRUImageCache(max_size=cache_size)
        self.max_samples_per_label = max_samples_per_label
        self.verbose = verbose
        
        # Build sample list (one per image/field)
        self.samples = self._build_sample_list()
        
        # Balance samples per label if requested
        if self.max_samples_per_label is not None:
            self.samples = self._balance_samples(self.samples)
        
        # Build tile index: list of (sample_idx, row, col) for all tiles
        self.tiles = self._build_tile_index()
    
    def _build_sample_list(self) -> List[Dict]:
        """Build list of unique samples based on available files."""
        samples = {}
        
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in self.root_dir.glob(f"*{ext}"):
                match = self.FILENAME_PATTERN.match(file_path.name)
                if not match:
                    continue
                
                info = match.groupdict()
                plate = info["plate"]
                well = info["well"]
                field = info["field"]
                channel = int(info["channel"])
                
                if channel not in self.channels:
                    continue
                
                sample_key = (plate, well, field)
                
                if sample_key not in samples:
                    if (plate, well) not in self.labels_dict:
                        continue
                    
                    samples[sample_key] = {
                        "plate": plate,
                        "well": well,
                        "field": field,
                        "label": self.labels_dict[(plate, well)],
                        "channel_files": {},
                        "image_size": None,
                    }
                
                samples[sample_key]["channel_files"][channel] = file_path
        
        # Filter to samples with all channels
        valid_samples = []
        for sample_key, sample in samples.items():
            if all(ch in sample["channel_files"] for ch in self.channels):
                valid_samples.append(sample)
        
        if not valid_samples:
            return valid_samples
        
        # Get image size from first sample only (all images are same size)
        first_sample = valid_samples[0]
        first_channel = self.channels[0]
        file_path = first_sample["channel_files"][first_channel]
        img = self._load_single_image(file_path)
        image_size = (img.shape[0], img.shape[1])
        
        # Apply same size to all samples
        for sample in valid_samples:
            sample["image_size"] = image_size
        
        return valid_samples
    
    def _balance_samples(self, samples: List[Dict]) -> List[Dict]:
        """Balance samples by taking max N per label with random selection.
        
        Args:
            samples: List of sample dicts with 'label' key
            
        Returns:
            Balanced list with max_samples_per_label samples per class
        """
        import random
        from collections import defaultdict
        
        # Group samples by label
        samples_by_label = defaultdict(list)
        for sample in samples:
            samples_by_label[sample["label"]].append(sample)
        
        # Random sample from each label
        balanced = []
        for label, label_samples in samples_by_label.items():
            random.shuffle(label_samples)
            selected = label_samples[:self.max_samples_per_label]
            balanced.extend(selected)
            print(f"  {label}: {len(label_samples)} -> {len(selected)} samples")
        
        print(f"Balanced to {self.max_samples_per_label} samples/label: {len(samples)} -> {len(balanced)}")
        
        # Print selected samples details if verbose
        if self.verbose:
            print("Selected samples:")
            for sample in balanced:
                print(f"  {sample['plate']}/{sample['well']}/F{sample['field']} -> {sample['label']}")
        
        return balanced
    
    def _build_tile_index(self) -> List[Tuple[int, int, int]]:
        """Build index of all tiles: (sample_idx, top, left)."""
        tiles = []
        
        for sample_idx, sample in enumerate(self.samples):
            h, w = sample["image_size"]
            
            # Calculate number of tiles in each dimension
            n_rows = max(1, (h - self.crop_size) // self.stride + 1)
            n_cols = max(1, (w - self.crop_size) // self.stride + 1)
            
            for row in range(n_rows):
                for col in range(n_cols):
                    top = min(row * self.stride, h - self.crop_size)
                    left = min(col * self.stride, w - self.crop_size)
                    tiles.append((sample_idx, top, left))
        
        return tiles
    
    def _load_single_image(self, file_path: Path) -> np.ndarray:
        """Load a single image file."""
        suffix = file_path.suffix.lower()
        if suffix == ".tif":
            image_source = cv2.imread(str(file_path), -1)
        elif suffix == ".jxl":
            image_source = Image.open(file_path)
        else:
            raise ValueError("image path should end with .tif or .jxl")
        return np.array(image_source, dtype=np.float32)
    
    def _load_sample_image(self, sample_idx: int) -> np.ndarray:
        """Load full multi-channel image for a sample, using cache."""
        sample = self.samples[sample_idx]
        cache_key = f"{sample['plate']}_{sample['well']}_{sample['field']}"
        
        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Load all channels
        channel_images = []
        for ch in self.channels:
            file_path = sample["channel_files"][ch]
            img = self._load_single_image(file_path)
            
            # Percentile-clip to [0, 1]: robust to outliers (hot/dead pixels)
            p_lo, p_hi = np.percentile(img, [1, 99.5])
            if p_hi - p_lo > 0:
                img = np.clip(img, p_lo, p_hi)
                img = ((img - p_lo) / (p_hi - p_lo)).astype(np.float32)
            else:
                img = np.zeros_like(img)
            
            channel_images.append(img)
        
        # Stack channels: (C, H, W)
        image = np.stack(channel_images, axis=0)
        
        # Store in cache
        self.cache.put(cache_key, image)
        
        return image
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_idx, top, left = self.tiles[idx]
        sample = self.samples[sample_idx]
        
        # Load full image (from cache if available)
        image = self._load_sample_image(sample_idx)
        
        # Extract tile
        tile = image[:, top:top + self.crop_size, left:left + self.crop_size]
        
        # Convert to tensor
        tile = torch.from_numpy(tile.copy())
        
        # Encode label
        label = self.label_encoder.encode(sample["label"])
        
        # Apply transforms (e.g., augmentation, normalization)
        if self.transform:
            tile = self.transform(tile)
        
        return tile, label
    
    @property
    def num_samples(self) -> int:
        """Number of original samples (images)."""
        return len(self.samples)
    
    @property
    def num_tiles(self) -> int:
        """Total number of tiles across all samples."""
        return len(self.tiles)
    
    def get_tiles_per_sample(self) -> List[int]:
        """Get number of tiles for each sample."""
        counts = [0] * len(self.samples)
        for sample_idx, _, _ in self.tiles:
            counts[sample_idx] += 1
        return counts


class DummyLabelsProvider:
    """Dummy label provider for testing without MongoDB.
    
    Generates random string labels based on plate/well combinations.
    Replace with MongoDB-based provider in production.
    """
    
    # Default class names for dummy labels
    DEFAULT_CLASSES = ["ClassA", "ClassB", "ClassC", "ClassD"]
    
    def __init__(self, class_names: Optional[List[str]] = None, seed: int = 42):
        self.class_names = class_names or self.DEFAULT_CLASSES
        self.seed = seed
    
    def get_labels(
        self,
        root_dir: str,
        exclude_wells: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[Tuple[str, str], str]:
        """Generate dummy string labels for all plate/well combinations in directory.
        
        Args:
            root_dir: Root directory containing images.
            exclude_wells: List of (plate, well) tuples to exclude (corrupted images).
        """
        root_path = Path(root_dir)
        plate_wells = set()
        exclude_set = set(exclude_wells) if exclude_wells else set()
        
        for ext in MultiChannelImageDataset.SUPPORTED_EXTENSIONS:
            for file_path in root_path.glob(f"*{ext}"):
                match = MultiChannelImageDataset.FILENAME_PATTERN.match(file_path.name)
                if match:
                    info = match.groupdict()
                    well_key = (info["plate"], info["well"])
                    if well_key not in exclude_set:
                        plate_wells.add(well_key)
        
        if exclude_wells:
            print(f"Excluded {len(exclude_wells)} wells from dataset")
        
        # Generate consistent random labels
        np.random.seed(self.seed)
        labels = {}
        for plate, well in sorted(plate_wells):
            label_idx = np.random.randint(0, len(self.class_names))
            labels[(plate, well)] = self.class_names[label_idx]
        
        return labels


class MongoDBLabelsProvider:
    """Label provider using MongoDB.
    
    Queries MongoDB to get labels for plate/well combinations.
    """
    
    def __init__(
        self,
        collection: str = "labels",
    ):
        self.collection = collection
    
    def get_labels(
        self,
        root_dir: str,
        exclude_wells: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[Tuple[str, str], str]:
        """Query MongoDB for labels (returns string Treatment values).
        
        Args:
            root_dir: Root directory containing images.
            exclude_wells: List of (plate, well) tuples to exclude (corrupted images).
        """
        from tools.loading import getCategories
        
        root_path = Path(root_dir)
        plate_wells = set()
        exclude_set = set(exclude_wells) if exclude_wells else set()
        
        # Find all plate/well combinations in directory
        for ext in MultiChannelImageDataset.SUPPORTED_EXTENSIONS:
            for file_path in root_path.glob(f"*{ext}"):
                match = MultiChannelImageDataset.FILENAME_PATTERN.match(file_path.name)
                if match:
                    info = match.groupdict()
                    well_key = (info["plate"], info["well"])
                    if well_key not in exclude_set:
                        plate_wells.add(well_key)
        
        if exclude_wells:
            print(f"Excluded {len(exclude_wells)} wells from dataset")
        
        # Convert plate_wells set to list for DataFrame creation
        plates = [pw[0] for pw in plate_wells]
        wells = [pw[1] for pw in plate_wells]
        
        import pandas as pd
        df = pd.DataFrame({"Plate": plates, "Well": wells})
        df.drop_duplicates(inplace=True)
        df_labels =  getCategories(df,collection= "tags")

        # Convert to dictionary format
        labels = dict(zip(zip(df_labels["Plate"], df_labels["Well"]), df_labels["Treatment"]))
        
        return labels
