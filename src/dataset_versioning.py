"""Dataset versioning utilities for tracking dataset composition and code."""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess


def get_git_commit_for_model_files() -> Optional[str]:
    """Get the last git commit hash that modified model-related files."""
    model_files = [
        'src/model.py',
    ]
    
    try:
        git_hash = subprocess.check_output(
            ['git', 'log', '-1', '--format=%H', '--'] + model_files,
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return git_hash if git_hash else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def check_model_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes to model files."""
    model_files = ['src/model.py']
    try:
        result = subprocess.check_output(
            ['git', 'status', '--porcelain', '--'] + model_files,
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return len(result) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_git_commit_for_dataset_files() -> Optional[str]:
    """Get the last git commit hash that modified dataset-related files."""
    dataset_files = [
        'src/dataset.py',
        'src/datamodule.py',
        'src/transforms.py',
    ]
    
    try:
        # Get last commit that touched any of these files
        git_hash = subprocess.check_output(
            ['git', 'log', '-1', '--format=%H', '--'] + dataset_files,
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return git_hash if git_hash else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def check_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes to dataset files."""
    dataset_files = [
        'src/dataset.py',
        'src/datamodule.py',
        'src/transforms.py',
    ]
    
    try:
        # Check for uncommitted changes
        result = subprocess.check_output(
            ['git', 'status', '--porcelain', '--'] + dataset_files,
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return len(result) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Compute hash of data and dataloader configuration."""
    relevant_config = {
        'data': config_dict.get('data', {}),
        'dataloader': config_dict.get('dataloader', {}),
    }
    
    config_str = json.dumps(relevant_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def compute_dataset_version(config_dict: Dict[str, Any]) -> str:
    """
    Compute dataset version from git commit + config.
    
    Format: {git_commit[:8]}_{config_hash[:8]}
    Changes only when dataset code or data/dataloader config changes.
    """
    git_commit = get_git_commit_for_dataset_files()
    config_hash = compute_config_hash(config_dict)
    
    if git_commit:
        code_part = git_commit[:8]
    else:
        code_part = "no_git"
    
    # Warn about uncommitted changes
    if check_uncommitted_changes():
        print("⚠ Warning: Dataset code has uncommitted changes!")
        print("  Commit your changes for full reproducibility.")
        code_part = f"{code_part}_dirty"
    
    return f"{code_part}_{config_hash[:8]}"


def create_dataset_metadata(
    datamodule,
    config,
    train_samples: int,
    val_samples: int,
) -> Dict[str, Any]:
    """
    Create comprehensive dataset metadata for MLflow logging.
    
    Args:
        datamodule: PyTorch Lightning DataModule instance
        config: Configuration object
        train_samples: Number of training samples
        val_samples: Number of validation samples
    
    Returns:
        Dictionary with dataset metadata
    """
    # Get label encoder
    label_encoder = datamodule.label_encoder
    
    # Collect wells used (if available)
    wells_used = []
    if hasattr(datamodule, 'train_dataset') and hasattr(datamodule.train_dataset, 'samples'):
        wells_set = set()
        for sample in datamodule.train_dataset.samples:
            if isinstance(sample, dict) and ('plate' in sample) and ('well' in sample):
                wells_set.add(f"{sample['plate']}/{sample['well']}")
        wells_used = sorted(list(wells_set))
        
        if not wells_used and len(datamodule.train_dataset.samples) > 0:
            # Debug: print first sample structure if wells_used is empty
            print(f"⚠ Warning: Could not extract wells. First sample structure: {list(datamodule.train_dataset.samples[0].keys())}")
    
    config_dict = {
        'data': {
            'root_dir': config.data.root_dir,
            'channels': config.data.channels,
            'crop_size': config.data.crop_size,
        },
        'dataloader': {
            'batch_size': config.dataloader.batch_size,
            'train_val_split': config.dataloader.train_val_split,
            'use_tiling': config.dataloader.use_tiling,
            'tile_stride': config.dataloader.tile_stride,
            'max_wells_per_label': config.dataloader.max_wells_per_label,
            'max_samples_per_label': config.dataloader.max_samples_per_label,
            'exclude_wells': config.dataloader.exclude_wells,
        }
    }
    
    metadata = {
        'dataset_version': compute_dataset_version(config_dict),
        'dataset_code_commit': get_git_commit_for_dataset_files(),
        'config_hash': compute_config_hash(config_dict),
        'has_uncommitted_changes': check_uncommitted_changes(),
        'source_path': config.data.root_dir,
        'num_classes': label_encoder.num_classes,
        'class_names': label_encoder.classes,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'total_samples': train_samples + val_samples,
        'train_val_split': config.dataloader.train_val_split,
        'channels': config.data.channels,
        'crop_size': config.data.crop_size,
        'use_tiling': config.dataloader.use_tiling,
        'max_wells_per_label': config.dataloader.max_wells_per_label,
        'max_samples_per_label': config.dataloader.max_samples_per_label,
        'excluded_wells': config.dataloader.exclude_wells or [],
        'wells_used': wells_used,
    }
    
    return metadata


def create_dataset_manifest(
    datamodule,
    output_path: str = "dataset_manifest.json"
) -> str:
    """
    Create a detailed manifest of all samples in the dataset.
    
    Args:
        datamodule: PyTorch Lightning DataModule
        output_path: Path to save manifest
    
    Returns:
        Path to manifest file
    """
    manifest = {
        'train_samples': [],
        'val_samples': [],
    }
    
    # Collect train samples
    if hasattr(datamodule, 'train_dataset'):
        for idx, sample in enumerate(datamodule.train_dataset.samples):
            manifest['train_samples'].append({
                'index': idx,
                'plate': sample['plate'],
                'well': sample['well'],
                'field': sample['field'],
                'label': sample['label'],
            })
    
    # Collect val samples
    if hasattr(datamodule, 'val_dataset'):
        for idx, sample in enumerate(datamodule.val_dataset.samples):
            manifest['val_samples'].append({
                'index': idx,
                'plate': sample['plate'],
                'well': sample['well'],
                'field': sample['field'],
                'label': sample['label'],
            })
    
    # Save manifest
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return output_path
