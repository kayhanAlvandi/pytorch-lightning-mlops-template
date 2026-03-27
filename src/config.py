"""Configuration management for CNN classifier."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    root_dir: str
    channels: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    crop_size: int = 224


@dataclass
class DataLoaderConfig:
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    use_mongodb: bool = False
    train_val_split: float = 0.8
    use_tiling: bool = True
    tile_stride: Optional[int] = None
    cache_size: int = 16
    max_wells_per_label: Optional[int] = None  # Limit wells per label class
    max_samples_per_label: Optional[int] = None  # Limit samples per label (balanced)
    verbose: bool = False  # Print detailed sample selection info
    exclude_wells: Optional[List[Tuple[str, str]]] = None  # Wells to exclude (corrupted)


@dataclass
class SchedulerConfig:
    type: str = "ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.5
    patience: int = 5
    T_max: Optional[int] = None  # For CosineAnnealingLR
    eta_min: float = 1e-6  # For CosineAnnealingLR


@dataclass
class TrainingConfig:
    max_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class ModelConfig:
    name: str = "SimpleCNN"
    dropout: float = 0.5


@dataclass
class Config:
    data: DataConfig
    dataloader: DataLoaderConfig
    training: TrainingConfig
    model: ModelConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Convert exclude_wells from list of lists to list of tuples
        dataloader_config = config_dict.get("dataloader", {})
        if dataloader_config.get("exclude_wells"):
            dataloader_config["exclude_wells"] = [
                tuple(w) for w in dataloader_config["exclude_wells"]
            ]
        
        training_config = config_dict.get("training", {})
        scheduler_config_data = training_config.get("scheduler")
        if scheduler_config_data is not None:
            training_config["scheduler"] = SchedulerConfig(**scheduler_config_data)
        else:
            training_config["scheduler"] = SchedulerConfig()
        
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            dataloader=DataLoaderConfig(**dataloader_config),
            training=TrainingConfig(**training_config),
            model=ModelConfig(**config_dict.get("model", {})),
        )
    
    def save_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "data": {
                "root_dir": self.data.root_dir,
                "channels": self.data.channels,
                "crop_size": self.data.crop_size,
            },
            "dataloader": {
                "batch_size": self.dataloader.batch_size,
                "num_workers": self.dataloader.num_workers,
                "pin_memory": self.dataloader.pin_memory,
                "train_val_split": self.dataloader.train_val_split,
                "use_mongodb": self.dataloader.use_mongodb,
                "use_tiling": self.dataloader.use_tiling,
                "tile_stride": self.dataloader.tile_stride,
                "cache_size": self.dataloader.cache_size,
            },
            "training": {
                "max_epochs": self.training.max_epochs,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
            },
            "model": {
                "name": self.model.name,
                "dropout": self.model.dropout,
            },
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
