"""Configuration management for CNN classifier using Hydra structured configs."""
from dataclasses import dataclass, field
from typing import List, Optional, Any

from omegaconf import MISSING


# ── Dataset configs (nested inside datamodule) ────────────────────────────
@dataclass
class TiledDatasetConfig:
    _target_: str = "src.dataset.TiledMultiChannelDataset"
    root_dir: str = MISSING
    channels: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    crop_size: int = 224
    stride: Optional[int] = None
    cache_size: int = 16
    max_samples_per_label: Optional[int] = None
    verbose: bool = False


@dataclass
class RandomCropDatasetConfig:
    _target_: str = "src.dataset.MultiChannelImageDataset"
    root_dir: str = MISSING
    channels: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    crop_size: int = 224


# ── Transform configs (nested inside datamodule) ─────────────────────────
@dataclass
class TileTrainTransformConfig:
    _target_: str = "src.transforms.get_tile_train_transforms"


@dataclass
class TileValTransformConfig:
    _target_: str = "src.transforms.get_tile_val_transforms"


@dataclass
class CropTrainTransformConfig:
    _target_: str = "src.transforms.get_train_transforms"
    crop_size: int = 224
    num_channels: int = 5


@dataclass
class CropValTransformConfig:
    _target_: str = "src.transforms.get_val_transforms"
    crop_size: int = 224
    num_channels: int = 5


# ── Datamodule (dataloader-level config + nested dataset + transforms) ────
@dataclass
class DataModuleConfig:
    _target_: str = "src.datamodule.MultiChannelDataModule"
    dataset: Any = field(default_factory=TiledDatasetConfig)
    train_transform: Any = field(default_factory=TileTrainTransformConfig)
    val_transform: Any = field(default_factory=TileValTransformConfig)
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    use_mongodb: bool = False
    train_val_split: float = 0.8
    max_wells_per_label: Optional[int] = None
    exclude_wells: Optional[Any] = None


# ── Model ──────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    _target_: str = "src.model.CNNClassifier"
    dropout: float = 0.5
    num_blocks: int = 4
    base_channels: int = 32
    channel_multiplier: float = 2.0
    hidden_dim: int = 128


# ── Optimizer and Scheduler (sibling keys) ────────────────────────────────
@dataclass
class OptimizerInnerConfig:
    _target_: str = "torch.optim.AdamW"
    lr: float = 0.001
    weight_decay: float = 0.0001


@dataclass
class SchedulerConfig:
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.5
    patience: int = 5


@dataclass
class OptimizerConfig:
    optimizer: OptimizerInnerConfig = field(default_factory=OptimizerInnerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


# ── Callbacks (list of _target_ dicts, instantiated generically) ──────────
# No structured config needed — callbacks is a plain list in YAML.
# Each entry has a _target_ and its constructor args.


# ── Trainer (with nested logger) ──────────────────────────────────────────
@dataclass
class TensorBoardLoggerConfig:
    _target_: str = "pytorch_lightning.loggers.TensorBoardLogger"
    save_dir: str = "logs"
    name: str = "image_classifier"


@dataclass
class MLflowLoggerConfig:
    _target_: str = "pytorch_lightning.loggers.MLFlowLogger"
    experiment_name: str = "image_classifier"
    tracking_uri: str = "file:./mlruns"


@dataclass
class LoggerConfig:
    tensorboard: TensorBoardLoggerConfig = field(default_factory=TensorBoardLoggerConfig)
    mlflow: MLflowLoggerConfig = field(default_factory=MLflowLoggerConfig)


@dataclass
class TrainerConfig:
    _target_: str = "pytorch_lightning.Trainer"
    max_epochs: int = 50
    accelerator: str = "cuda"
    devices: int = 1
    precision: str = "16-mixed"
    log_every_n_steps: int = 10
    deterministic: bool = True
    gradient_clip_val: float = 10000
    logger: LoggerConfig = field(default_factory=LoggerConfig)


# ── Top-level config ──────────────────────────────────────────────────────
@dataclass
class Config:
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    callbacks: List[Any] = field(default_factory=list)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # Runtime parameters
    seed: int = 42
    ckpt: Optional[str] = None
    resume_from_model: Optional[str] = None
    run_name: Optional[str] = None
    tags: Optional[List[str]] = None
