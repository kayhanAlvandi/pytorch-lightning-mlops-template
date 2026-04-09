"""Training script for CNN classifier."""
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger

from src.callbacks import LogBestModelToMLflow
from src.config import Config
from src.datamodule import MultiChannelDataModule
from src.model import CNNClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="Override channels from config (e.g., --channels 1 2 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help="Override crop size from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this run (for MLflow and TensorBoard)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Tags for this run (e.g., --tags baseline shallow)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')  # Enable Tensor Cores
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Override config with command line arguments
    if args.channels is not None:
        config.data.channels = args.channels
    if args.batch_size is not None:
        config.dataloader.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.crop_size is not None:
        config.data.crop_size = args.crop_size
    
    print("=" * 60)
    print("Configuration:")
    print(f"  Data root: {config.data.root_dir}")
    print(f"  Channels: {config.data.channels}")
    print(f"  Crop size: {config.data.crop_size}")
    print(f"  Batch size: {config.dataloader.batch_size}")
    print(f"  Max epochs: {config.training.max_epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Use MongoDB: {config.dataloader.use_mongodb}")
    print("=" * 60)
    
    # Create data module
    datamodule = MultiChannelDataModule(
        root_dir=config.data.root_dir,
        channels=config.data.channels,
        crop_size=config.data.crop_size,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        train_val_split=config.dataloader.train_val_split,
        use_mongodb=config.dataloader.use_mongodb,
        use_tiling=config.dataloader.use_tiling,
        tile_stride=config.dataloader.tile_stride,
        cache_size=config.dataloader.cache_size,
        max_wells_per_label=config.dataloader.max_wells_per_label,
        max_samples_per_label=config.dataloader.max_samples_per_label,
        verbose=config.dataloader.verbose,
        exclude_wells=config.dataloader.exclude_wells,
    )
    
    # Setup data module to get label encoder with num_classes
    datamodule.setup()
    num_classes = datamodule.label_encoder.num_classes

    # Prepare example input for MLflow logging callback (single sample from val loader)
    val_loader = datamodule.val_dataloader()
    try:
        example_batch = next(iter(val_loader))
    except StopIteration:
        raise RuntimeError("Validation dataloader returned no batches; cannot build example input.")
    example_input, _ = example_batch
    example_input = example_input[:1].detach().cpu()
    
    # Create model
    model = CNNClassifier(
        in_channels=len(config.data.channels),
        num_classes=num_classes,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        dropout=config.model.dropout,
        num_blocks=config.model.num_blocks,
        base_channels=config.model.base_channels,
        channel_multiplier=config.model.channel_multiplier,
        hidden_dim=config.model.hidden_dim,
        class_names=datamodule.label_encoder.classes,
    )
    
    # Store optimizer and scheduler config in model for configure_optimizers
    model._optimizer_config = config.training.optimizer
    model._scheduler_config = config.training.scheduler
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="cnn-{epoch:02d}-{val_acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [
        checkpoint_callback,
        # EarlyStopping(
        #     monitor="val/loss",
        #     patience=10,
        #     mode="min",
        # ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]
    
    # Setup dual logging: TensorBoard for visualization + MLflow for tracking
    # Use custom version name if provided
    tb_version = args.run_name if args.run_name else None
    
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name="image_classifier",
        version=tb_version,
    )
    
    mlflow_logger = MLFlowLogger(
        experiment_name="image_classifier",
        tracking_uri="file:./mlruns",
        log_model=False,
        run_name=args.run_name,
    )
    
    # Log all config parameters to MLflow
    mlflow_logger.log_hyperparams({
        "data.root_dir": config.data.root_dir,
        "data.channels": config.data.channels,
        "data.crop_size": config.data.crop_size,
        "dataloader.batch_size": config.dataloader.batch_size,
        "dataloader.num_workers": config.dataloader.num_workers,
        "dataloader.train_val_split": config.dataloader.train_val_split,
        "dataloader.use_tiling": config.dataloader.use_tiling,
        "dataloader.cache_size": config.dataloader.cache_size,
        "dataloader.max_wells_per_label": config.dataloader.max_wells_per_label,
        "dataloader.max_samples_per_label": config.dataloader.max_samples_per_label,
        "training.max_epochs": config.training.max_epochs,
        "training.learning_rate": config.training.learning_rate,
        "training.weight_decay": config.training.weight_decay,
        "model.name": config.model.name,
        "model.dropout": config.model.dropout,
        "model.num_blocks": config.model.num_blocks,
        "model.base_channels": config.model.base_channels,
        "model.channel_multiplier": config.model.channel_multiplier,
        "model.hidden_dim": config.model.hidden_dim,
        "seed": args.seed,
        # Optimizer parameters
        "optimizer.type": config.training.optimizer.type,
        "optimizer.momentum": config.training.optimizer.momentum,
        "optimizer.nesterov": config.training.optimizer.nesterov,
        # Scheduler parameters
        "scheduler.type": config.training.scheduler.type,
        "scheduler.mode": config.training.scheduler.mode,
        "scheduler.factor": config.training.scheduler.factor,
        "scheduler.patience": config.training.scheduler.patience,
        "scheduler.T_max": config.training.scheduler.T_max,
        "scheduler.eta_min": config.training.scheduler.eta_min,
        "scheduler.step_size": config.training.scheduler.step_size,
        "scheduler.gamma": config.training.scheduler.gamma,
    })
    
    # Log config file as artifact
    import shutil
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, args.config)
    
    # Add custom tags if provided
    if args.tags:
        for tag in args.tags:
            mlflow_logger.experiment.set_tag(mlflow_logger.run_id, f"tag_{tag}", "true")
        mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "tags", ",".join(args.tags))
    
    log_best_callback = LogBestModelToMLflow(
        checkpoint_callback=checkpoint_callback,
        example_input=example_input,
        mlflow_logger=mlflow_logger,
        registered_model_name=config.model.name,
        artifact_path="best_model",
    )
    callbacks.append(log_best_callback)

    loggers = [tb_logger, mlflow_logger]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        callbacks=callbacks,
        logger=loggers,
        accelerator="cuda",
        devices=1,
        precision="16-mixed",  # Mixed precision for faster training
        log_every_n_steps=10,
        deterministic=True,
        gradient_clip_val=1e4,  # Clip gradients to prevent spikes
    )
    
    # Train
    print("\nStarting training...")
    if args.ckpt:
        print(f"Resuming from checkpoint: {args.ckpt}")
    trainer.fit(model, datamodule, ckpt_path=args.ckpt)
    
    # Test with best model
    print("\nTesting with best model...")
    trainer.test(model, datamodule, ckpt_path="best")
    
    print("\nTraining complete!")
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_ckpt_path}")
    print("Best model logging handled automatically by LogBestModelToMLflow callback.")
    print(f"\nMLflow tracking URI: {mlflow_logger.experiment.tracking_uri}")
    print(f"MLflow run ID: {mlflow_logger.run_id}")
    print("View experiments: mlflow ui")


if __name__ == "__main__":
    main()
