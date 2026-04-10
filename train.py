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
from src.dataset_versioning import (
    create_dataset_metadata,
    create_dataset_manifest,
    compute_dataset_version,
    get_git_commit_for_model_files,
    check_model_uncommitted_changes,
)


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
        "--resume-from-model",
        type=str,
        default=None,
        help="Resume training from MLflow registered model, e.g. 'SimpleCNN/11' or 'models:/SimpleCNN/latest'",
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


def get_checkpoint_from_mlflow_model(model_ref: str) -> str:
    """
    Download checkpoint artifact from an MLflow registered model version.

    Args:
        model_ref: 'ModelName/version' e.g. 'SimpleCNN/11', or 'models:/SimpleCNN/11'
                   Use 'latest' as version to get the most recent version.

    Returns:
        Local path to downloaded .ckpt file.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri="file:./mlruns")

    # Parse model reference
    ref = model_ref.removeprefix("models:/")
    parts = ref.split("/")
    model_name = parts[0]
    version = parts[1] if len(parts) > 1 else "latest"

    if version == "latest":
        versions = client.get_latest_versions(model_name)
        if not versions:
            raise ValueError(f"No versions found for registered model '{model_name}'")
        version = versions[0].version
        print(f"Resolved 'latest' to version {version} for model '{model_name}'")

    mv = client.get_model_version(model_name, version)
    run_id = mv.run_id
    print(f"Resuming from {model_name} v{version} (run_id={run_id[:8]}...)")

    # Download the checkpoint artifact logged under 'checkpoints/'
    try:
        artifacts = client.list_artifacts(run_id, path="checkpoints")
        if not artifacts:
            raise FileNotFoundError("No files found under 'checkpoints/' artifact path")
        ckpt_filename = artifacts[0].path  # e.g. 'checkpoints/epoch=5-step=100.ckpt'
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=ckpt_filename,
            tracking_uri="file:./mlruns",
        )
        print(f"Downloaded checkpoint: {local_dir}")
        return local_dir
    except Exception as e:
        raise RuntimeError(
            f"Could not download checkpoint for {model_name}/v{version}: {e}\n"
            "Make sure the model was logged with LogBestModelToMLflow after this feature was added."
        ) from e


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
    
    # Get dataset sample counts
    train_samples = len(datamodule.train_dataset)
    val_samples = len(datamodule.val_dataset)
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {train_samples}")
    print(f"  Validation samples: {val_samples}")
    print(f"  Total samples: {train_samples + val_samples}")
    print(f"  Number of classes: {num_classes}")

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
    
    # Log model code version
    model_code_commit = get_git_commit_for_model_files()
    if check_model_uncommitted_changes():
        print("⚠ Warning: model.py has uncommitted changes - reproducibility not guaranteed!")
    
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
        "model.code_commit": model_code_commit or "unknown",
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
    
    # Dataset versioning and tracking
    print("\nCreating dataset metadata...")
    dataset_metadata = create_dataset_metadata(
        datamodule=datamodule,
        config=config,
        train_samples=train_samples,
        val_samples=val_samples,
    )
    
    # Log dataset version and metadata
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "dataset_version", dataset_metadata['dataset_version'])
    mlflow_logger.log_hyperparams({
        "dataset.version": dataset_metadata['dataset_version'],
        "dataset.code_commit": dataset_metadata['dataset_code_commit'] or "unknown",
        "dataset.config_hash": dataset_metadata['config_hash'],
        "dataset.has_uncommitted_changes": dataset_metadata['has_uncommitted_changes'],
        "dataset.train_samples": dataset_metadata['train_samples'],
        "dataset.val_samples": dataset_metadata['val_samples'],
        "dataset.total_samples": dataset_metadata['total_samples'],
        "dataset.num_classes": dataset_metadata['num_classes'],
    })
    
    # Log full dataset metadata as artifact
    import json
    metadata_path = "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(dataset_metadata, f, indent=2, default=str)
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, metadata_path)
    
    # Create and log dataset manifest (detailed sample list)
    print("Creating dataset manifest...")
    manifest_path = create_dataset_manifest(datamodule, output_path="dataset_manifest.json")
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, manifest_path)
    
    # Use MLflow Dataset API for proper dataset tracking
    try:
        import mlflow
        import pandas as pd
        
        # Create a summary DataFrame for MLflow Dataset API
        dataset_summary = pd.DataFrame([{
            'split': 'train',
            'num_samples': train_samples,
            'num_classes': num_classes,
        }, {
            'split': 'val',
            'num_samples': val_samples,
            'num_classes': num_classes,
        }])
        
        # Log as MLflow dataset (must be within active run context)
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            dataset = mlflow.data.from_pandas(
                dataset_summary,
                source=config.data.root_dir,
                name=f"microscopy_dataset_{dataset_metadata['dataset_version']}",
            )
            mlflow.log_input(dataset, context="training")
            print(f"✓ Dataset tracked with version: {dataset_metadata['dataset_version']}")
    except Exception as e:
        print(f"⚠ Warning: Could not log dataset with MLflow Dataset API: {e}")
        import traceback
        traceback.print_exc()
    
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
    
    # Resolve checkpoint path (local file or MLflow registered model)
    ckpt_path = args.ckpt
    if args.resume_from_model:
        if ckpt_path:
            raise ValueError("Cannot use both --ckpt and --resume-from-model at the same time.")
        ckpt_path = get_checkpoint_from_mlflow_model(args.resume_from_model)

    # Train
    print("\nStarting training...")
    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    
    # Test with best model
    print("\nTesting with best model...")
    trainer.test(model, datamodule, ckpt_path="best")
    
    # Clean up current checkpoint after it's been logged to MLflow and tested
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path and Path(best_ckpt_path).exists():
        Path(best_ckpt_path).unlink()
        print(f"Cleaned up local checkpoint: {best_ckpt_path}")
    
    print("\nTraining complete!")
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model was at: {best_ckpt_path} (now deleted, stored in MLflow)")
    print("Best model logging handled automatically by LogBestModelToMLflow callback.")
    print(f"\nMLflow tracking URI: {mlflow_logger.experiment.tracking_uri}")
    print(f"MLflow run ID: {mlflow_logger.run_id}")
    print("View experiments: mlflow ui")


if __name__ == "__main__":
    main()
