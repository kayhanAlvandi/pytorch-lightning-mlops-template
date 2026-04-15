"""Training script for CNN classifier."""
import json
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger

from src.callbacks import LogBestModelToMLflow
from src.datamodule import MultiChannelDataModule
from src.model import CNNClassifier
from src.dataset_versioning import (
    create_dataset_metadata,
    create_dataset_manifest,
    compute_dataset_version,
    get_git_commit_for_model_files,
    check_model_uncommitted_changes,
)


def get_checkpoint_from_mlflow_model(model_ref: str, tracking_uri: str = "file:./mlruns") -> str:
    """
    Download checkpoint artifact from an MLflow registered model version.

    Args:
        model_ref: 'ModelName/version' e.g. 'SimpleCNN/11', or 'models:/SimpleCNN/11'
                   Use 'latest' as version to get the most recent version.
        tracking_uri: MLflow tracking URI.

    Returns:
        Local path to downloaded .ckpt file.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)

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
            tracking_uri=tracking_uri,
        )
        print(f"Downloaded checkpoint: {local_dir}")
        return local_dir
    except Exception as e:
        raise RuntimeError(
            f"Could not download checkpoint for {model_name}/v{version}: {e}\n"
            "Make sure the model was logged with LogBestModelToMLflow after this feature was added."
        ) from e


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Print resolved config
    print("=" * 60)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision('medium')  # Enable Tensor Cores
    
    # Convert exclude_wells from list of lists to list of tuples
    exclude_wells = None
    if cfg.datamodule.exclude_wells is not None:
        exclude_wells = [
            tuple(w) for w in OmegaConf.to_container(cfg.datamodule.exclude_wells)
        ]
    
    # Create data module
    datamodule = MultiChannelDataModule(
        dataset_cfg=cfg.datamodule.dataset,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        pin_memory=cfg.datamodule.pin_memory,
        train_val_split=cfg.datamodule.train_val_split,
        use_mongodb=cfg.datamodule.use_mongodb,
        max_wells_per_label=cfg.datamodule.max_wells_per_label,
        exclude_wells=exclude_wells,
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
        in_channels=len(cfg.datamodule.dataset.channels),
        num_classes=num_classes,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        dropout=cfg.model.dropout,
        num_blocks=cfg.model.num_blocks,
        base_channels=cfg.model.base_channels,
        channel_multiplier=cfg.model.channel_multiplier,
        hidden_dim=cfg.model.hidden_dim,
        class_names=datamodule.label_encoder.classes,
    )
    
    # Store optimizer and scheduler config in model for configure_optimizers
    model._optimizer_config = cfg.optimizer
    model._scheduler_config = cfg.optimizer.scheduler
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.callbacks.checkpoint.dirpath,
        filename=cfg.callbacks.checkpoint.filename,
        monitor=cfg.callbacks.checkpoint.monitor,
        mode=cfg.callbacks.checkpoint.mode,
        save_top_k=cfg.callbacks.checkpoint.save_top_k,
        save_last=cfg.callbacks.checkpoint.save_last,
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]
    
    # Add early stopping if enabled
    if cfg.callbacks.early_stopping.enabled:
        callbacks.append(EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
            mode=cfg.callbacks.early_stopping.mode,
        ))
    
    # Setup dual logging: TensorBoard for visualization + MLflow for tracking
    tb_version = cfg.run_name if cfg.run_name else None
    
    tb_logger = TensorBoardLogger(
        save_dir=cfg.trainer.logger.tensorboard.save_dir,
        name=cfg.trainer.logger.tensorboard.name,
        version=tb_version,
    )
    
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.trainer.logger.mlflow.experiment_name,
        tracking_uri=cfg.trainer.logger.mlflow.tracking_uri,
        log_model=False,
        run_name=cfg.run_name,
    )
    
    # Enable system metrics logging (CPU, GPU, memory usage)
    mlflow.enable_system_metrics_logging()
    
    # Log model code version
    model_code_commit = get_git_commit_for_model_files()
    if check_model_uncommitted_changes():
        print("⚠ Warning: model.py has uncommitted changes - reproducibility not guaranteed!")
    
    # Log all config parameters to MLflow (flatten the Hydra config)
    flat_cfg = {
        "datamodule.dataset._target_": cfg.datamodule.dataset._target_,
        "datamodule.dataset.root_dir": cfg.datamodule.dataset.root_dir,
        "datamodule.dataset.channels": str(list(cfg.datamodule.dataset.channels)),
        "datamodule.dataset.crop_size": cfg.datamodule.dataset.crop_size,
        "datamodule.batch_size": cfg.datamodule.batch_size,
        "datamodule.num_workers": cfg.datamodule.num_workers,
        "datamodule.train_val_split": cfg.datamodule.train_val_split,
        "datamodule.max_wells_per_label": cfg.datamodule.max_wells_per_label,
        "training.max_epochs": cfg.training.max_epochs,
        "training.learning_rate": cfg.training.learning_rate,
        "training.weight_decay": cfg.training.weight_decay,
        "model.name": cfg.model.name,
        "model.dropout": cfg.model.dropout,
        "model.num_blocks": cfg.model.num_blocks,
        "model.base_channels": cfg.model.base_channels,
        "model.channel_multiplier": cfg.model.channel_multiplier,
        "model.hidden_dim": cfg.model.hidden_dim,
        "model.code_commit": model_code_commit or "unknown",
        "seed": cfg.seed,
        "optimizer.type": cfg.optimizer.type,
        "optimizer.momentum": cfg.optimizer.momentum,
        "optimizer.nesterov": cfg.optimizer.nesterov,
        "optimizer.scheduler.type": cfg.optimizer.scheduler.type,
        "optimizer.scheduler.mode": cfg.optimizer.scheduler.mode,
        "optimizer.scheduler.factor": cfg.optimizer.scheduler.factor,
        "optimizer.scheduler.patience": cfg.optimizer.scheduler.patience,
        "optimizer.scheduler.T_max": cfg.optimizer.scheduler.T_max,
        "optimizer.scheduler.eta_min": cfg.optimizer.scheduler.eta_min,
        "optimizer.scheduler.step_size": cfg.optimizer.scheduler.step_size,
        "optimizer.scheduler.gamma": cfg.optimizer.scheduler.gamma,
    }
    mlflow_logger.log_hyperparams(flat_cfg)
    
    # Log full Hydra config as artifact
    config_artifact_path = "hydra_config.yaml"
    with open(config_artifact_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, config_artifact_path)
    Path(config_artifact_path).unlink()  # Clean up temp file
    
    # Add custom tags if provided
    if cfg.tags:
        tags = list(cfg.tags)
        for tag in tags:
            mlflow_logger.experiment.set_tag(mlflow_logger.run_id, f"tag_{tag}", "true")
        mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "tags", ",".join(tags))
    
    # Dataset versioning and tracking
    print("\nCreating dataset metadata...")
    dataset_metadata = create_dataset_metadata(
        datamodule=datamodule,
        config=cfg,
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
                source=cfg.datamodule.dataset.root_dir,
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
        registered_model_name=cfg.model.name,
        artifact_path="best_model",
    )
    callbacks.append(log_best_callback)

    loggers = [tb_logger, mlflow_logger]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        logger=loggers,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=cfg.trainer.deterministic,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )
    
    # Resolve checkpoint path (local file or MLflow registered model)
    ckpt_path = cfg.ckpt
    if cfg.resume_from_model:
        if ckpt_path:
            raise ValueError("Cannot use both ckpt and resume_from_model at the same time.")
        ckpt_path = get_checkpoint_from_mlflow_model(
            cfg.resume_from_model,
            tracking_uri=cfg.trainer.logger.mlflow.tracking_uri,
        )

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
