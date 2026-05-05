"""Training script for CNN classifier."""
import json
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from src.dataset_versioning import (
    create_dataset_metadata,
    create_dataset_manifest,
    compute_dataset_version,
    get_git_commit_for_model_files,
    check_model_uncommitted_changes,
)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict into dot-separated keys for MLflow logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, (list, tuple)):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


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
    
    # Create data module via Hydra instantiate
    # _recursive_=False prevents Hydra from instantiating nested dataset/transform configs
    # (the datamodule does that itself in setup())
    datamodule = instantiate(cfg.datamodule, _recursive_=False)
    
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

    # Build unified model config by merging model, optimizer, scheduler, and loss
    # Hydra recursively instantiates:
    #   - criterion: fully (all params known)
    #   - optimizer: partially (_partial_=true, needs params=)
    #   - scheduler: partially (_partial_=true, needs optimizer=)
    # Disable struct mode to allow adding new keys (Hydra enables it by default)
    OmegaConf.set_struct(cfg.model, False)
    model_cfg = OmegaConf.merge(
        cfg.model,
        {
            "criterion": cfg.loss,
            "optimizer": cfg.optimizer.optimizer,
            "scheduler": cfg.optimizer.scheduler,
        },
    )
    
    model = instantiate(
        model_cfg,
        in_channels=len(cfg.datamodule.dataset.channels),
        num_classes=num_classes,
        class_names=list(datamodule.label_encoder.classes),
        _convert_="all",
    )
    
    # Pass batch-level transform (Mixup/CutMix) from datamodule to model
    if datamodule.batch_transform is not None:
        model.batch_transform = datamodule.batch_transform
        print(f"Batch transform: {datamodule.batch_transform.__class__.__name__}")
    
    # Setup callbacks — instantiate every entry in the callbacks list from config
    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks]
    
    # Setup dual logging: TensorBoard for visualization + MLflow for tracking
    tb_version = cfg.run_name if cfg.run_name else None
    
    tb_logger = instantiate(
        cfg.trainer.logger.tensorboard,
        version=tb_version,
    )
    
    mlflow_logger = instantiate(
        cfg.trainer.logger.mlflow,
        log_model=False,
        run_name=cfg.run_name,
    )
    
    # Enable system metrics logging (CPU, GPU, memory usage)
    mlflow.enable_system_metrics_logging()
    
    # Log model code version
    model_code_commit = get_git_commit_for_model_files()
    if check_model_uncommitted_changes():
        print("⚠ Warning: model.py has uncommitted changes - reproducibility not guaranteed!")
    
    # Log all config parameters to MLflow (auto-flatten the entire Hydra config)
    flat_cfg = _flatten_dict(OmegaConf.to_container(cfg, resolve=True))
    flat_cfg["model.code_commit"] = model_code_commit or "unknown"
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
                name=f"{cfg.dataset_name or 'microscopy_dataset'}_{dataset_metadata['dataset_version']}",
            )
            mlflow.log_input(dataset, context="training")
            print(f"✓ Dataset tracked with version: {dataset_metadata['dataset_version']}")
    except Exception as e:
        print(f"⚠ Warning: Could not log dataset with MLflow Dataset API: {e}")
        import traceback
        traceback.print_exc()
    
    loggers = [tb_logger, mlflow_logger]
    
    # Create trainer via Hydra instantiate
    # Exclude nested logger config (already instantiated separately)
    trainer_cfg = {k: v for k, v in OmegaConf.to_container(cfg.trainer, resolve=True).items()
                   if k != "logger"}
    trainer = instantiate(
        DictConfig(trainer_cfg),
        callbacks=callbacks,
        logger=loggers,
    )
    
    # Resolve checkpoint path (local file or MLflow registered model)
    ckpt_path = cfg.ckpt
    if cfg.resume_from_model:
        if ckpt_path:
            raise ValueError("Cannot use both ckpt and resume_from_model at the same time.")
        model_ckpt_path = get_checkpoint_from_mlflow_model(
            cfg.resume_from_model,
            tracking_uri=cfg.trainer.logger.mlflow.tracking_uri,
        )
        if cfg.resume_weights_only:
            # Load only model weights (fresh optimizer/scheduler/epoch)
            # Safe with gradual unfreezing — no param group mismatch
            print(f"\nLoading model weights only from: {model_ckpt_path}")
            checkpoint = torch.load(model_ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            print("  Weights loaded (optimizer/scheduler start fresh)")
        else:
            # Full resume (weights + optimizer + scheduler + epoch)
            # Requires matching param groups — use only if config is unchanged
            ckpt_path = model_ckpt_path

    # Train
    print("\nStarting training...")
    if ckpt_path:
        print(f"Resuming full training state from: {ckpt_path}")
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
