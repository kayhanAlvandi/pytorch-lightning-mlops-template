"""Integration test: training with an MLFlowLogger records a run, params and metrics.

Uses a temporary sqlite tracking DB and temp artifact dir so the real
``mlflow.db`` / ``mlruns`` are never touched.
"""
import pytest
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from src.model import CNNClassifier

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_mlflow_run_records_params_and_metrics(tiled_datamodule, tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_dir = (tmp_path / "artifacts").as_uri()
    experiment_name = "unit-test-experiment"

    dm = tiled_datamodule
    dm.setup()
    model = CNNClassifier(
        in_channels=len(dm.channels),
        num_classes=dm.label_encoder.num_classes,
        num_blocks=2,
        base_channels=8,
        hidden_dim=16,
    )

    logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name="smoke-run",
        artifact_location=artifact_dir,
    )
    logger.log_hyperparams({
        "num_classes": dm.label_encoder.num_classes,
        "learning_rate": 1e-3,
    })

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_train_batches=2,
        limit_val_batches=2,
    )
    trainer.fit(model, dm)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) >= 1

    run = runs[0]
    # Logged hyperparameter is present
    assert run.data.params.get("num_classes") == str(dm.label_encoder.num_classes)
    # At least one training metric was recorded during fit
    assert any(k.startswith("train") for k in run.data.metrics)
