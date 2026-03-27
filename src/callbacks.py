"""Custom PyTorch Lightning callbacks."""
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
import mlflow
from mlflow.models import infer_signature


class LogBestModelToMLflow(Callback):
    """Logs the best checkpointed model to MLflow with a signature."""

    def __init__(
        self,
        checkpoint_callback: ModelCheckpoint,
        example_input: torch.Tensor,
        mlflow_logger,
        registered_model_name: str = "cnn_classifier",
        artifact_path: str = "best_model",
    ) -> None:
        self.checkpoint_callback = checkpoint_callback
        self.example_input = example_input.detach().cpu()
        self.mlflow_logger = mlflow_logger
        self.registered_model_name = registered_model_name
        self.artifact_path = artifact_path
        self.logged = False

    def _log_best_model(self, trainer, pl_module) -> None:
        if self.logged:
            return
        best_path = self.checkpoint_callback.best_model_path
        if not best_path or not Path(best_path).is_file():
            rank_zero_info("LogBestModelToMLflow: Best checkpoint path not found.")
            return

        current_best = self.checkpoint_callback.best_model_score
        if current_best is None:
            rank_zero_info("LogBestModelToMLflow: No best model score available.")
            return

        model = pl_module.__class__.load_from_checkpoint(best_path)
        model.eval()
        model.to(pl_module.device)

        example_input = self.example_input.to(pl_module.device)
        with torch.no_grad():
            example_output = model(example_input)

        signature = infer_signature(
            example_input.cpu().numpy(),
            example_output.cpu().numpy(),
        )

        if self.mlflow_logger is None or self.mlflow_logger.run_id is None:
            rank_zero_info("LogBestModelToMLflow: MLflow logger/run not available.")
            return

        run_id = self.mlflow_logger.run_id
        active_run = mlflow.active_run()
        started_run = False
        if not active_run or active_run.info.run_id != run_id:
            mlflow.start_run(run_id=run_id)
            started_run = True

        try:
            mlflow.pytorch.log_model(
                model,
                artifact_path=self.artifact_path,
                signature=signature,
                input_example=example_input.cpu().numpy(),
                registered_model_name=self.registered_model_name,
            )
            self.logged = True
            rank_zero_info(
                f"LogBestModelToMLflow: Logged best model (score={current_best:.4f})."
            )
        finally:
            if started_run:
                mlflow.end_run()

    @rank_zero_only
    def on_train_end(self, trainer, pl_module) -> None:
        self._log_best_model(trainer, pl_module)

    @rank_zero_only
    def on_exception(self, trainer, pl_module, exception) -> None:
        self._log_best_model(trainer, pl_module)
