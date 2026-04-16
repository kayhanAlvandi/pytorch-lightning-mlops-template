"""Custom PyTorch Lightning callbacks."""
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
import mlflow
from mlflow.models import infer_signature


class LogBestModelToMLflow(Callback):
    """Logs the best checkpointed model to MLflow with a signature.
    
    Fully self-contained: resolves ModelCheckpoint, MLFlowLogger, and
    example input from the trainer at runtime. No special wiring needed
    in train.py — just add to the callbacks list via config.
    """

    def __init__(
        self,
        artifact_path: str = "best_model",
    ) -> None:
        self.artifact_path = artifact_path
        self.logged = False
        self._example_input: Optional[torch.Tensor] = None

    def _get_checkpoint_callback(self, trainer) -> Optional[ModelCheckpoint]:
        """Find ModelCheckpoint callback from trainer."""
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                return cb
        return None

    def _get_mlflow_logger(self, trainer) -> Optional[MLFlowLogger]:
        """Find MLFlowLogger from trainer."""
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                return logger
        return None

    @rank_zero_only
    def on_train_start(self, trainer, pl_module) -> None:
        """Capture a single example input from the val dataloader."""
        val_loader = trainer.datamodule.val_dataloader()
        try:
            batch = next(iter(val_loader))
        except StopIteration:
            rank_zero_info("LogBestModelToMLflow: Val dataloader empty, cannot capture example input.")
            return
        images, _ = batch
        self._example_input = images[:1].detach().cpu()

    def _get_registered_model_name(self, pl_module) -> str:
        """Derive registered model name from the module's class name."""
        return pl_module.__class__.__name__

    def _log_best_model(self, trainer, pl_module) -> None:
        if self.logged:
            return

        checkpoint_cb = self._get_checkpoint_callback(trainer)
        if checkpoint_cb is None:
            rank_zero_info("LogBestModelToMLflow: No ModelCheckpoint callback found.")
            return

        best_path = checkpoint_cb.best_model_path
        if not best_path or not Path(best_path).is_file():
            rank_zero_info("LogBestModelToMLflow: Best checkpoint path not found.")
            return

        current_best = checkpoint_cb.best_model_score
        if current_best is None:
            rank_zero_info("LogBestModelToMLflow: No best model score available.")
            return

        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None or mlflow_logger.run_id is None:
            rank_zero_info("LogBestModelToMLflow: MLflow logger/run not available.")
            return

        if self._example_input is None:
            rank_zero_info("LogBestModelToMLflow: No example input captured.")
            return

        model = pl_module.__class__.load_from_checkpoint(best_path)
        model.eval()
        model.to(pl_module.device)

        example_input = self._example_input.to(pl_module.device)
        with torch.no_grad():
            example_output = model(example_input)

        signature = infer_signature(
            example_input.cpu().numpy(),
            example_output.cpu().numpy(),
        )

        registered_model_name = self._get_registered_model_name(pl_module)

        run_id = mlflow_logger.run_id
        active_run = mlflow.active_run()
        started_run = False
        if not active_run or active_run.info.run_id != run_id:
            mlflow.start_run(run_id=run_id)
            started_run = True

        try:
            run_name = mlflow_logger.experiment.get_run(run_id).info.run_name or run_id[:8]
            # Sanitize run name for use as artifact path
            safe_run_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)
            artifact_path = f"{self.artifact_path}_{safe_run_name}"

            # Log raw checkpoint file first so it can be retrieved for training resumption
            mlflow.log_artifact(best_path, artifact_path="checkpoints")

            model_info = mlflow.pytorch.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=example_input.cpu().numpy(),
                registered_model_name=registered_model_name,
                code_paths=["src/model.py"]
            )
            
            # Add description and tags to the registered model version
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                
                # Get the version that was just registered
                model_version = model_info.registered_model_version
                
                # Build description from model hyperparams
                hp = pl_module.hparams
                num_blocks = hp.get('num_blocks', '?')
                base_channels = hp.get('base_channels', '?')
                channel_multiplier = hp.get('channel_multiplier', '?')
                hidden_dim = hp.get('hidden_dim', '?')
                description = (
                    f"Run: {run_name}\n"
                    f"Architecture: {num_blocks} blocks, {base_channels} base_ch, "
                    f"x{channel_multiplier} growth, {hidden_dim} hidden\n"
                    f"Val score: {current_best:.4f}\n"
                    f"in_channels={hp.get('in_channels','?')}, "
                    f"num_classes={hp.get('num_classes','?')}, "
                    f"dropout={hp.get('dropout','?')}"
                )
                client.update_model_version(
                    name=registered_model_name,
                    version=model_version,
                    description=description,
                )
                
                # Add searchable tags to the model version
                client.set_model_version_tag(registered_model_name, model_version, "run_name", run_name)
                client.set_model_version_tag(registered_model_name, model_version, "num_blocks", str(num_blocks))
                client.set_model_version_tag(registered_model_name, model_version, "base_channels", str(base_channels))
                client.set_model_version_tag(registered_model_name, model_version, "channel_multiplier", str(channel_multiplier))
                client.set_model_version_tag(registered_model_name, model_version, "hidden_dim", str(hidden_dim))
                client.set_model_version_tag(registered_model_name, model_version, "val_score", f"{current_best:.4f}")
            except Exception as tag_err:
                rank_zero_info(f"LogBestModelToMLflow: Could not set model version description/tags: {tag_err}")
            
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
