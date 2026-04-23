"""Custom PyTorch Lightning callbacks."""
from pathlib import Path
from typing import Dict, List, Optional

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
                
                # Detect model type and build appropriate description
                backbone_name = hp.get('backbone_name', None)
                if backbone_name:
                    # Transfer learning model
                    pretrained = hp.get('pretrained', True)
                    freeze_backbone = hp.get('freeze_backbone', False)
                    description = (
                        f"Run: {run_name}\n"
                        f"Backbone: {backbone_name} (pretrained={pretrained}, frozen={freeze_backbone})\n"
                        f"Val score: {current_best:.4f}\n"
                        f"in_channels={hp.get('in_channels','?')}, "
                        f"num_classes={hp.get('num_classes','?')}, "
                        f"dropout={hp.get('dropout','?')}"
                    )
                else:
                    # SimpleCNN model
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
                client.set_model_version_tag(registered_model_name, model_version, "val_score", f"{current_best:.4f}")
                
                if backbone_name:
                    # Transfer learning tags
                    client.set_model_version_tag(registered_model_name, model_version, "backbone_name", backbone_name)
                    client.set_model_version_tag(registered_model_name, model_version, "pretrained", str(hp.get('pretrained', True)))
                    client.set_model_version_tag(registered_model_name, model_version, "freeze_backbone", str(hp.get('freeze_backbone', False)))
                else:
                    # SimpleCNN tags
                    client.set_model_version_tag(registered_model_name, model_version, "num_blocks", str(hp.get('num_blocks', '?')))
                    client.set_model_version_tag(registered_model_name, model_version, "base_channels", str(hp.get('base_channels', '?')))
                    client.set_model_version_tag(registered_model_name, model_version, "channel_multiplier", str(hp.get('channel_multiplier', '?')))
                    client.set_model_version_tag(registered_model_name, model_version, "hidden_dim", str(hp.get('hidden_dim', '?')))
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


class EarlyStoppingWithWarmup(Callback):
    """Early stopping callback that only starts monitoring after a warmup period.
    
    Unlike the standard EarlyStopping with trainer.min_epochs, this callback
    does NOT accumulate patience during the warmup period. Patience counting
    only begins after min_epochs is reached.
    
    Args:
        monitor: Metric to monitor (e.g., 'val/loss')
        min_epochs: Number of epochs before early stopping can activate
        patience: Number of checks with no improvement after min_epochs before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' or 'max' - whether to minimize or maximize the metric
    """
    
    def __init__(
        self,
        monitor: str = "val/loss",
        min_epochs: int = 30,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        super().__init__()
        self.monitor = monitor
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_score = None
        self.wait_count = 0
        
        if mode == "min":
            self.monitor_op = lambda current, best: current < best - min_delta
        elif mode == "max":
            self.monitor_op = lambda current, best: current > best + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def on_validation_end(self, trainer, pl_module) -> None:
        current_epoch = trainer.current_epoch
        
        # Don't do anything during warmup period
        if current_epoch < self.min_epochs:
            return
        
        # Get current metric value
        logs = trainer.callback_metrics
        current = logs.get(self.monitor)
        
        if current is None:
            rank_zero_info(f"EarlyStoppingWithWarmup: {self.monitor} not found in metrics")
            return
        
        current = current.item() if hasattr(current, 'item') else float(current)
        
        # Initialize best score on first check after warmup
        if self.best_score is None:
            self.best_score = current
            rank_zero_info(
                f"EarlyStoppingWithWarmup: Started monitoring at epoch {current_epoch}, "
                f"initial {self.monitor}={current:.4f}"
            )
            return
        
        # Check for improvement
        if self.monitor_op(current, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            rank_zero_info(
                f"EarlyStoppingWithWarmup: No improvement for {self.wait_count}/{self.patience} epochs "
                f"(best={self.best_score:.4f}, current={current:.4f})"
            )
            
            if self.wait_count >= self.patience:
                rank_zero_info(
                    f"EarlyStoppingWithWarmup: Stopping training at epoch {current_epoch}"
                )
                trainer.should_stop = True


class GradualUnfreezing(Callback):
    """Gradually unfreeze backbone stages during training.
    
    Unfreezes backbone stages from deep to shallow according to a schedule.
    After unfreezing, rebuilds the optimizer with proper param groups so
    newly unfrozen parameters get discriminative learning rates.
    
    Schedule format: list of [epoch, num_stages_from_end] pairs.
    
    Example schedule for EfficientNet-B3 (9 stages: stem, blocks.0-6, neck):
        unfreeze_schedule:
          - [0,  0]     # Epochs 0-4:  only head (warmup)
          - [5,  2]     # Epochs 5-9:  unfreeze last 2 stages (neck + blocks.6)
          - [10, 5]     # Epochs 10-19: unfreeze last 5 stages
          - [20, -1]    # Epoch 20+:   unfreeze all (-1 = all)
    
    Args:
        unfreeze_schedule: List of [epoch, num_stages_from_end] pairs.
                          num_stages_from_end=-1 means unfreeze all stages.
                          Must be sorted by epoch ascending.
    """
    
    def __init__(self, unfreeze_schedule: List[List[int]]):
        super().__init__()
        # Sort by epoch to be safe
        self.schedule = sorted(unfreeze_schedule, key=lambda x: x[0])
        self._last_applied_idx = -1
    
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        current_epoch = trainer.current_epoch
        
        # Find the latest schedule entry that should be active
        active_idx = -1
        for i, (epoch, _) in enumerate(self.schedule):
            if current_epoch >= epoch:
                active_idx = i
        
        # Skip if no change from last applied
        if active_idx == self._last_applied_idx or active_idx < 0:
            return
        
        self._last_applied_idx = active_idx
        _, num_stages = self.schedule[active_idx]
        
        # Get the backbone model
        if not hasattr(pl_module, 'model') or not hasattr(pl_module.model, 'unfreeze_from_stage'):
            rank_zero_info("GradualUnfreezing: Model does not support stage-based unfreezing, skipping")
            return
        
        backbone = pl_module.model
        
        if num_stages == 0:
            # Freeze all backbone (warmup phase - only head trains)
            backbone.freeze_all_backbone()
            rank_zero_info(f"GradualUnfreezing: Epoch {current_epoch} - backbone frozen (head warmup)")
        elif num_stages == -1:
            # Unfreeze everything
            for param in backbone.backbone.parameters():
                param.requires_grad = True
            rank_zero_info(f"GradualUnfreezing: Epoch {current_epoch} - all backbone stages unfrozen")
        else:
            # Freeze all first, then unfreeze last N stages
            backbone.freeze_all_backbone()
            n_stages = backbone.get_num_stages()
            start_idx = max(0, n_stages - num_stages)
            backbone.unfreeze_from_stage(start_idx)
            rank_zero_info(
                f"GradualUnfreezing: Epoch {current_epoch} - unfroze last {num_stages} of {n_stages} stages"
            )
        
        # Rebuild optimizer with updated param groups
        self._rebuild_optimizer(trainer, pl_module)
        
        # Log trainable param count
        trainable = backbone.get_trainable_params()
        total = backbone.get_total_params()
        rank_zero_info(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def _rebuild_optimizer(self, trainer, pl_module):
        """Rebuild optimizer with updated param groups, preserving scheduler state.
        
        Instead of using setup_optimizers() (which resets the scheduler),
        we create a new optimizer and patch it into the existing scheduler
        so the LR schedule continues without jumping back up.
        """
        # Save old scheduler state before rebuild
        old_scheduler_states = []
        if trainer.lr_scheduler_configs:
            for config in trainer.lr_scheduler_configs:
                old_scheduler_states.append(config.scheduler.state_dict())
        
        # Rebuild optimizer + scheduler from configure_optimizers
        trainer.strategy.setup_optimizers(trainer)
        
        # Restore scheduler state so LR continues from where it was
        if old_scheduler_states and trainer.lr_scheduler_configs:
            for config, old_state in zip(trainer.lr_scheduler_configs, old_scheduler_states):
                config.scheduler.load_state_dict(old_state)
                # Point scheduler to the new optimizer
                config.scheduler.optimizer = trainer.optimizers[0]
            rank_zero_info("  Optimizer rebuilt with updated param groups (scheduler state preserved)")
        else:
            rank_zero_info("  Optimizer rebuilt with updated param groups")
