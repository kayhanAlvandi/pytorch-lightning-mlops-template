"""Simple CNN model with PyTorch Lightning."""
from functools import partial
from typing import Any, Dict, List, Optional

import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
import timm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for threading
import matplotlib.pyplot as plt
import numpy as np


class SimpleCNN(nn.Module):
    """Configurable CNN architecture for multi-channel image classification.
    
    Architecture:
        - N convolutional blocks with BatchNorm and MaxPool (configurable depth)
        - Global Average Pooling
        - Fully connected classifier (configurable width)
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        dropout: Dropout rate
        num_blocks: Number of conv blocks (depth) - 2, 3, 4, or 5
        base_channels: Starting number of channels (width) - 16, 32, or 64
        channel_multiplier: How much to multiply channels per block - 1.5, 2, or 3
        hidden_dim: Hidden layer dimension in classifier - 64, 128, or 256
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.5,
        num_blocks: int = 4,
        base_channels: int = 32,
        channel_multiplier: float = 2.0,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        # Build feature extractor dynamically
        layers = []
        current_channels = in_channels
        
        for i in range(num_blocks):
            out_channels = int(base_channels * (channel_multiplier ** i))
            
            # Conv block: 2 conv layers + BatchNorm + ReLU + MaxPool
            layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ])
            current_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.final_channels = current_channels
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.final_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TransferLearningBackbone(nn.Module):
    """Transfer learning backbone using timm pretrained models.
    
    Supports ResNet, EfficientNet, and Vision Transformer (ViT) architectures
    with stage-aware freezing/unfreezing, BatchNorm freezing, and
    discriminative learning rates.
    
    Stage layout (auto-detected from backbone):
        - Stage 0: stem (first conv + BN)
        - Stage 1..N: backbone blocks/layers
        - Head: custom classifier (always trainable)
    
    Args:
        backbone_name: Name of the backbone model
        in_channels: Number of input channels (will adapt first conv if != 3)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone weights initially
        freeze_bn: Whether to keep BatchNorm layers in eval mode (preserves pretrained stats)
        dropout: Dropout rate for classifier head
    """
    
    # Mapping of simple names to timm model names
    MODEL_MAPPING = {
        # ResNet variants
        'resnet18': 'resnet18',
        'resnet34': 'resnet34',
        'resnet50': 'resnet50',
        'resnet101': 'resnet101',
        # EfficientNet variants
        'efficientnet_b0': 'efficientnet_b0',
        'efficientnet_b1': 'efficientnet_b1',
        'efficientnet_b2': 'efficientnet_b2',
        'efficientnet_b3': 'efficientnet_b3',
        'efficientnet_b4': 'efficientnet_b4',
        # Vision Transformer variants
        'vit_tiny': 'vit_tiny_patch16_224',
        'vit_small': 'vit_small_patch16_224',
        'vit_base': 'vit_base_patch16_224',
        'vit_large': 'vit_large_patch16_224',
    }
    
    def __init__(
        self,
        backbone_name: str,
        in_channels: int,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_bn: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.freeze_backbone_flag = freeze_backbone
        self.freeze_bn = freeze_bn
        
        # Get timm model name
        timm_name = self.MODEL_MAPPING.get(backbone_name, backbone_name)
        
        # Create backbone without classifier head
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            in_chans=in_channels,
        )
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features
        
        # Discover stage structure from backbone
        self._stage_names, self._stage_params = self._discover_stages()
        
        # Create custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_all_backbone()
    
    def _discover_stages(self) -> tuple:
        """Auto-discover backbone stages from model structure.
        
        Groups parameters into stages:
          - EfficientNet: conv_stem/bn1 (stem), blocks.0..blocks.N (stages), conv_head/bn2 (neck)
          - ResNet: conv1/bn1 (stem), layer1..layer4 (stages)
          - ViT: patch_embed/cls_token/pos_embed (stem), blocks.0..blocks.N (stages), norm (neck)
        
        Returns:
            Tuple of (stage_names, stage_params) where stage_params maps
            stage_name -> list of (param_name, param) tuples.
        """
        stage_params = {}
        
        for name, param in self.backbone.named_parameters():
            # Determine stage from parameter name
            parts = name.split(".")
            
            # EfficientNet: blocks.0.*, blocks.1.*, etc.
            # ResNet: layer1.*, layer2.*, etc.
            # ViT: blocks.0.*, blocks.1.*, etc.
            if parts[0] == "blocks" and len(parts) > 1 and parts[1].isdigit():
                stage = f"blocks.{parts[1]}"
            elif parts[0].startswith("layer") and parts[0][5:].isdigit():
                stage = parts[0]  # ResNet layer1, layer2, etc.
            elif parts[0] in ("conv_stem", "bn1", "conv1", "bn1", 
                              "patch_embed", "cls_token", "pos_embed",
                              "pos_drop"):
                stage = "stem"
            elif parts[0] in ("conv_head", "bn2", "norm", "fc_norm",
                              "norm_pre"):
                stage = "neck"
            else:
                stage = "stem"
            
            if stage not in stage_params:
                stage_params[stage] = []
            stage_params[stage].append((name, param))
        
        # Order: stem first, then numbered stages, then neck
        ordered_names = []
        if "stem" in stage_params:
            ordered_names.append("stem")
        
        # Sort numbered stages (blocks.0, blocks.1, ... or layer1, layer2, ...)
        numbered = [s for s in stage_params if s not in ("stem", "neck")]
        numbered.sort(key=lambda s: int(s.split(".")[-1]) if "." in s 
                      else int(s.replace("layer", "")))
        ordered_names.extend(numbered)
        
        if "neck" in stage_params:
            ordered_names.append("neck")
        
        return ordered_names, stage_params
    
    def get_stage_names(self) -> List[str]:
        """Return ordered list of backbone stage names (stem -> deep -> neck)."""
        return list(self._stage_names)
    
    def get_num_stages(self) -> int:
        """Return number of backbone stages."""
        return len(self._stage_names)
    
    def freeze_all_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Backbone '{self.backbone_name}' frozen ({self.get_num_stages()} stages)")
    
    def unfreeze_stages(self, stage_indices: List[int]):
        """Unfreeze specific backbone stages by index.
        
        Args:
            stage_indices: List of stage indices to unfreeze (0=stem, -1=last stage).
                          Negative indices are supported.
        """
        n = self.get_num_stages()
        resolved = [i % n for i in stage_indices]
        unfrozen_names = []
        
        for idx in resolved:
            stage_name = self._stage_names[idx]
            unfrozen_names.append(stage_name)
            for _, param in self._stage_params[stage_name]:
                param.requires_grad = True
        
        print(f"Unfroze stages: {unfrozen_names}")
    
    def unfreeze_from_stage(self, stage_idx: int):
        """Unfreeze from a given stage index to the end (deep layers first).
        
        E.g., unfreeze_from_stage(-3) unfreezes the last 3 stages + neck.
        
        Args:
            stage_idx: Stage index to start unfreezing from.
                      Negative indices count from the end.
        """
        n = self.get_num_stages()
        resolved = stage_idx % n
        stages_to_unfreeze = list(range(resolved, n))
        self.unfreeze_stages(stages_to_unfreeze)
    
    def get_param_groups(self, head_lr: float, backbone_lr: float,
                         lr_decay_factor: float = 1.0) -> List[dict]:
        """Create parameter groups with discriminative learning rates.
        
        LR decreases from deep layers to early layers using lr_decay_factor.
        If lr_decay_factor=1.0, all backbone stages share the same backbone_lr.
        
        Example with 8 stages, backbone_lr=1e-4, lr_decay_factor=0.8:
            neck:     1e-4
            blocks.6: 1e-4 * 0.8^1 = 8e-5
            blocks.5: 1e-4 * 0.8^2 = 6.4e-5
            ...
            stem:     1e-4 * 0.8^8 = ~1.7e-5
            head:     head_lr (always separate)
        
        Args:
            head_lr: Learning rate for the classifier head
            backbone_lr: Base learning rate for backbone (applied to deepest stages)
            lr_decay_factor: Multiplier applied per stage going from deep to shallow.
                            1.0 = uniform backbone LR, <1.0 = lower LR for early layers.
        
        Returns:
            List of param group dicts for the optimizer.
        """
        param_groups = []
        
        # Backbone stages: reversed so deepest stages get highest LR
        reversed_stages = list(reversed(self._stage_names))
        for depth, stage_name in enumerate(reversed_stages):
            trainable = [(n, p) for n, p in self._stage_params[stage_name] 
                        if p.requires_grad]
            if trainable:
                stage_lr = backbone_lr * (lr_decay_factor ** depth)
                param_groups.append({
                    "params": [p for _, p in trainable],
                    "lr": stage_lr,
                    "name": f"backbone_{stage_name}",
                })
        
        # Classifier head
        param_groups.append({
            "params": list(self.classifier.parameters()),
            "lr": head_lr,
            "name": "head",
        })
        
        return param_groups
    
    def train(self, mode: bool = True):
        """Override train() to optionally keep BN layers in eval mode."""
        super().train(mode)
        if mode and self.freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    module.eval()
        return self
    
    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Return count of total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class BaseClassifier(pl.LightningModule):
    """Base PyTorch Lightning module for image classification.
    
    Provides shared training, validation, and logging logic.
    Subclasses must set self.model in their __init__.
    
    Accepts either:
      - Live objects from Hydra (during training):
            criterion=nn.CrossEntropyLoss(), optimizer=partial(AdamW, lr=0.001), ...
      - Config dicts from checkpoint (during load_from_checkpoint):
            criterion_config={"_target_": "torch.nn.CrossEntropyLoss"}, ...
    """
    
    def __init__(
        self,
        num_classes: int,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[partial] = None,
        scheduler: Optional[partial] = None,
        criterion_config: Optional[dict] = None,
        optimizer_config: Optional[dict] = None,
        scheduler_config: Optional[dict] = None,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        
        # Resolve criterion, optimizer, scheduler from either live objects or config dicts
        self.criterion, self._optimizer_partial, self._scheduler_partial = \
            self._resolve_components(criterion, optimizer, scheduler,
                                     criterion_config, optimizer_config, scheduler_config)
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        # Store validation outputs for logging (2 images per class)
        self._val_class_images: Dict[int, List] = {}  # class_idx -> [(image, pred, label), ...]
        self._n_images_per_class = 2
        
        # Store test outputs for logging
        self._test_class_images: Dict[int, List] = {}
        
        # Gradient monitoring
        self.grad_alert_threshold = 1e5
        self._last_batch_meta: Dict[str, Any] = {}
        
        # Batch-level transform (Mixup/CutMix) — set externally by train.py
        self.batch_transform: Optional[nn.Module] = None
    
    @staticmethod
    def _import_class(dotpath: str):
        """Import a class from a dot-separated module path (e.g., 'torch.optim.AdamW')."""
        module_path, class_name = dotpath.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    @staticmethod
    def _partial_to_config(p: partial) -> dict:
        """Convert a functools.partial to a serializable config dict."""
        if p is None:
            return None
        return {
            "_target_": f"{p.func.__module__}.{p.func.__qualname__}",
            **{k: v for k, v in p.keywords.items()},
        }
    
    @staticmethod
    def _criterion_to_config(criterion: nn.Module) -> dict:
        """Convert an instantiated criterion to a serializable config dict."""
        import inspect
        cls = type(criterion)
        config = {"_target_": f"{cls.__module__}.{cls.__qualname__}"}
        # Get actual constructor params via inspect, not __dict__ (which has runtime state)
        sig = inspect.signature(cls.__init__)
        for param_name in sig.parameters:
            if param_name == "self":
                continue
            if hasattr(criterion, param_name):
                val = getattr(criterion, param_name)
                if not isinstance(val, (torch.Tensor, nn.Module)):
                    config[param_name] = val
        return config
    
    def _resolve_components(self, criterion, optimizer, scheduler,
                            criterion_config, optimizer_config, scheduler_config):
        """Resolve criterion/optimizer/scheduler from live objects or config dicts.
        
        During training: live objects are passed (from Hydra instantiation).
        During checkpoint loading: config dicts are passed (from saved hparams).
        """
        # Resolve criterion
        if criterion is not None:
            resolved_criterion = criterion
        elif criterion_config is not None:
            cfg = dict(criterion_config)
            cls = self._import_class(cfg.pop("_target_"))
            resolved_criterion = cls(**cfg)
        else:
            resolved_criterion = nn.CrossEntropyLoss()
        
        # Resolve optimizer
        if optimizer is not None:
            resolved_optimizer = optimizer
        elif optimizer_config is not None:
            cfg = dict(optimizer_config)
            cls = self._import_class(cfg.pop("_target_"))
            resolved_optimizer = partial(cls, **cfg)
        else:
            resolved_optimizer = partial(torch.optim.AdamW, lr=1e-3)
        
        # Resolve scheduler
        if scheduler is not None:
            resolved_scheduler = scheduler
        elif scheduler_config is not None:
            cfg = dict(scheduler_config)
            cls = self._import_class(cfg.pop("_target_"))
            resolved_scheduler = partial(cls, **cfg)
        else:
            resolved_scheduler = None
        
        return resolved_criterion, resolved_optimizer, resolved_scheduler
    
    def _save_config_hparams(self, criterion, optimizer, scheduler):
        """Store serializable config dicts in hparams for checkpoint serialization."""
        self.hparams["criterion_config"] = self._criterion_to_config(criterion)
        self.hparams["optimizer_config"] = self._partial_to_config(optimizer)
        self.hparams["scheduler_config"] = self._partial_to_config(scheduler)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        images, labels = batch
        
        # Apply batch-level transform (Mixup/CutMix) if configured
        # Uses torchvision.transforms.v2 — returns (images, soft_labels)
        if self.batch_transform is not None and self.training:
            images, soft_labels = self.batch_transform(images, labels)
            logits = self(images)
            # Soft labels (B, num_classes): use soft cross-entropy
            soft_labels = soft_labels.to(dtype=logits.dtype, device=logits.device)
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            loss = -(soft_labels * log_probs).sum(dim=1).mean()
        else:
            logits = self(images)
            loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Cache metadata for gradient spike diagnostics
        label_counts = torch.bincount(labels.detach().cpu(), minlength=self.num_classes)
        current_lr = None
        if self.trainer is not None and getattr(self.trainer, "optimizers", None):
            if self.trainer.optimizers:
                current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self._last_batch_meta = {
            "batch_idx": batch_idx,
            "loss": loss.detach().item(),
            "label_counts": label_counts.tolist(),
            "lr": current_lr,
        }
        
        return loss
    
    def on_after_backward(self):
        """Log gradient statistics after backward pass."""
        if self.global_step % 50 == 0:  # Log every 50 steps
            total_norm = 0.0
            grad_norms = {}
            
            for name, param in self.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    # Log per-layer gradients (simplified names)
                    short_name = name.replace("model.", "").replace(".weight", ".w").replace(".bias", ".b")
                    grad_norms[short_name] = param_norm
            
            total_norm = total_norm ** 0.5
            
            # Only log finite values to avoid MLflow UI errors
            if math.isfinite(total_norm):
                self.log("grad/total_norm", total_norm, on_step=True, on_epoch=False)
            
            # Log first and last layer gradients
            for key in ["features.0.w", "classifier.4.w"]:
                if key in grad_norms and math.isfinite(grad_norms[key]):
                    self.log(f"grad/{key}", grad_norms[key], on_step=True, on_epoch=False)
            
            if total_norm > self.grad_alert_threshold:
                meta = getattr(self, "_last_batch_meta", {})
                alert_msg = (
                    f"Grad spike | step={self.global_step} norm={total_norm:.2e} "
                    f"batch={meta.get('batch_idx')} loss={meta.get('loss')} "
                    f"lr={meta.get('lr')} labels={meta.get('label_counts')}"
                )
                self.print(alert_msg)
                if self.logger is not None and hasattr(self.logger, "experiment"):
                    exp = self.logger.experiment
                    if hasattr(exp, "add_text"):
                        exp.add_text("grad/alerts", alert_msg, self.global_step)
                    if hasattr(exp, "add_scalar"):
                        exp.add_scalar("grad/alert_total_norm", total_norm, self.global_step)
    
    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_confmat(preds, labels)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        # Collect 2 images per class for balanced visualization
        for i in range(len(labels)):
            cls = labels[i].item()
            if cls not in self._val_class_images:
                self._val_class_images[cls] = []
            if len(self._val_class_images[cls]) < self._n_images_per_class:
                self._val_class_images[cls].append((
                    images[i].detach().cpu(),
                    preds[i].detach().cpu(),
                    labels[i].detach().cpu(),
                ))
        
        return {"val_loss": loss, "preds": preds, "labels": labels}
    
    def _collect_class_balanced_samples(self, class_images_dict):
        """Flatten class-balanced image dict into (images, preds, labels) tensors."""
        images, preds, labels = [], [], []
        for cls in sorted(class_images_dict.keys()):
            for img, pred, label in class_images_dict[cls]:
                images.append(img)
                preds.append(pred)
                labels.append(label)
        if not images:
            return None, None, None
        return torch.stack(images), torch.stack(preds), torch.stack(labels)
    
    def on_validation_epoch_end(self):
        """Log confusion matrix and sample images to TensorBoard only."""
        # Log confusion matrix to TensorBoard
        confmat = self.val_confmat.compute().cpu().numpy()
        fig_cm = self._plot_confusion_matrix(confmat)
        
        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'add_figure'):
            self.logger.experiment.add_figure("val/confusion_matrix", fig_cm, self.current_epoch)
        plt.close(fig_cm)
        self.val_confmat.reset()
        
        # Log sample images with predictions to TensorBoard (2 per class)
        images, preds, labels = self._collect_class_balanced_samples(self._val_class_images)
        if images is not None:
            fig_pred = self._plot_predictions(images, preds, labels)
            
            if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'add_figure'):
                self.logger.experiment.add_figure("val/predictions", fig_pred, self.current_epoch)
            plt.close(fig_pred)
        
        # Reset for next epoch
        self._val_class_images = {}
    
    def _log_figure_to_mlflow(self, fig: plt.Figure, filename: str) -> None:
        """Log a matplotlib figure to MLflow if MLflow logger is available."""
        try:
            import tempfile
            import os
            from pytorch_lightning.loggers import MLFlowLogger
            
            # Find MLflow logger from trainer's loggers
            mlflow_logger = None
            if self.trainer and self.trainer.loggers:
                for logger in self.trainer.loggers:
                    if isinstance(logger, MLFlowLogger):
                        mlflow_logger = logger
                        break
            
            if mlflow_logger:
                # Save figure to temp file and log as artifact
                with tempfile.TemporaryDirectory() as tmpdir:
                    filepath = os.path.join(tmpdir, filename)
                    fig.savefig(filepath, dpi=100, bbox_inches='tight')
                    mlflow_logger.experiment.log_artifact(
                        mlflow_logger.run_id, 
                        filepath, 
                        artifact_path="figures"
                    )
        except Exception:
            pass  # Silently skip if MLflow logging fails
    
    def _plot_confusion_matrix(self, confmat: np.ndarray, title: str = "Confusion Matrix") -> plt.Figure:
        """Create confusion matrix figure."""
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(
            xticks=np.arange(confmat.shape[1]),
            yticks=np.arange(confmat.shape[0]),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ylabel='True label',
            xlabel='Predicted label',
            title=title
        )
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = confmat.max() / 2.
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(j, i, format(int(confmat[i, j]), 'd'),
                       ha="center", va="center",
                       color="white" if confmat[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig
    
    def _plot_predictions(self, images: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor, title: str = None) -> plt.Figure:
        """Create figure with sample images and their predictions.
        
        Grid layout adapts to number of images (2 per class).
        """
        n_images = len(images)
        n_cols = min(n_images, self.num_classes)  # One column per class
        n_rows = max(1, (n_images + n_cols - 1) // n_cols)  # Enough rows
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes).flatten()
        
        for i in range(n_images):
            img = images[i]
            # Use first 3 channels for RGB visualization, or repeat if fewer
            if img.shape[0] >= 3:
                img_rgb = img[:3].permute(1, 2, 0).numpy()
            else:
                img_rgb = img[0].numpy()
                img_rgb = np.stack([img_rgb] * 3, axis=-1)
            
            # Normalize to [0, 1]
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
            
            pred_label = self.class_names[preds[i].item()]
            true_label = self.class_names[labels[i].item()]
            correct = preds[i] == labels[i]
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"P: {pred_label}\nT: {true_label}", 
                            color='green' if correct else 'red', fontsize=9)
            axes[i].axis('off')
        
        # Hide unused axes
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        if title is None:
            title = f'Predictions (Epoch {self.current_epoch})'
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        return fig
    
    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_confmat(preds, labels)
        
        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        
        # Collect 2 images per class for balanced visualization
        for i in range(len(labels)):
            cls = labels[i].item()
            if cls not in self._test_class_images:
                self._test_class_images[cls] = []
            if len(self._test_class_images[cls]) < self._n_images_per_class:
                self._test_class_images[cls].append((
                    images[i].detach().cpu(),
                    preds[i].detach().cpu(),
                    labels[i].detach().cpu(),
                ))
        
        return {"test_loss": loss, "preds": preds, "labels": labels}
    
    def on_test_epoch_end(self):
        """Log confusion matrix and prediction images to MLflow (best model)."""
        # Confusion matrix
        confmat = self.test_confmat.compute().cpu().numpy()
        fig_cm = self._plot_confusion_matrix(confmat, title="Test Confusion Matrix (Best Model)")
        self._log_figure_to_mlflow(fig_cm, "test_confusion_matrix.png")
        plt.close(fig_cm)
        self.test_confmat.reset()
        
        # Prediction images (2 per class)
        images, preds, labels = self._collect_class_balanced_samples(self._test_class_images)
        if images is not None:
            fig_pred = self._plot_predictions(images, preds, labels, title="Test Predictions (Best Model)")
            self._log_figure_to_mlflow(fig_pred, "test_predictions.png")
            plt.close(fig_pred)
        
        self._test_class_images = {}
    
    def _create_scheduler(self, optimizer):
        """Create scheduler from partial, auto-resolving epoch-dependent params.
        
        Auto-resolves from trainer.max_epochs:
          - T_max (CosineAnnealingLR)
          - total_iters (LinearLR, PolynomialLR)
        
        Supports optional LR warmup via extra keyword 'warmup_fraction':
          - warmup_fraction: fraction of total epochs for linear warmup (e.g., 0.1 = 10%)
          - During warmup, LR ramps from ~0 to the target LR
          - After warmup, the main scheduler takes over for the remaining epochs
          - Uses SequentialLR to chain warmup + main scheduler
        """
        if self._scheduler_partial is None:
            return None
        
        # Auto-resolve epoch-dependent scheduler params from trainer
        max_epochs = getattr(self.trainer, 'max_epochs', None) if self.trainer else None
        
        # Extract warmup_fraction (not a real scheduler param, we handle it ourselves)
        keywords = dict(self._scheduler_partial.keywords)
        warmup_fraction = keywords.pop('warmup_fraction', 0.0)
        
        if max_epochs is not None:
            warmup_epochs = int(max_epochs * warmup_fraction)
            remaining_epochs = max_epochs - warmup_epochs
            
            # CosineAnnealingLR: T_max = remaining epochs after warmup
            if 'T_max' in keywords:
                keywords['T_max'] = remaining_epochs
            # LinearLR / PolynomialLR: total_iters = remaining epochs after warmup
            if 'total_iters' in keywords:
                keywords['total_iters'] = remaining_epochs
        else:
            warmup_epochs = 0
        
        # Create the main scheduler
        main_scheduler = self._scheduler_partial.func(optimizer=optimizer, **keywords)
        
        # If no warmup, return main scheduler directly
        if warmup_epochs <= 0:
            return main_scheduler
        
        # Create warmup scheduler: ramp from ~0 to target LR
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,   # Start at 1% of target LR
            end_factor=1.0,      # Ramp to 100% of target LR
            total_iters=warmup_epochs,
        )
        
        # Chain: warmup -> main
        combined = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
        
        print(f"  LR schedule: warmup {warmup_epochs} epochs -> main decay {remaining_epochs} epochs")
        
        return combined
    
    def configure_optimizers(self):
        # Complete the partial optimizer with model parameters
        optimizer = self._optimizer_partial(params=self.parameters())
        
        # Complete the partial scheduler with optimizer, if configured
        scheduler = self._create_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        
        # ReduceLROnPlateau needs a monitor key
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
        }
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler_config["monitor"] = "val/loss"
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }


class CNNClassifier(BaseClassifier):
    """PyTorch Lightning module for SimpleCNN classification."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[partial] = None,
        scheduler: Optional[partial] = None,
        criterion_config: Optional[dict] = None,
        optimizer_config: Optional[dict] = None,
        scheduler_config: Optional[dict] = None,
        dropout: float = 0.5,
        num_blocks: int = 4,
        base_channels: int = 32,
        channel_multiplier: float = 2.0,
        hidden_dim: int = 128,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion_config=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            class_names=class_names,
        )
        self.save_hyperparameters(ignore=["criterion", "optimizer", "scheduler"])
        self._save_config_hparams(
            self.criterion, self._optimizer_partial, self._scheduler_partial
        )
        
        self.model = SimpleCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
            num_blocks=num_blocks,
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            hidden_dim=hidden_dim,
        )


class TransferLearningClassifier(BaseClassifier):
    """PyTorch Lightning module for transfer learning classification.
    
    Supports ResNet, EfficientNet, and Vision Transformer (ViT) architectures
    with fine-tuning options:
    
    - **Gradual unfreezing**: Controlled by GradualUnfreezing callback via unfreeze schedule
    - **Discriminative learning rates**: Lower LR for early layers, higher for deep/head
    - **Freeze BatchNorm**: Keep BN running stats from pretrained model
    
    Fine-tuning config params:
        head_lr: Learning rate for classifier head (default: uses optimizer LR)
        backbone_lr: Learning rate for backbone layers (default: head_lr / 10)
        lr_decay_factor: Per-stage LR decay from deep->shallow (default: 1.0 = uniform)
        freeze_bn: Keep BatchNorm in eval mode during training
        freeze_backbone: Start with backbone fully frozen (for warmup phase)
    """
    
    def __init__(
        self,
        backbone_name: str,
        in_channels: int,
        num_classes: int,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[partial] = None,
        scheduler: Optional[partial] = None,
        criterion_config: Optional[dict] = None,
        optimizer_config: Optional[dict] = None,
        scheduler_config: Optional[dict] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_bn: bool = False,
        dropout: float = 0.5,
        head_lr: Optional[float] = None,
        backbone_lr: Optional[float] = None,
        lr_decay_factor: float = 1.0,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion_config=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            class_names=class_names,
        )
        self.save_hyperparameters(ignore=["criterion", "optimizer", "scheduler"])
        self._save_config_hparams(
            self.criterion, self._optimizer_partial, self._scheduler_partial
        )
        
        # Store fine-tuning LR config
        self._head_lr = head_lr
        self._backbone_lr = backbone_lr
        self._lr_decay_factor = lr_decay_factor
        
        self.model = TransferLearningBackbone(
            backbone_name=backbone_name,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            freeze_bn=freeze_bn,
            dropout=dropout,
        )
        
        # Log parameter counts and stage info
        total_params = self.model.get_total_params()
        trainable_params = self.model.get_trainable_params()
        print(f"Model: {backbone_name} | Total params: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"Backbone stages: {self.model.get_stage_names()}")
        if freeze_bn:
            print("BatchNorm layers frozen in eval mode (pretrained stats preserved)")
    
    def configure_optimizers(self):
        """Create optimizer with discriminative learning rates if configured.
        
        If head_lr or backbone_lr is set, creates separate param groups:
          - Each backbone stage gets its own LR (decayed by lr_decay_factor from deep->shallow)
          - Classifier head gets head_lr
        Otherwise, falls back to standard single-LR optimizer from base class.
        """
        use_discriminative_lr = (self._head_lr is not None or self._backbone_lr is not None)
        
        if use_discriminative_lr:
            # Determine LRs: get base LR from optimizer partial if not explicitly set
            base_lr = self._optimizer_partial.keywords.get("lr", 1e-3)
            head_lr = self._head_lr if self._head_lr is not None else base_lr
            backbone_lr = self._backbone_lr if self._backbone_lr is not None else head_lr / 10.0
            
            # Get param groups from backbone
            param_groups = self.model.get_param_groups(
                head_lr=head_lr,
                backbone_lr=backbone_lr,
                lr_decay_factor=self._lr_decay_factor,
            )
            
            # Log LR per group
            for pg in param_groups:
                print(f"  Param group '{pg['name']}': lr={pg['lr']:.2e}, params={sum(p.numel() for p in pg['params']):,}")
            
            # Create optimizer with param groups (strip 'lr' from partial keywords
            # since each group has its own)
            opt_kwargs = {k: v for k, v in self._optimizer_partial.keywords.items() if k != "lr"}
            optimizer = self._optimizer_partial.func(param_groups, **opt_kwargs)
            
            # Scheduler (auto-resolves T_max/total_iters from trainer)
            scheduler = self._create_scheduler(optimizer)
            if scheduler is None:
                return optimizer
            
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
            }
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler_config["monitor"] = "val/loss"
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
        else:
            # Standard single-LR optimizer from base class
            return super().configure_optimizers()
