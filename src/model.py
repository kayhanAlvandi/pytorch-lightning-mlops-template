"""Simple CNN model with PyTorch Lightning."""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for threading
import matplotlib.pyplot as plt
import numpy as np
import io


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


class CNNClassifier(pl.LightningModule):
    """PyTorch Lightning module for CNN classification."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        dropout: float = 0.5,
        num_blocks: int = 4,
        base_channels: int = 32,
        channel_multiplier: float = 2.0,
        hidden_dim: int = 128,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SimpleCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
            num_blocks=num_blocks,
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            hidden_dim=hidden_dim,
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        # Store validation outputs for logging
        self.val_images = []
        self.val_preds = []
        self.val_labels = []
        
        # Gradient monitoring
        self.grad_alert_threshold = 1e5
        self._last_batch_meta: Dict[str, Any] = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        images, labels = batch
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
            self.log("grad/total_norm", total_norm, on_step=True, on_epoch=False)
            
            # Log first and last layer gradients
            for key in ["features.0.w", "classifier.4.w"]:
                if key in grad_norms:
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
        
        # Store first batch images for visualization
        if batch_idx == 0:
            self.val_images = images[:8].detach().cpu()  # Keep first 8 images
            self.val_preds = preds[:8].detach().cpu()
            self.val_labels = labels[:8].detach().cpu()
        
        return {"val_loss": loss, "preds": preds, "labels": labels}
    
    def on_validation_epoch_end(self):
        """Log confusion matrix and sample images at end of validation."""
        # Log confusion matrix
        confmat = self.val_confmat.compute().cpu().numpy()
        fig = self._plot_confusion_matrix(confmat)
        self.logger.experiment.add_figure("val/confusion_matrix", fig, self.current_epoch)
        plt.close(fig)
        self.val_confmat.reset()
        
        # Log sample images with predictions
        if len(self.val_images) > 0:
            fig = self._plot_predictions(self.val_images, self.val_preds, self.val_labels)
            self.logger.experiment.add_figure("val/predictions", fig, self.current_epoch)
            plt.close(fig)
    
    def _plot_confusion_matrix(self, confmat: np.ndarray) -> plt.Figure:
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
            title='Confusion Matrix'
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
    
    def _plot_predictions(self, images: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor) -> plt.Figure:
        """Create figure with sample images and their predictions."""
        n_images = min(8, len(images))
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
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
            
            pred_label = self.class_names[preds[i]]
            true_label = self.class_names[labels[i]]
            correct = preds[i] == labels[i]
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"P: {pred_label}\nT: {true_label}", 
                            color='green' if correct else 'red', fontsize=9)
            axes[i].axis('off')
        
        # Hide unused axes
        for i in range(n_images, 8):
            axes[i].axis('off')
        
        fig.suptitle(f'Validation Predictions (Epoch {self.current_epoch})', fontsize=12)
        fig.tight_layout()
        return fig
    
    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        
        self.log("test/loss", loss)
        
        return {"test_loss": loss, "preds": preds, "labels": labels}
    
    def configure_optimizers(self):
        # Create optimizer based on config
        optimizer_config = self._optimizer_config
        
        if optimizer_config.type == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_config.type == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_config.type == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=optimizer_config.momentum,
                nesterov=optimizer_config.nesterov,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_config.type}")
        
        # Create scheduler based on config
        scheduler_config = self._scheduler_config
        
        if scheduler_config.type == "None" or scheduler_config.type is None:
            return optimizer
        elif scheduler_config.type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.mode,
                factor=scheduler_config.factor,
                patience=scheduler_config.patience
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                },
            }
        elif scheduler_config.type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.T_max,
                eta_min=scheduler_config.eta_min
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif scheduler_config.type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_config.type}")
