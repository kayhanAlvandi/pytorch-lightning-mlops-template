"""Model loading and prediction logic for the API.

Supports loading models from:
  1. MLflow registered model: e.g. "TransferLearning/20" or "TransferLearning/latest"
  2. MLflow run name: e.g. "Vits_finetune_cosine_warmup_..."
  3. Direct checkpoint path (fallback)
"""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import mlflow
import mlflow.pytorch
from omegaconf import OmegaConf

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import CNNClassifier, TransferLearningClassifier
from src.transforms import Normalize


class TilePredictor:
    """Loads a trained model and performs tiled prediction on images.
    
    Workflow:
        1. Load model from MLflow (registered model, run name, or checkpoint)
        2. Auto-detect config (crop_size, channels, class names) from MLflow artifacts
        3. Accept a multi-channel image (C, H, W) numpy array
        4. Tile it into crop_size x crop_size patches
        5. Run inference on each tile
        6. Return per-tile predictions + majority vote
    """
    
    def __init__(
        self,
        tracking_uri: str = "file:./mlruns",
        experiment_name: str = "image_classifier",
        model_name: str = "",
        run_name: str = "",
        checkpoint_path: str = "",
        crop_size: int = 224,
        stride: Optional[int] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Load model and extract config from MLflow artifacts
        self.model, self.model_info = self._load_model(
            model_name=model_name,
            run_name=run_name,
            checkpoint_path=checkpoint_path,
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Use config from MLflow artifacts if available, otherwise use provided values
        self.crop_size = self.model_info.get("crop_size", crop_size)
        self.stride = stride if stride is not None else self.crop_size
        self.class_names = self.model_info["class_names"]
        self.in_channels = self.model_info.get("in_channels", 5)
        
        # Preprocessing: normalize per-channel (zero mean, unit variance)
        self.normalize = Normalize()
    
    def _load_model(self, model_name: str, run_name: str, checkpoint_path: str):
        """Load model with priority: model_name > run_name > checkpoint_path.
        
        For MLflow sources, also downloads hydra_config.yaml and dataset_manifest.json
        to auto-configure class names, crop size, channels, etc.
        """
        run_id = None
        model = None
        info = {}
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # ── 1. Load from MLflow registered model (e.g. "TransferLearning/20") ──
        if model_name:
            run_id, model = self._load_from_registry(model_name)
            info["source"] = f"registry:{model_name}"
        
        # ── 2. Load from MLflow run name ──
        elif run_name:
            run_id, model = self._load_from_run_name(run_name)
            info["source"] = f"run_name:{run_name}"
        
        # ── 3. Fall back to direct checkpoint path ──
        elif checkpoint_path:
            model = self._load_from_checkpoint(checkpoint_path)
            info["source"] = f"checkpoint:{checkpoint_path}"
        
        else:
            raise ValueError(
                "No model source specified. Set one of: "
                "model_name (e.g. 'TransferLearning/20'), "
                "run_name (e.g. 'my_training_run'), or "
                "checkpoint_path (e.g. 'checkpoints/model.ckpt')"
            )
        
        # Extract config from MLflow run artifacts if we have a run_id
        if run_id:
            info.update(self._load_run_config(run_id))
        
        # Ensure class_names is always set
        if "class_names" not in info:
            if hasattr(model, "class_names") and model.class_names:
                info["class_names"] = model.class_names
            else:
                info["class_names"] = [str(i) for i in range(model.num_classes)]
        
        info["run_id"] = run_id
        info["model_class"] = model.__class__.__name__
        info["num_classes"] = model.num_classes
        
        return model, info
    
    def _load_from_registry(self, model_ref: str):
        """Load from MLflow model registry. e.g. 'TransferLearning/20' or 'TransferLearning/latest'."""
        client = mlflow.MlflowClient()
        
        ref = model_ref.removeprefix("models:/")
        parts = ref.split("/")
        name = parts[0]
        version = parts[1] if len(parts) > 1 else "latest"
        
        if version == "latest":
            versions = client.get_latest_versions(name)
            if not versions:
                raise ValueError(f"No versions found for registered model '{name}'")
            version = versions[0].version
            print(f"Resolved 'latest' -> version {version} for model '{name}'")
        
        mv = client.get_model_version(name, version)
        run_id = mv.run_id
        print(f"Loading {name}/v{version} (run_id={run_id[:8]}...)")
        
        model = self._load_model_from_run(run_id, client)
        return run_id, model
    
    def _load_from_run_name(self, run_name: str):
        """Load from MLflow run by its display name."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found")
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"run_name = '{run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise ValueError(f"No run found with name '{run_name}' in experiment '{self.experiment_name}'")
        
        run_id = runs.iloc[0].run_id
        print(f"Found run '{run_name}' (run_id={run_id[:8]}...)")
        
        client = mlflow.MlflowClient()
        model = self._load_model_from_run(run_id, client)
        return run_id, model
    
    def _load_model_from_run(self, run_id: str, client):
        """Try loading model from a run: run artifact -> registry -> checkpoint fallback.
        
        Follows the same 3-tier pattern as Error_analysis.ipynb.
        """
        model = None
        
        # 1. Try runs:/{run_id}/model (logged via mlflow.pytorch.log_model)
        try:
            model_uri = f"runs:/{run_id}/model"
            print(f"  Trying: {model_uri}")
            model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            print(f"  ✓ Loaded from run artifact")
            return model
        except Exception:
            pass
        
        # 2. Try model registry (find version linked to this run)
        try:
            for mv in client.search_model_versions():
                if mv.run_id == run_id:
                    model_uri = f"models:/{mv.name}/{mv.version}"
                    print(f"  Trying registry: {model_uri}")
                    model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
                    print(f"  ✓ Loaded from model registry")
                    return model
        except Exception as e:
            print(f"  Registry lookup failed: {e}")
        
        # 3. Fall back to checkpoint artifact
        ckpt_artifacts = client.list_artifacts(run_id, path="checkpoints")
        if not ckpt_artifacts:
            raise FileNotFoundError(f"No model or checkpoint found in run {run_id}")
        
        ckpt_local = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=ckpt_artifacts[0].path,
            tracking_uri=self.tracking_uri,
        )
        print(f"  Loading from checkpoint: {ckpt_local}")
        
        # Determine model class from hydra config
        model = self._load_from_checkpoint(ckpt_local, run_id=run_id, client=client)
        print(f"  ✓ Loaded from checkpoint artifact")
        return model
    
    def _load_from_checkpoint(self, ckpt_path: str, run_id: str = None, client=None):
        """Load Lightning model from a .ckpt file, auto-detecting model class."""
        # Try to determine model class from MLflow config
        model_cls = None
        if run_id and client:
            try:
                cfg = self._download_hydra_config(run_id)
                model_target = cfg.model._target_
                if "TransferLearning" in model_target:
                    model_cls = TransferLearningClassifier
                else:
                    model_cls = CNNClassifier
            except Exception:
                pass
        
        # Auto-detect from checkpoint if config not available
        if model_cls is None:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            hparams = ckpt.get("hyper_parameters", {})
            if "backbone_name" in hparams:
                model_cls = TransferLearningClassifier
            else:
                model_cls = CNNClassifier
        
        return model_cls.load_from_checkpoint(
            ckpt_path, map_location=self.device, weights_only=False,
        )
    
    def _download_hydra_config(self, run_id: str):
        """Download and parse hydra_config.yaml from MLflow run artifacts."""
        artifact_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, tracking_uri=self.tracking_uri,
        )
        config_path = Path(artifact_dir) / "hydra_config.yaml"
        return OmegaConf.load(config_path)
    
    def _load_run_config(self, run_id: str) -> dict:
        """Extract crop_size, channels, class_names from MLflow run artifacts."""
        info = {}
        
        try:
            cfg = self._download_hydra_config(run_id)
            ds_cfg = cfg.datamodule.dataset
            info["crop_size"] = ds_cfg.get("crop_size", 224)
            info["in_channels"] = len(list(ds_cfg.get("channels", [1, 2, 3, 4, 5])))
            
            backbone = cfg.model.get("backbone_name", cfg.model.get("_target_", "unknown"))
            info["backbone"] = backbone
        except Exception as e:
            print(f"  Warning: could not load hydra config: {e}")
        
        # Load class names from dataset manifest
        try:
            artifact_dir = mlflow.artifacts.download_artifacts(
                run_id=run_id, tracking_uri=self.tracking_uri,
            )
            manifest_path = Path(artifact_dir) / "dataset_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                all_labels = sorted(set(
                    s["label"] for s in manifest.get("train_samples", []) + manifest.get("val_samples", [])
                ))
                if all_labels:
                    info["class_names"] = all_labels
        except Exception as e:
            print(f"  Warning: could not load dataset manifest: {e}")
        
        return info
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess raw multi-channel image.
        
        Args:
            image: numpy array of shape (C, H, W), float32, raw pixel values
            
        Returns:
            Tensor of shape (C, H, W) with per-channel percentile normalization
        """
        # Percentile-clip per channel (same as dataset.py)
        processed_channels = []
        for c in range(image.shape[0]):
            ch = image[c]
            p_lo, p_hi = np.percentile(ch, [1, 99.5])
            if p_hi - p_lo > 0:
                ch = np.clip(ch, p_lo, p_hi)
                ch = ((ch - p_lo) / (p_hi - p_lo)).astype(np.float32)
            else:
                ch = np.zeros_like(ch, dtype=np.float32)
            processed_channels.append(ch)
        
        tensor = torch.from_numpy(np.stack(processed_channels, axis=0))
        return tensor
    
    def tile_image(self, image: torch.Tensor) -> list[dict]:
        """Split image into non-overlapping (or strided) tiles.
        
        Args:
            image: Tensor of shape (C, H, W)
            
        Returns:
            List of dicts with keys: 'tile' (C, crop_size, crop_size), 'row', 'col', 'y', 'x'
        """
        _, h, w = image.shape
        tiles = []
        
        row_idx = 0
        for y in range(0, h - self.crop_size + 1, self.stride):
            col_idx = 0
            for x in range(0, w - self.crop_size + 1, self.stride):
                tile = image[:, y:y + self.crop_size, x:x + self.crop_size]
                tiles.append({
                    "tile": tile,
                    "row": row_idx,
                    "col": col_idx,
                    "y": y,
                    "x": x,
                })
                col_idx += 1
            row_idx += 1
        
        return tiles
    
    @torch.no_grad()
    def predict_tiles(self, tiles: list[dict]) -> list[dict]:
        """Run model inference on all tiles.
        
        Args:
            tiles: List from tile_image()
            
        Returns:
            List of dicts with prediction info per tile
        """
        if not tiles:
            return []
        
        # Batch all tiles together
        batch = torch.stack([t["tile"] for t in tiles]).to(self.device)
        
        # Apply normalization per tile
        normalized = []
        for i in range(batch.shape[0]):
            normalized.append(self.normalize(batch[i]))
        batch = torch.stack(normalized)
        
        # Forward pass
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        results = []
        for i, tile_info in enumerate(tiles):
            results.append({
                "row": tile_info["row"],
                "col": tile_info["col"],
                "y": tile_info["y"],
                "x": tile_info["x"],
                "predicted_class": self.class_names[preds[i].item()],
                "predicted_idx": preds[i].item(),
                "confidence": probs[i, preds[i]].item(),
                "probabilities": {
                    name: probs[i, j].item()
                    for j, name in enumerate(self.class_names)
                },
            })
        
        return results
    
    def majority_vote(self, tile_predictions: list[dict]) -> dict:
        """Compute majority vote across all tile predictions.
        
        Args:
            tile_predictions: List from predict_tiles()
            
        Returns:
            Dict with overall prediction and vote counts
        """
        if not tile_predictions:
            return {"predicted_class": "unknown", "confidence": 0.0, "vote_counts": {}}
        
        votes = [t["predicted_class"] for t in tile_predictions]
        counter = Counter(votes)
        winner, winner_count = counter.most_common(1)[0]
        
        # Average the winner's probability across ALL tiles (not just tiles that voted winner)
        winner_confidences = [t["probabilities"][winner] for t in tile_predictions]
        avg_confidence = sum(winner_confidences) / len(winner_confidences)
        
        return {
            "predicted_class": winner,
            "confidence": avg_confidence,
            "total_tiles": len(tile_predictions),
            "vote_counts": dict(counter),
            "vote_fraction": winner_count / len(tile_predictions),
        }
    
    def predict(self, image: np.ndarray) -> dict:
        """Full prediction pipeline: preprocess -> tile -> predict -> majority vote.
        
        Args:
            image: numpy array of shape (C, H, W), raw pixel values
            
        Returns:
            Dict with 'tile_predictions' and 'image_prediction' (majority vote)
        """
        # Preprocess
        tensor = self.preprocess_image(image)
        
        # Tile
        tiles = self.tile_image(tensor)
        
        # Predict per tile
        tile_predictions = self.predict_tiles(tiles)
        
        # Majority vote
        image_prediction = self.majority_vote(tile_predictions)
        
        return {
            "image_prediction": image_prediction,
            "tile_predictions": tile_predictions,
            "image_shape": list(image.shape),
            "num_tiles": len(tile_predictions),
            "tile_grid": {
                "rows": max((t["row"] for t in tile_predictions), default=0) + 1,
                "cols": max((t["col"] for t in tile_predictions), default=0) + 1,
            } if tile_predictions else {"rows": 0, "cols": 0},
        }
