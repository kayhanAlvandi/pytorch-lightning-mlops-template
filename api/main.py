"""FastAPI application for tiled image classification.

Upload a multi-channel image (or multiple single-channel files) and get:
  - Per-tile predictions
  - Whole-image prediction via majority vote across tiles

Models are loaded from MLflow (registered model name, run name, or checkpoint).
"""
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.config import Settings
from api.predictor import TilePredictor

# Global predictor instance (loaded at startup)
predictor: Optional[TilePredictor] = None
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup from MLflow or checkpoint."""
    global predictor
    
    if not settings.has_model_source:
        print("WARNING: No model source configured.")
        print("Set one of: API_MODEL_NAME, API_RUN_NAME, or API_CHECKPOINT_PATH")
        print("The /predict endpoint will return an error until a model is loaded.")
    else:
        source = settings.model_name or settings.run_name or settings.checkpoint_path
        print(f"Loading model: {source}")
        predictor = TilePredictor(
            tracking_uri=settings.tracking_uri,
            experiment_name=settings.experiment_name,
            model_name=settings.model_name,
            run_name=settings.run_name,
            checkpoint_path=settings.checkpoint_path,
            crop_size=settings.crop_size,
            stride=settings.effective_stride,
            device=settings.device,
        )
        print(f"Model loaded. Source: {predictor.model_info['source']}")
        print(f"  Classes: {predictor.class_names}")
        print(f"  Tile size: {predictor.crop_size}, Stride: {predictor.stride}")
        print(f"  In channels: {predictor.in_channels}")
    
    yield
    
    # Cleanup
    predictor = None


app = FastAPI(
    title="Image Classifier API",
    description="Tiled multi-channel image classification with majority voting. "
                "Models loaded from MLflow registry, run name, or checkpoint.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "device": settings.device,
    }


@app.get("/model")
async def model_info():
    """Return detailed info about the loaded model and its config."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="No model loaded.")
    
    return {
        "source": predictor.model_info.get("source"),
        "model_class": predictor.model_info.get("model_class"),
        "backbone": predictor.model_info.get("backbone"),
        "run_id": predictor.model_info.get("run_id"),
        "num_classes": predictor.model_info.get("num_classes"),
        "class_names": predictor.class_names,
        "in_channels": predictor.in_channels,
        "crop_size": predictor.crop_size,
        "stride": predictor.stride,
        "device": str(predictor.device),
    }


@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(..., description="Image files (one per channel, ordered C1..CN) or a single multi-channel .tif/.npy file"),
    crop_size: Optional[int] = Query(None, description="Override tile crop size"),
    stride: Optional[int] = Query(None, description="Override tile stride"),
):
    """Predict on an uploaded image.
    
    Accepts either:
    - A single .npy file containing a pre-stacked (C, H, W) array
    - Multiple image files (one per channel), which will be stacked in upload order
    
    Returns per-tile predictions and a majority-vote whole-image prediction.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Set API_MODEL_NAME, API_RUN_NAME, or API_CHECKPOINT_PATH.",
        )
    
    try:
        image = await _load_image_from_uploads(files)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Optionally override tiling params for this request
    original_crop = predictor.crop_size
    original_stride = predictor.stride
    
    if crop_size is not None:
        predictor.crop_size = crop_size
        if stride is None:
            predictor.stride = crop_size
    if stride is not None:
        predictor.stride = stride
    
    try:
        result = predictor.predict(image)
    finally:
        # Restore original settings
        predictor.crop_size = original_crop
        predictor.stride = original_stride
    
    return JSONResponse(content=result)


@app.post("/predict/single-channel")
async def predict_single_channel(
    file: UploadFile = File(..., description="Single grayscale image (will be replicated to all channels)"),
):
    """Predict on a single-channel image by replicating it across all expected channels.
    
    Useful for quick testing with standard grayscale images.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Set API_MODEL_NAME, API_RUN_NAME, or API_CHECKPOINT_PATH.",
        )
    
    try:
        content = await file.read()
        img = _load_single_image(content, file.filename)
        # Replicate to expected number of channels (auto-detected from model)
        image = np.stack([img] * predictor.in_channels, axis=0)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    result = predictor.predict(image)
    return JSONResponse(content=result)


async def _load_image_from_uploads(files: list[UploadFile]) -> np.ndarray:
    """Load uploaded files into a (C, H, W) numpy array.
    
    - Single .npy file: already stacked (C, H, W)
    - One or more image files (.tif, .jxl, etc.): each file = one channel, stacked in order
    """
    if len(files) == 1:
        content = await files[0].read()
        filename = files[0].filename or "upload"
        
        if filename.endswith(".npy"):
            image = np.load(io.BytesIO(content))
            if image.ndim != 3:
                raise ValueError(f"Expected 3D array (C, H, W), got shape {image.shape}")
            return image.astype(np.float32)
        
        # Single image file = single channel
        img = _load_single_image(content, filename)
        return img[np.newaxis].astype(np.float32)
    
    # Multiple files: each file = one channel
    channels = []
    for f in files:
        content = await f.read()
        channels.append(_load_single_image(content, f.filename or "upload"))
    
    shapes = [ch.shape for ch in channels]
    if len(set(shapes)) > 1:
        raise ValueError(f"All channel images must have same dimensions. Got: {shapes}")
    
    return np.stack(channels, axis=0).astype(np.float32)


def _load_single_image(content: bytes, filename: str) -> np.ndarray:
    """Load a single image from bytes into a 2D numpy array."""
    import cv2
    from PIL import Image
    import pillow_jxl 

    suffix = Path(filename).suffix.lower()
    if suffix in (".tif", ".tiff"):
        buf = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(buf, -1)
    elif suffix == ".jxl":
        img = np.array(Image.open(io.BytesIO(content)))
    else:
        buf = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(buf, -1)
 
    if img is None:
        raise ValueError(f"Failed to decode image: {filename}")
 
    return img.astype(np.float32)
