# Image Classifier API

FastAPI service for tiled multi-channel image classification with majority voting.
Models are loaded directly from **MLflow** (registered model or run name).

## Overview

Upload a large image (e.g., 2000x2000 pixels) and the API will:
1. Load the model + config from MLflow (class names, crop size, channels auto-detected)
2. Split the image into tiles (e.g., 224x224)
3. Run the trained model on each tile
4. Return per-tile predictions + whole-image prediction via majority vote

## Setup

```bash
pip install -r api/requirements.txt
```

## Run

Pick **one** model source (priority: model_name > run_name > checkpoint_path):

```bash
# Option 1: Load from MLflow registered model (e.g. "TransferLearning/20")
set API_MODEL_NAME=TransferLearning/20
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Option 2: Load from MLflow run name
set API_RUN_NAME=Vits_finetune_cosine_warmup_autoGradual_moredata
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Option 3: Load from checkpoint file (fallback)
set API_CHECKPOINT_PATH=checkpoints/model.ckpt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### All Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_MODEL_NAME` | `""` | MLflow registered model, e.g. `TransferLearning/20` or `TransferLearning/latest` |
| `API_RUN_NAME` | `""` | MLflow run name |
| `API_CHECKPOINT_PATH` | `""` | Direct `.ckpt` path (fallback) |
| `API_TRACKING_URI` | `file:./mlruns` | MLflow tracking URI |
| `API_EXPERIMENT_NAME` | `image_classifier` | MLflow experiment name |
| `API_CROP_SIZE` | `224` | Tile size (auto-detected from MLflow if available) |
| `API_STRIDE` | `null` | Tile stride (defaults to crop_size = non-overlapping) |
| `API_DEVICE` | `cpu` | `cpu` or `cuda` |

## Endpoints

### `GET /health`
Health check - returns model status.

### `GET /model`
Returns detailed info about the loaded model: source, backbone, class names, 
crop size, channels, etc. All auto-detected from MLflow artifacts.

### `POST /predict`
Upload image file(s) for tiled prediction.

**Input options:**
- Single `.npy` file: pre-stacked `(C, H, W)` array
- Single multi-page `.tif`: each page = one channel
- Multiple files: one per channel, stacked in upload order

**Query params:**
- `crop_size` (optional): override tile size for this request
- `stride` (optional): override tile stride for this request

**Response:**
```json
{
  "image_prediction": {
    "predicted_class": "class_name",
    "confidence": 0.95,
    "total_tiles": 64,
    "vote_counts": {"class_A": 50, "class_B": 14},
    "vote_fraction": 0.78
  },
  "tile_predictions": [
    {
      "row": 0, "col": 0,
      "y": 0, "x": 0,
      "predicted_class": "class_A",
      "confidence": 0.92,
      "probabilities": {"class_A": 0.92, "class_B": 0.08}
    }
  ],
  "image_shape": [5, 2000, 2000],
  "num_tiles": 64,
  "tile_grid": {"rows": 8, "cols": 8}
}
```

### `POST /predict/single-channel`
Upload a single grayscale image - it will be replicated across all expected channels.
Useful for quick testing.

## Example Usage

```python
import requests

# Check model info
resp = requests.get("http://localhost:8000/model")
print(resp.json())

# Single .npy file
with open("test_image.npy", "rb") as f:
    resp = requests.post("http://localhost:8000/predict", files={"files": f})

# Multiple channel files
files = [
    ("files", open("ch1.tif", "rb")),
    ("files", open("ch2.tif", "rb")),
    ("files", open("ch3.tif", "rb")),
    ("files", open("ch4.tif", "rb")),
    ("files", open("ch5.tif", "rb")),
]
resp = requests.post("http://localhost:8000/predict", files=files)
print(resp.json())
```
