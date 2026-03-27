# MLflow Integration Guide

## Overview

This project uses **dual logging** for comprehensive experiment tracking:
- **TensorBoard**: Real-time training visualization (loss curves, gradients, confusion matrices)
- **MLflow**: Experiment tracking, hyperparameter comparison, model versioning

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start MLflow UI** (in a separate terminal):
```bash
mlflow ui
```
Access at: http://localhost:5000

3. **Start TensorBoard** (optional, for detailed visualizations):
```bash
tensorboard --logdir logs
```
Access at: http://localhost:6006

## Running Experiments

### Basic Training
```bash
python train.py --config config/config.yaml
```

### Experiment with Different Hyperparameters
```bash
# Experiment 1: Lower learning rate
python train.py --config config/config.yaml --lr 0.0001

# Experiment 2: Higher dropout
# (Edit config.yaml: model.dropout = 0.7)
python train.py --config config/config.yaml

# Experiment 3: More data
# (Edit config.yaml: max_wells_per_label = 4)
python train.py --config config/config.yaml
```

Each run is automatically tracked in MLflow with:
- All hyperparameters from config.yaml
- Metrics: train/val loss, accuracy, F1-score
- Artifacts: config file, model checkpoints
- Model registered in Model Registry

## MLflow UI Features

### 1. Compare Runs
- Go to http://localhost:5000
- Select multiple runs (checkbox)
- Click "Compare" button
- View side-by-side metrics and parameters

### 2. Search & Filter
```
# Find runs with val_acc > 0.7
metrics.val/acc > 0.7

# Find runs with specific LR
params.training.learning_rate = "0.001"

# Find recent runs
attributes.start_time > "2026-02-20"
```

### 3. Download Artifacts
- Click on a run
- Go to "Artifacts" tab
- Download: config.yaml, model checkpoints, confusion matrices

### 4. Model Registry
- Navigate to "Models" tab
- See all versions of "cnn_classifier"
- Promote to stages: None → Staging → Production
- Add descriptions and tags

## What Gets Logged

### Parameters (from config.yaml)
```python
data.root_dir
data.channels
data.crop_size
dataloader.batch_size
dataloader.max_wells_per_label
dataloader.max_samples_per_label
training.learning_rate
training.weight_decay
model.dropout
seed
```

### Metrics (per epoch)
```python
train/loss
train/acc
val/loss
val/acc
val/f1
grad/total_norm  # Every 50 steps
```

### Artifacts
```
config.yaml           # Exact config used
model/                # PyTorch model
checkpoints/          # Best model checkpoint
confusion_matrix.png  # From TensorBoard
predictions.png       # Sample predictions
```

## Model Versioning Workflow

### Scenario: Testing Different Architectures

1. **Baseline (SimpleCNN)**:
```bash
python train.py --config config/config.yaml
# MLflow: Run 1, Model v1
```

2. **Try ResNet18** (after implementing):
```bash
# Edit config.yaml: model.name = "ResNet18"
python train.py --config config/config.yaml
# MLflow: Run 2, Model v2
```

3. **Compare in MLflow UI**:
- Select both runs
- Compare val_acc: SimpleCNN (0.698) vs ResNet18 (0.75)
- Promote ResNet18 to "Production"

4. **Rollback if needed**:
- Go to Model Registry
- Load v1 (SimpleCNN) from "Archived"

## Advanced: Programmatic Access

### Load Best Model
```python
import mlflow

# Load latest production model
model_uri = "models:/cnn_classifier/Production"
model = mlflow.pytorch.load_model(model_uri)

# Or load specific version
model_uri = "models:/cnn_classifier/2"
model = mlflow.pytorch.load_model(model_uri)
```

### Query Runs
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get best run by val_acc
runs = client.search_runs(
    experiment_ids=["0"],
    order_by=["metrics.val/acc DESC"],
    max_results=1
)

best_run = runs[0]
print(f"Best val_acc: {best_run.data.metrics['val/acc']}")
print(f"LR: {best_run.data.params['training.learning_rate']}")
```

## File Structure

```
cnn_classifier/
├── mlruns/              # MLflow tracking data
│   ├── 0/              # Experiment ID
│   │   ├── <run_id>/   # Individual runs
│   │   │   ├── artifacts/
│   │   │   ├── metrics/
│   │   │   └── params/
├── logs/               # TensorBoard logs
│   └── cnn_classifier/
└── checkpoints/        # Model checkpoints
```

## Tips

1. **Tag runs** for easy filtering:
```python
mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "experiment", "lr_tuning")
```

2. **Add notes** to runs in UI (click run → "Edit" → "Description")

3. **Export comparison** as CSV (Compare view → "Download CSV")

4. **Remote tracking**: Change `tracking_uri` to shared server for team collaboration

5. **Clean old runs**:
```bash
# Delete experiment
mlflow experiments delete --experiment-id 0
```

## Troubleshooting

### MLflow UI not starting
```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000

# Use different port
mlflow ui --port 5001
```

### Model not registered
- Check logs for errors during `mlflow.pytorch.log_model`
- Ensure training completed successfully
- Verify `mlruns/` directory exists

### Metrics not showing
- Metrics logged via PyTorch Lightning automatically sync to MLflow
- Check TensorBoard first to verify metrics are being logged
- Refresh MLflow UI (Ctrl+R)

## Next Steps

1. Run 3-5 experiments with different hyperparameters
2. Compare results in MLflow UI
3. Identify best configuration
4. Promote best model to Production
5. Use for inference or further tuning
