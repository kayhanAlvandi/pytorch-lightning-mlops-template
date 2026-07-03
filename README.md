# pytorch-lightning-mlops-template

An end-to-end **MLOps template** for image classification, built on **PyTorch Lightning** and **Hydra**, with **MLflow** experiment tracking + model registry, a **FastAPI** serving layer, and **Docker Compose** for reproducible training and deployment.

The template ships with a working multi-channel microscopy classifier as a reference implementation, but the training loop, config system, tracking, and serving stack are model- and dataset-agnostic — swap in your own dataset, transforms, and architecture via config.

## Features

- **PyTorch Lightning** training loop with a swappable `LightningModule` and `LightningDataModule`.
- **Hydra** config groups for `datamodule`, `model`, `optimizer`, `loss`, `callbacks`, and `trainer` — compose experiments from the CLI, no code edits.
- **MLflow** tracking (params, metrics, system metrics), artifact logging, and Model Registry with resume-from-registered-model support.
- **Dual logging**: MLflow + TensorBoard.
- **Dataset versioning**: content-hashed dataset version, manifest, and metadata logged per run for reproducibility.
- **FastAPI** serving that loads a model directly from MLflow (by registered name or run name) and performs tiled inference with majority voting.
- **Docker Compose** stacks for MLflow, GPU training, and the API, orchestrated via a `Makefile`.
- A model zoo of ready-to-use configs: `simplecnn`, `resnet18/50`, `efficientnet_b0/b3`, `vit_small/base` (+ fine-tune variants).

## Project Structure

```
pytorch-lightning-mlops-template/
├── configs/                    # Hydra config groups
│   ├── config.yaml             # Root config (composes the groups below)
│   ├── datamodule/             # Dataset + dataloader configs
│   ├── model/                  # Architecture configs (cnn, resnet, efficientnet, vit, ...)
│   ├── optimizer/              # Optimizer + scheduler configs
│   ├── loss/                   # Loss function configs
│   ├── callbacks/              # Lightning callback configs
│   └── trainer/                # Trainer configs (local / container)
├── src/
│   ├── config.py               # Config dataclasses
│   ├── dataset.py              # Multi-channel image dataset
│   ├── datamodule.py           # LightningDataModule
│   ├── transforms.py           # Image / batch transforms (incl. Mixup/CutMix)
│   ├── model.py                # LightningModule + architectures
│   ├── callbacks.py            # Custom callbacks (e.g. LogBestModelToMLflow)
│   └── dataset_versioning.py   # Dataset hashing, manifest, git tracking
├── api/                        # FastAPI serving layer (loads model from MLflow)
├── docker/                     # Dockerfiles, compose files, Makefile
├── tests/                      # Checkpoint / hparams tests
├── benchmarks/                 # Dataloader & model benchmarks
├── train.py                    # Hydra entrypoint for training
├── MLFLOW_GUIDE.md             # MLflow usage guide
└── README.md
```

## Installation

### 1. (Reference dataset only) Set up PYTHONPATH for the shared tools library

The bundled microscopy example uses the shared `tools.loading` module. If you use the template with your own dataset you can skip this.

```bash
conda env config vars set PYTHONPATH=L:\GITHUB\LIB_Python
conda activate tools  # reactivate to apply
```

### 2. Install dependencies

```bash
pip install -r src/requirements.txt
```

## Configuration (Hydra)

The root config `configs/config.yaml` composes one option from each config group:

```yaml
defaults:
  - datamodule: tiled
  - model: simplecnn
  - optimizer: adamw
  - loss: cross_entropy
  - callbacks: default
  - trainer: default
```

Override any group or value from the CLI — no code changes required.

## Usage

### Basic training

```bash
python train.py
```

### Compose experiments via config overrides

```bash
# Swap the architecture
python train.py model=resnet50
python train.py model=vit_base_finetune

# Swap optimizer / loss / datamodule
python train.py optimizer=sgd loss=focal

# Override individual values
python train.py seed=123 run_name=my_experiment tags=[baseline,lr_sweep]
```

### Resume / fine-tune from a registered MLflow model

```bash
# Full resume (weights + optimizer + scheduler + epoch)
python train.py resume_from_model=SimpleCNN/11

# Load weights only (fresh optimizer/scheduler — safe for gradual unfreezing)
python train.py resume_from_model=SimpleCNN/latest resume_weights_only=true
```

## Experiment Tracking

Training uses dual logging:

- **MLflow** — params, metrics, system metrics, artifacts (Hydra config, dataset manifest/metadata), and Model Registry.
- **TensorBoard** — real-time curves, confusion matrices, sample predictions.

```bash
mlflow ui                 # http://localhost:5000
tensorboard --logdir logs # http://localhost:6006
```

Each run automatically logs a content-hashed **dataset version** plus the model code git commit, so experiments are reproducible. See `MLFLOW_GUIDE.md` for details.

## Serving (FastAPI)

The API loads a trained model directly from MLflow and serves tiled predictions with majority voting. See `api/README.md` for full endpoint docs.

```bash
# Point the API at a registered model or a run name
set API_MODEL_NAME=SimpleCNN/latest
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker

Reproducible stacks are defined under `docker/` and orchestrated via the `Makefile` (run from the `docker/` directory):

```bash
make mlflow      # start the MLflow tracking server (prerequisite)
make train       # run a one-off GPU training job
make train-run CMD="python train.py model=vit_base"  # training with overrides
make serve       # start MLflow + the API
make down        # stop API stack
make down-all    # stop all services
```

Services:

- **mlflow** — tracking server (SQLite backend, artifact store) on port `5000`.
- **training** — GPU-enabled one-off training container.
- **api** — FastAPI serving container on port `8000`.

## Using This Template

1. Click **"Use this template"** on GitHub (or clone) to create your own repo.
2. Replace the dataset logic in `src/dataset.py` / `src/datamodule.py` and its config in `configs/datamodule/`.
3. Add or pick an architecture in `configs/model/` (or extend `src/model.py`).
4. Adjust `configs/` for your optimizer, loss, callbacks, and trainer.
5. Train, track in MLflow, register your best model, and serve it via the API.

## Reference Implementation: Multi-Channel Microscopy

The included example classifies multi-channel microscopy images with configurable channel selection and tiled cropping.

Image naming pattern:
```
{plate}_{well}_T{time}F{field}L{layer}A{action}Z{z}C{channel}.jxl
```
Example: `MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A01Z01C01.jxl`

Labels are resolved from plate/well information (dummy labels by default, or MongoDB).
