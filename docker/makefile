# Makefile

# Start MLflow (prerequisite for training)
mlflow:
	docker compose -f Docker-compose.mlflow.yaml up -d

# Run a training job (one-off, container exits when done)
train:
	docker compose -f Docker-compose.mlflow.yaml -f Docker-compose.training.yaml run --rm training

# Run training with custom config override
# Usage: make train-run CMD="python train.py trainer=trainer_container model=vit_large"
train-run:
	docker compose -f Docker-compose.mlflow.yaml -f Docker-compose.training.yaml run --rm training $(CMD)

# Serving
serve:
	docker compose -f Docker-compose.mlflow.yaml -f Docker-compose.api.yaml up -d

down:
	docker compose -f Docker-compose.mlflow.yaml -f Docker-compose.api.yaml down

down-all:
	docker compose -f Docker-compose.mlflow.yaml -f Docker-compose.training.yaml -f Docker-compose.api.yaml down