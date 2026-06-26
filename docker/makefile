# Makefile
serve:
	docker compose -f Docker-compose.mlflow.yaml -f Docker-compose.api.yml up -d

train:
	docker compose -f Docker-compose.mlflow.yaml -f Docker-compose.training.yml up -d

all:
	docker compose -f Docker-compose.mlflow.yaml -f Docker-compose.api.yml up -d

down:
	docker compose -f Docker-compose.mlflow.yaml -f Docker-compose.api.yml down