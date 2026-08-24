# Makefile — independent services connected via external Docker network

# ── Network (shared by all services/jobs) ────────────────────────────────
network:
	-docker network create ml-platform

# ── Services (long-running) ──────────────────────────────────────────────
mlflow: network
	docker compose -f services/mlflow/docker-compose.yaml up -d

serve: network
	docker compose -f services/api/docker-compose.yaml up -d

# Dev: bind-mounts api/utils/database/monitoring source + --reload, no rebuild on edit
serve-dev: network
	docker compose -f services/api/docker-compose.yaml -f services/api/docker-compose.dev.yaml up -d

# ── Jobs (ephemeral, run-and-exit) ───────────────────────────────────────
# Usage: make train-run CMD="python train.py trainer=trainer_container model=vit_large"
train-run: network
	docker compose -f jobs/train/docker-compose.yaml run --rm training $(CMD)

# Usage: make compute-reference (reads model/db config from services/api/.env.api)
compute-reference: network
	docker compose -f jobs/compute-reference/docker-compose.yaml run --rm compute-reference \
		$(if $(CMD),$(CMD),python -m monitoring.compute_reference)

# ── Teardown (each is independent — won't affect others) ─────────────────
down-mlflow:
	docker compose -f services/mlflow/docker-compose.yaml down

down-serve:
	docker compose -f services/api/docker-compose.yaml down

down-all: down-serve down-mlflow