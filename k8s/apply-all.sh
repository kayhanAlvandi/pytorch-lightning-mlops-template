#!/usr/bin/env bash
# One-off script to apply everything built so far, in dependency order.
# Run once against a fresh `kind-up` cluster; safe to delete after running --
# nothing here is sensitive on its own (real secret *values* live in the
# gitignored config/*.yaml files this script references, not in this file).
set -euo pipefail
cd "$(dirname "$0")"

echo "== namespaces (+ default ServiceAccount ghcr-pull wiring) =="
kubectl apply -f namespaces/

echo "== secrets (gitignored real values) =="
kubectl apply -f config/ghcr-pull-secret.yaml
kubectl apply -f config/api-secrets.yaml
kubectl apply -f config/monitoring-secrets.yaml

echo "== configmaps =="
kubectl apply -f config/api-configs.yaml
kubectl apply -f config/monitoring-configs.yaml

echo "== external services (postgres/mlflow/api DNS aliases -> host.docker.internal) =="
kubectl apply -f external-services/

echo "== jobs =="
# Jobs are immutable once created -- if you re-run this after a job already
# ran, `kubectl apply` on the same Job will error; `kubectl delete -f jobs/`
# first if you need to re-trigger one.
kubectl apply -f jobs/

echo "== done. check status with: kubectl get all -n monitoring -n api =="
