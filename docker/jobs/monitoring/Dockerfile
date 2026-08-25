FROM python:3.11-slim

LABEL maintainer="Kayhan Alvandi <keyhanalvandi96@gmail.com>"
LABEL description="Drift monitoring job (Evidently) for image classifier"

WORKDIR /app

# Lightweight on purpose: only Evidently + pandas + psycopg + pydantic-settings.
# No torch/timm/mlflow -- run_id is read off the running API's /model endpoint,
# and this job never loads the model itself.
COPY requirements/monitoring_req.txt .
RUN pip install --no-cache-dir -r monitoring_req.txt

# Code baked in (this is a deployable/scheduled job, meant for cron/Airflow later).
COPY database/ ./database/
COPY monitoring/ ./monitoring/

CMD ["python", "-m", "monitoring.run_drift_report"]
