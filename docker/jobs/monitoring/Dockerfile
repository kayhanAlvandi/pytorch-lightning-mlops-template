FROM python:3.11-slim

LABEL maintainer="Kayhan Alvandi <keyhanalvandi96@gmail.com>"
LABEL description="Monitoring jobs (drift report, quality report, label backfill)"

WORKDIR /app

# Lightweight on purpose: Evidently + pandas + psycopg + pydantic-settings +
# pymongo/joblib (for label backfill via external tools package).
# No torch/timm/mlflow -- run_id is read off the running API's /model endpoint,
# and this job never loads the model itself.
COPY requirements/monitoring_req.txt .
RUN pip install --no-cache-dir -r monitoring_req.txt

# Code baked in (this is a deployable/scheduled job, meant for cron/Airflow later).
COPY database/ ./database/
COPY monitoring/ ./monitoring/

# Entrypoint: if the external tools package is mounted (label-backfill service),
# install it in editable mode; otherwise the install is a no-op and drift/quality
# runs are unaffected.
RUN printf '#!/bin/bash\npip install -e /app/tools 2>/dev/null\nexec "$@"\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["python", "-m", "monitoring.run_drift_report"]
