"""Configuration for monitoring jobs.

Deliberately separate from ``api.config.Settings``: the drift-report job only
needs a database URI and the base URL of the running API (to read the served
model's run_id off ``/model``). It has no business with model-serving settings
like ``model_name`` / ``crop_size`` / ``device``, so reusing the API's Settings
here would just pull in irrelevant, confusing fields.

Env vars share the ``MONITORING_`` prefix so a single ``.env.monitoring`` works for both the
API and the monitoring jobs (``MONITORING_DB_URI``, ``MONITORING_BASE_URL``).
"""
from pydantic_settings import BaseSettings


class MonitoringSettings(BaseSettings):
    """Settings for monitoring jobs, loaded from environment or .env file."""

    # ── Database ─────────────────────────────────────────────────────────
    db_uri: str | None = None

    # ── Running API (source of the served model's run_id via /model) ─────
    base_url: str = "http://localhost:8000"

    model_config = {"env_prefix": "MONITORING_", "env_file": ".env", "extra": "ignore"}

    @property
    def has_db_uri(self) -> bool:
        return bool(self.db_uri)
