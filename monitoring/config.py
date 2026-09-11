"""Configuration for monitoring jobs.

Deliberately separate from ``api.config.Settings``: the drift-report job only
needs a database URI and the base URL of the running API (to read the served
model's run_id off ``/model``). It has no business with model-serving settings
like ``model_name`` / ``crop_size`` / ``device``, so reusing the API's Settings
here would just pull in irrelevant, confusing fields.

Env vars share the ``MONITORING_`` prefix so a single ``.env.monitoring`` works for both the
API and the monitoring jobs (``MONITORING_DB_URI``, ``MONITORING_BASE_URL``).

Also holds small helpers shared by the report jobs (drift + quality): resolving
the served model's run_id and parsing the analysis window, so both scripts share
one implementation instead of duplicating it.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta

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


def resolve_run_id(run_id: str | None, api_url: str) -> str | None:
    """Resolve the target serving model's MLflow run_id.

    An explicit ``run_id`` wins. Otherwise it's read off the running API's
    ``/model`` endpoint -- the API already loaded the model and knows its
    run_id, so the report jobs don't need mlflow (or the tracking server) at
    all, just an HTTP GET against whatever model is actually being served.
    """
    if run_id:
        return run_id

    url = f"{api_url.rstrip('/')}/model"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            info = json.load(resp)
    except Exception as e:
        raise RuntimeError(
            f"Could not read run_id from the API at {url} ({e}). "
            f"Is the API running? Otherwise pass --run-id explicitly."
        ) from e
    return info.get("run_id")


def resolve_window(
    window_start: str | None,
    window_end: str | None,
    window_days: float,
) -> tuple[datetime, datetime]:
    """Resolve the analysis window [start, end).

    Explicit ISO ``window_start``/``window_end`` win; otherwise the window is
    the last ``window_days`` ending now. Raises ValueError if start >= end.
    """
    end = datetime.fromisoformat(window_end) if window_end else datetime.now()
    start = (
        datetime.fromisoformat(window_start)
        if window_start
        else end - timedelta(days=window_days)
    )
    if start >= end:
        raise ValueError(f"window_start ({start}) must be before window_end ({end}).")
    return start, end
