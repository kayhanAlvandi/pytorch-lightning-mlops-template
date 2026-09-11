"""Unit tests for monitoring/config.py.

Covers MonitoringSettings defaults/env-loading, resolve_run_id (with mocked
HTTP), and resolve_window parsing/validation. No real API or database needed.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from monitoring.config import MonitoringSettings, resolve_run_id, resolve_window

# ─────────────────────────────────────────────────────────────────────────────
# MonitoringSettings
# ─────────────────────────────────────────────────────────────────────────────

class TestMonitoringSettings:
    def test_defaults(self):
        s = MonitoringSettings()
        assert s.db_uri is None
        assert s.base_url == "http://localhost:8000"
        assert s.has_db_uri is False

    def test_has_db_uri_true(self):
        s = MonitoringSettings(db_uri="postgresql://localhost/db")
        assert s.has_db_uri is True

    def test_env_prefix_loading(self, monkeypatch):
        monkeypatch.setenv("MONITORING_DB_URI", "postgresql://envhost/db")
        monkeypatch.setenv("MONITORING_BASE_URL", "http://envapi:9000")
        s = MonitoringSettings()
        assert s.db_uri == "postgresql://envhost/db"
        assert s.base_url == "http://envapi:9000"


# ─────────────────────────────────────────────────────────────────────────────
# resolve_run_id
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveRunId:
    def test_explicit_run_id_wins(self):
        assert resolve_run_id("abc123", "http://localhost:8000") == "abc123"

    def test_explicit_run_id_skips_http(self):
        """Even with a broken API URL, an explicit run_id is returned."""
        assert resolve_run_id("explicit", "http://nonexistent:9999") == "explicit"

    def test_reads_from_api(self):
        """When no run_id given, reads it off the API's /model endpoint."""
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps({"run_id": "from-api"}).encode()
        fake_resp.__enter__ = MagicMock(return_value=fake_resp)
        fake_resp.__exit__ = MagicMock(return_value=False)

        with patch("monitoring.config.urllib.request.urlopen", return_value=fake_resp):
            result = resolve_run_id(None, "http://localhost:8000")
        assert result == "from-api"

    def test_api_url_trailing_slash_stripped(self):
        """Trailing slash in api_url doesn't create a double-slash path."""
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps({"run_id": "x"}).encode()
        fake_resp.__enter__ = MagicMock(return_value=fake_resp)
        fake_resp.__exit__ = MagicMock(return_value=False)

        with patch("monitoring.config.urllib.request.urlopen", return_value=fake_resp) as mock:
            resolve_run_id(None, "http://localhost:8000/")
            called_url = mock.call_args[0][0]
            assert called_url == "http://localhost:8000/model"

    def test_api_failure_raises_runtime_error(self):
        with patch("monitoring.config.urllib.request.urlopen", side_effect=ConnectionError("refused")), \
             pytest.raises(RuntimeError, match="Could not read run_id"):
            resolve_run_id(None, "http://nonexistent:9999")

    def test_api_returns_null_run_id(self):
        """If the API responds with null run_id, None is returned."""
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps({"run_id": None}).encode()
        fake_resp.__enter__ = MagicMock(return_value=fake_resp)
        fake_resp.__exit__ = MagicMock(return_value=False)

        with patch("monitoring.config.urllib.request.urlopen", return_value=fake_resp):
            result = resolve_run_id(None, "http://localhost:8000")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# resolve_window
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveWindow:
    def test_explicit_start_end(self):
        start, end = resolve_window("2024-01-01T00:00:00", "2024-01-08T00:00:00", 7.0)
        assert start == datetime(2024, 1, 1)
        assert end == datetime(2024, 1, 8)

    def test_default_end_is_now(self):
        before = datetime.now()
        start, end = resolve_window(None, None, 7.0)
        after = datetime.now()
        assert before <= end <= after
        assert end - start == timedelta(days=7)

    def test_default_start_from_explicit_end(self):
        start, end = resolve_window(None, "2024-06-10T12:00:00", 3.0)
        assert end == datetime(2024, 6, 10, 12)
        assert start == datetime(2024, 6, 7, 12)

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="must be before"):
            resolve_window("2024-06-10", "2024-06-01", 7.0)

    def test_start_equals_end_raises(self):
        with pytest.raises(ValueError, match="must be before"):
            resolve_window("2024-06-10T00:00:00", "2024-06-10T00:00:00", 7.0)

    def test_window_days_fractional(self):
        start, end = resolve_window(None, "2024-06-10T12:00:00", 0.5)
        assert end - start == timedelta(hours=12)
