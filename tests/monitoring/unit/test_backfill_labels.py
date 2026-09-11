"""Unit tests for monitoring/backfill_labels.py.

The MongoDB lookup (utils.labels.resolve_labels_from_mongodb) is mocked so no
external services are needed. The DBLogger is replaced by the FakeDBLogger
from conftest.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import monitoring.backfill_labels as backfill


class TestBackfillLabelsMain:
    def test_no_db_uri(self, monkeypatch):
        """Without MONITORING_DB_URI, main() prints an error and returns."""
        monkeypatch.delenv("MONITORING_DB_URI", raising=False)
        result = backfill.main()
        assert result is None

    def test_no_unlabeled_wells(self, monkeypatch, fake_db):
        """When fetch_unlabeled_wells returns [], nothing is backfilled."""
        monkeypatch.setenv("MONITORING_DB_URI", "postgresql://test/db")
        with patch("monitoring.backfill_labels.DBLogger", return_value=fake_db), \
             patch("monitoring.backfill_labels.resolve_labels") as mock_resolve:
            backfill.main()
        mock_resolve.assert_not_called()
        assert fake_db.closed is True

    def test_no_labels_resolved(self, monkeypatch, fake_db):
        """When MongoDB returns no labels, nothing is written."""
        monkeypatch.setenv("MONITORING_DB_URI", "postgresql://test/db")
        fake_db.unlabeled_wells = [("P1", "K07"), ("P1", "K08")]
        with patch("monitoring.backfill_labels.DBLogger", return_value=fake_db), \
             patch("monitoring.backfill_labels.resolve_labels", return_value={}) as mock_resolve:
            backfill.main()
        mock_resolve.assert_called_once()
        assert fake_db.updated_labels == []

    def test_labels_updated(self, monkeypatch, fake_db):
        """Resolved labels trigger update_t_label for each well."""
        monkeypatch.setenv("MONITORING_DB_URI", "postgresql://test/db")
        fake_db.unlabeled_wells = [("P1", "K07"), ("P1", "K08")]
        labels = {("P1", "K07"): "ClassA", ("P1", "K08"): "ClassB"}
        with patch("monitoring.backfill_labels.DBLogger", return_value=fake_db), \
             patch("monitoring.backfill_labels.resolve_labels", return_value=labels):
            backfill.main()
        assert fake_db.updated_labels == [("P1", "K07", "ClassA"), ("P1", "K08", "ClassB")]

    def test_db_connection_failure(self, monkeypatch, fake_db):
        """If DB connect fails, main() returns without raising."""
        monkeypatch.setenv("MONITORING_DB_URI", "postgresql://test/db")
        fake_db.connect = lambda: (_ for _ in ()).throw(ConnectionError("refused"))
        with patch("monitoring.backfill_labels.DBLogger", return_value=fake_db):
            result = backfill.main()
        assert result is None

    def test_resolve_labels_called_with_wells(self, monkeypatch, fake_db):
        """resolve_labels receives the well list from fetch_unlabeled_wells."""
        monkeypatch.setenv("MONITORING_DB_URI", "postgresql://test/db")
        fake_db.unlabeled_wells = [("P1", "K07"), ("P1", "K08")]
        with patch("monitoring.backfill_labels.DBLogger", return_value=fake_db), \
             patch("monitoring.backfill_labels.resolve_labels", return_value={}) as mock_resolve:
            backfill.main()
        called_wells = mock_resolve.call_args[0][0]
        assert ("P1", "K07") in called_wells
        assert ("P1", "K08") in called_wells

    def test_close_called_in_finally(self, monkeypatch, fake_db):
        """DBLogger.close() is called even if resolve_labels raises."""
        monkeypatch.setenv("MONITORING_DB_URI", "postgresql://test/db")
        fake_db.unlabeled_wells = [("P1", "K07")]
        with patch("monitoring.backfill_labels.DBLogger", return_value=fake_db), \
             patch("monitoring.backfill_labels.resolve_labels", side_effect=RuntimeError("boom")), \
             pytest.raises(RuntimeError):
            backfill.main()
        assert fake_db.closed is True
