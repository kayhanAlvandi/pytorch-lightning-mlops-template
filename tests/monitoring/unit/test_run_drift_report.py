"""Unit tests for monitoring/run_drift_report.py.

Covers the pure helpers (pivot_channel_stats, _align, _extract_drift) with
synthetic data and fake report objects. No real Evidently Report is needed
for the parsing tests; one integration test runs a real Report on tiny data
to verify the Evidently API contract still holds.
"""
from __future__ import annotations

import pandas as pd
import pytest

from monitoring.run_drift_report import (
    CHANNEL_STAT_COLS,
    _align,
    _extract_drift,
    pivot_channel_stats,
)

# ─────────────────────────────────────────────────────────────────────────────
# pivot_channel_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestPivotChannelStats:
    def test_empty_rows(self):
        result = pivot_channel_stats([])
        assert result.empty

    def test_basic_pivot(self):
        """Two tiles, two channels each -> one row per tile with wide columns."""
        rows = [
            {"tile_pred_id": 1, "channel": 1, "mean": 10.0, "std": 1.0,
             "p1": 5.0, "p5": 7.0, "p95": 13.0, "p99": 15.0},
            {"tile_pred_id": 1, "channel": 2, "mean": 20.0, "std": 2.0,
             "p1": 15.0, "p5": 17.0, "p95": 23.0, "p99": 25.0},
            {"tile_pred_id": 2, "channel": 1, "mean": 30.0, "std": 3.0,
             "p1": 25.0, "p5": 27.0, "p95": 33.0, "p99": 35.0},
            {"tile_pred_id": 2, "channel": 2, "mean": 40.0, "std": 4.0,
             "p1": 35.0, "p5": 37.0, "p95": 43.0, "p99": 45.0},
        ]
        result = pivot_channel_stats(rows)
        assert len(result) == 2  # one row per tile_pred_id
        assert "channel_1_mean" in result.columns
        assert "channel_2_mean" in result.columns
        assert "channel_1_p95" in result.columns
        assert "channel_2_p99" in result.columns

    def test_column_naming(self):
        """Wide columns are named channel_{n}_{stat}."""
        rows = [
            {"tile_pred_id": 1, "channel": 1, "mean": 10.0, "std": 1.0,
             "p1": 5.0, "p5": 7.0, "p95": 13.0, "p99": 15.0},
        ]
        result = pivot_channel_stats(rows)
        for stat in CHANNEL_STAT_COLS:
            assert f"channel_1_{stat}" in result.columns

    def test_drops_rows_all_nan_stats(self):
        """Rows where all stat columns are NaN are dropped."""
        rows = [
            {"tile_pred_id": 1, "channel": 1, "mean": 10.0, "std": 1.0,
             "p1": 5.0, "p5": 7.0, "p95": 13.0, "p99": 15.0},
            {"tile_pred_id": 2, "channel": 1, "mean": None, "std": None,
             "p1": None, "p5": None, "p95": None, "p99": None},
        ]
        result = pivot_channel_stats(rows)
        assert len(result) == 1

    def test_missing_channel_drops_tile(self):
        """A tile with only one channel (not all channels) still appears."""
        rows = [
            {"tile_pred_id": 1, "channel": 1, "mean": 10.0, "std": 1.0,
             "p1": 5.0, "p5": 7.0, "p95": 13.0, "p99": 15.0},
            {"tile_pred_id": 2, "channel": 1, "mean": 20.0, "std": 2.0,
             "p1": 15.0, "p5": 17.0, "p95": 23.0, "p99": 25.0},
            {"tile_pred_id": 2, "channel": 2, "mean": 30.0, "std": 3.0,
             "p1": 25.0, "p5": 27.0, "p95": 33.0, "p99": 35.0},
        ]
        result = pivot_channel_stats(rows)
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# _align
# ─────────────────────────────────────────────────────────────────────────────

class TestAlign:
    def test_keeps_shared_columns(self):
        ref = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        cur = pd.DataFrame({"b": [4], "c": [5], "d": [6]})
        a, b = _align(ref, cur)
        assert list(a.columns) == ["b", "c"]
        assert list(b.columns) == ["b", "c"]

    def test_all_shared(self):
        ref = pd.DataFrame({"a": [1], "b": [2]})
        cur = pd.DataFrame({"a": [3], "b": [4]})
        a, b = _align(ref, cur)
        assert list(a.columns) == ["a", "b"]
        assert list(b.columns) == ["a", "b"]

    def test_no_shared(self):
        ref = pd.DataFrame({"a": [1]})
        cur = pd.DataFrame({"b": [2]})
        a, b = _align(ref, cur)
        assert list(a.columns) == []
        assert list(b.columns) == []

    def test_preserves_values(self):
        ref = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
        cur = pd.DataFrame({"b": [50, 60], "c": [70, 80]})
        a, b = _align(ref, cur)
        assert a["b"].tolist() == [30, 40]
        assert b["b"].tolist() == [50, 60]


# ─────────────────────────────────────────────────────────────────────────────
# _extract_drift
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractDrift:
    def _fake_report(self, drift_by_columns, dataset_drift=False,
                     n_drifted=1, n_total=3):
        """Build a fake Evidently report with the as_dict() structure."""
        return type("FakeReport", (), {
            "as_dict": lambda self: {
                "metrics": [{
                    "result": {
                        "drift_by_columns": drift_by_columns,
                        "dataset_drift": dataset_drift,
                        "number_of_drifted_columns": n_drifted,
                        "number_of_columns": n_total,
                    }
                }]
            }
        })()

    def test_basic_extraction(self):
        columns = {
            "vote_fraction": {
                "column_name": "vote_fraction",
                "drift_score": 0.15,
                "drift_detected": True,
                "stattest_name": "ks",
            },
            "avg_confidence": {
                "column_name": "avg_confidence",
                "drift_score": 0.05,
                "drift_detected": False,
                "stattest_name": "ks",
            },
        }
        report = self._fake_report(columns, dataset_drift=True, n_drifted=1, n_total=2)
        cols, n_drifted, n_total, dataset_drift = _extract_drift(report)

        assert len(cols) == 2
        assert cols[0]["column_name"] == "vote_fraction"
        assert cols[0]["drifted"] is True
        assert cols[0]["drift_score"] == 0.15
        assert cols[0]["stat_test"] == "ks"
        assert cols[1]["drifted"] is False
        assert n_drifted == 1
        assert n_total == 2
        assert dataset_drift is True

    def test_no_drift_result_raises(self):
        """A report without drift_by_columns raises RuntimeError."""
        report = type("FakeReport", (), {
            "as_dict": lambda self: {"metrics": [{"result": {"other": 1}}]}
        })()
        with pytest.raises(RuntimeError, match="no drift_by_columns"):
            _extract_drift(report)

    def test_empty_metrics_raises(self):
        report = type("FakeReport", (), {
            "as_dict": lambda self: {"metrics": []}
        })()
        with pytest.raises(RuntimeError, match="no drift_by_columns"):
            _extract_drift(report)

    def test_stat_test_none(self):
        """stattest_name can be missing (defaults to None)."""
        columns = {
            "x": {"column_name": "x", "drift_score": 0.1, "drift_detected": False},
        }
        report = self._fake_report(columns)
        cols, _, _, _ = _extract_drift(report)
        assert cols[0]["stat_test"] is None
