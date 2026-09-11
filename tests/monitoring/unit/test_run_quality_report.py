"""Unit tests for monitoring/run_quality_report.py.

Covers _extract_quality with fake report objects and run_quality_report with
a real (tiny) Evidently ClassificationPreset to verify the API contract.
"""
from __future__ import annotations

import pandas as pd
import pytest

from monitoring.run_quality_report import _extract_quality, run_quality_report

# ─────────────────────────────────────────────────────────────────────────────
# _extract_quality
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractQuality:
    def _fake_report(self, reference, current):
        return type("FakeReport", (), {
            "as_dict": lambda self: {
                "metrics": [{
                    "result": {
                        "reference": reference,
                        "current": current,
                    }
                }]
            }
        })()

    def test_basic_extraction(self):
        report = self._fake_report(
            reference={"accuracy": 0.85, "f1": 0.80},
            current={"accuracy": 0.70, "f1": 0.65},
        )
        reference, current = _extract_quality(report)
        assert reference["accuracy"] == 0.85
        assert reference["f1"] == 0.80
        assert current["accuracy"] == 0.70
        assert current["f1"] == 0.65

    def test_no_reference_defaults_to_empty(self):
        """If the metric has no reference dict, an empty dict is returned."""
        report = self._fake_report(
            reference=None,
            current={"accuracy": 0.70, "f1": 0.65},
        )
        reference, current = _extract_quality(report)
        assert reference == {}
        assert current["accuracy"] == 0.70

    def test_no_classification_result_raises(self):
        report = type("FakeReport", (), {
            "as_dict": lambda self: {"metrics": [{"result": {"other": 1}}]}
        })()
        with pytest.raises(RuntimeError, match="no classification-quality"):
            _extract_quality(report)

    def test_empty_metrics_raises(self):
        report = type("FakeReport", (), {
            "as_dict": lambda self: {"metrics": []}
        })()
        with pytest.raises(RuntimeError, match="no classification-quality"):
            _extract_quality(report)

    def test_picks_first_matching_metric(self):
        """If multiple metrics exist, the first with 'accuracy' in current wins."""
        report = type("FakeReport", (), {
            "as_dict": lambda self: {
                "metrics": [
                    {"result": {"current": {"other": 1}}},
                    {"result": {
                        "reference": {"accuracy": 0.9, "f1": 0.88},
                        "current": {"accuracy": 0.75, "f1": 0.70},
                    }},
                ]
            }
        })()
        reference, current = _extract_quality(report)
        assert reference["accuracy"] == 0.9
        assert current["accuracy"] == 0.75


# ─────────────────────────────────────────────────────────────────────────────
# run_quality_report (real Evidently on tiny data)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunQualityReport:
    def test_small_dataset(self, tmp_path):
        """Run a real ClassificationPreset on 20 rows, verify HTML + metrics."""
        pytest.importorskip("evidently")
        benchmark_df = pd.DataFrame({
            "p_label": ["A"] * 8 + ["B"] * 12,
            "t_label": ["A"] * 7 + ["B"] * 12 + ["A"],
        })
        current_df = pd.DataFrame({
            "p_label": ["A"] * 9 + ["B"] * 11,
            "t_label": ["A"] * 6 + ["B"] * 11 + ["A"] * 2 + ["B"],
        })
        html_path = tmp_path / "quality" / "report.html"
        reference, current = run_quality_report(benchmark_df, current_df, html_path)

        assert html_path.exists()
        assert html_path.stat().st_size > 0
        assert "accuracy" in reference
        assert "f1" in reference
        assert "accuracy" in current
        assert "f1" in current
        assert 0.0 <= float(reference["accuracy"]) <= 1.0
        assert 0.0 <= float(current["accuracy"]) <= 1.0

    def test_perfect_prediction(self, tmp_path):
        """When predictions match labels perfectly, accuracy should be 1.0."""
        pytest.importorskip("evidently")
        df = pd.DataFrame({
            "p_label": ["A"] * 10 + ["B"] * 10,
            "t_label": ["A"] * 10 + ["B"] * 10,
        })
        html_path = tmp_path / "perfect" / "report.html"
        reference, current = run_quality_report(df, df, html_path)
        assert float(current["accuracy"]) == 1.0
        assert float(reference["accuracy"]) == 1.0
