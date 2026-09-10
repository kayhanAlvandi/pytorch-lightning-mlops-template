"""Supervised quality report: benchmark baseline vs. labeled production window.

Compares a served model's supervised quality on its fixed benchmark set (the
baseline, scored once by ``compute_predictions_references.py --target
benchmark``) against its quality on the labeled portion of a recent
production window (labels backfilled from MongoDB by backfill_labels.py).
Both sides compare the predicted label (``p_label``)
against the true label (``t_label``) with Evidently's ``ClassificationPreset``
(accuracy, F1, confusion matrix, ...).

The full Evidently HTML report (including the confusion matrix) is written under
``<reports_dir>/quality_<run>_<timestamp>/``; only top-line accuracy/F1 and
sample counts are persisted to the ``quality_report`` table.

Unlike drift, this is a supervised signal and only covers the fraction of the
window that already has ground truth. If no current-window rows are labeled yet,
the report is skipped cleanly (the common case before backfill catches up).
Image-level only for now (a tile-level label would need majority-vote
re-aggregation to be meaningful).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from database.dblogger import DBLogger
from monitoring.config import MonitoringSettings, resolve_run_id, resolve_window

MIN_CURRENT_SAMPLES = 20  # warn below this: metrics on a tiny labeled window are noisy


def _extract_quality(report) -> tuple[dict, dict]:
    """Pull (reference, current) quality dicts out of an Evidently report.

    Each returned dict has at least ``accuracy`` and ``f1``. Raises if the
    classification-quality metric isn't present.
    """
    result = report.as_dict()
    for metric in result.get("metrics", []):
        res = metric.get("result")
        if isinstance(res, dict) and isinstance(res.get("current"), dict) \
                and "accuracy" in res["current"]:
            current = res["current"]
            reference = res.get("reference") or {}
            return reference, current
    raise RuntimeError("Evidently report has no classification-quality result "
                       "(accuracy/f1). Check the ClassificationPreset output.")


def run_quality_report(
    benchmark_df: pd.DataFrame,
    current_df: pd.DataFrame,
    html_path: Path,
) -> tuple[dict, dict]:
    """Run Evidently's ClassificationPreset (benchmark as reference) and save HTML."""
    from evidently.metric_preset import ClassificationPreset
    from evidently.pipeline.column_mapping import ColumnMapping
    from evidently.report import Report

    mapping = ColumnMapping()
    mapping.target = "t_label"
    mapping.prediction = "p_label"

    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=benchmark_df, current_data=current_df, column_mapping=mapping)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(html_path))
    return _extract_quality(report)


def main():
    parser = argparse.ArgumentParser(description="Run a supervised quality report for a served model.")
    parser.add_argument("--run-id", default=None,
                        help="MLflow run_id of the served model. Defaults to reading it off "
                             "the running API's /model endpoint (see --api-url).")
    parser.add_argument("--api-url", default=None,
                        help="Base URL of the running API, used to resolve run_id from /model "
                             "when --run-id is not given (default: MONITORING_BASE_URL / "
                             "http://localhost:8000 from MonitoringSettings).")
    parser.add_argument("--window-days", type=float, default=7.0,
                        help="Look-back window size in days (default: 7). Ignored if "
                             "--window-start/--window-end are given.")
    parser.add_argument("--window-start", default=None, help="ISO timestamp (inclusive).")
    parser.add_argument("--window-end", default=None, help="ISO timestamp (exclusive).")
    parser.add_argument("--reports-dir", default="monitoring/reports",
                        help="Directory to write Evidently HTML/JSON reports into.")
    args = parser.parse_args()

    settings = MonitoringSettings()
    if not settings.has_db_uri:
        print("ERROR: No database URI configured. Set MONITORING_DB_URI.")
        return

    try:
        window_start, window_end = resolve_window(
            args.window_start, args.window_end, args.window_days
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    run_id = resolve_run_id(args.run_id, args.api_url or settings.base_url)
    if not run_id:
        print("ERROR: could not resolve a run_id. Pass --run-id, or ensure the API "
              "at --api-url is running and serving a model.")
        return

    print(f"Quality report for run_id={run_id}")
    print(f"Window: [{window_start}, {window_end})")

    db_logger = DBLogger(db_uri=settings.db_uri)
    try:
        db_logger.connect()
    except Exception as e:  # noqa: BLE001
        print(f"Failed to connect to database: {e}")
        return

    try:
        benchmark_df = pd.DataFrame(db_logger.fetch_benchmark_quality(run_id) or [])
        if benchmark_df.empty:
            print("ERROR: no benchmark predictions for this run_id. "
                  "Run 'make compute-predictions-references CMD=\"python -m "
                  "monitoring.compute_predictions_references --target benchmark\"' first.")
            return

        current_df = pd.DataFrame(db_logger.fetch_current_quality(run_id, window_start, window_end) or [])
        if current_df.empty:
            print("No labeled production predictions in the window yet -- nothing to compare. "
                  "Exiting (run backfill-labels once ground truth is available).")
            return

        n_benchmark = len(benchmark_df)
        n_current = len(current_df)
        if n_current < MIN_CURRENT_SAMPLES:
            print(f"WARNING: only {n_current} labeled current-window samples "
                  f"(< {MIN_CURRENT_SAMPLES}); quality metrics will be noisy.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.reports_dir) / f"quality_{run_id[:8]}_{ts}"
        html_path = out_dir / "report.html"
        reference, current = run_quality_report(benchmark_df, current_df, html_path)

        benchmark_acc = float(reference.get("accuracy"))
        benchmark_f1 = float(reference.get("f1"))
        current_acc = float(current.get("accuracy"))
        current_f1 = float(current.get("f1"))

        (out_dir / "metrics.json").write_text(json.dumps({
            "run_id": run_id,
            "window_start": str(window_start),
            "window_end": str(window_end),
            "n_benchmark_samples": n_benchmark,
            "n_current_samples": n_current,
            "benchmark": reference,
            "current": current,
        }, indent=2, default=str))

        report_path = str(out_dir)
        quality_report_id = db_logger.log_quality_report((
            run_id, window_start, window_end, n_benchmark, n_current,
            benchmark_acc, benchmark_f1, current_acc, current_f1, report_path,
        ))

        print(f"Done. benchmark: acc={benchmark_acc:.4f} f1={benchmark_f1:.4f} (n={n_benchmark}); "
              f"current: acc={current_acc:.4f} f1={current_f1:.4f} (n={n_current}). "
              f"quality_report.id={quality_report_id}, report_path={report_path}")
    finally:
        db_logger.close()


if __name__ == "__main__":
    main()
