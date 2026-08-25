"""Drift report: compare live traffic against a model's reference set.

Pulls a served model's reference (validation-set) predictions and a window of
its live production predictions from Postgres, and runs Evidently drift
detection across three groups. Predicted label (``p_label``) is included as a
categorical column in the comparison (there's no ground truth in production
traffic, so drift is checked on the predicted label's distribution, not the
true one) -- it is not used to group/join/key the reference and current data,
which are compared as flat, row-per-sample DataFrames.

  * ``image_level``   -- categorical drift on ``p_label`` + numeric drift on
                         ``vote_fraction`` / ``avg_confidence``.
  * ``tile_level``    -- categorical drift on ``p_label`` + numeric drift on
                         per-tile ``confidence``.
  * ``channel_stats`` -- numeric input drift on per-channel pixel statistics
                         (``channel_<n>_<mean|std|p1|p5|p95|p99>``), pivoted
                         from the long ``tile_channel_stats`` rows to wide.

Each group is a separate Evidently ``Report`` (the three have different row
cardinalities -- images vs. tiles vs. tile-channels -- so they cannot share one
DataFrame). The per-group HTML reports plus a combined ``metrics.json`` are
written under ``<reports_dir>/drift_<run>_<timestamp>/``, and one ``drift_report``
row (+ its ``drift_report_column`` rows) is inserted into Postgres pointing at
that directory.

Drift here is an unsupervised early-warning signal, not an accuracy metric:
production traffic has no ground truth, which is why both sides are compared on
the predicted label rather than the true label.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from database.dblogger import DBLogger
from monitoring.config import MonitoringSettings

CHANNEL_STAT_COLS = ["mean", "std", "p1", "p5", "p95", "p99"]


def resolve_run_id(run_id: str | None, api_url: str) -> str | None:
    """Resolve the target serving model's MLflow run_id.

    An explicit ``run_id`` wins. Otherwise it's read off the running API's
    ``/model`` endpoint -- the API already loaded the model and knows its
    run_id, so the drift job doesn't need mlflow (or the tracking server) at
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


def pivot_channel_stats(rows: list[dict]) -> pd.DataFrame:
    """Long tile-channel stat rows -> one row per tile, wide channel columns.

    Input rows carry (tile_pred_id, channel, mean, std, p1, p5, p95, p99);
    output has one row per tile_pred_id with columns like ``channel_1_mean``,
    ``channel_1_p95``, ... Tiles missing channel stats are dropped.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=CHANNEL_STAT_COLS, how="all")
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(index="tile_pred_id", columns="channel", values=CHANNEL_STAT_COLS)
    wide.columns = [f"channel_{int(channel)}_{stat}" for stat, channel in wide.columns]
    return wide.sort_index(axis=1).reset_index(drop=True)


def _extract_drift(report) -> tuple[list[dict], int, int, bool]:
    """Pull per-column + dataset-level results out of an Evidently report.

    Returns (columns, n_drifted, n_total, dataset_drift) where columns is a
    list of {column_name, drift_score, drifted, stat_test}.
    """
    result = report.as_dict()
    table = None
    for metric in result.get("metrics", []):
        if isinstance(metric.get("result"), dict) and "drift_by_columns" in metric["result"]:
            table = metric["result"]
            break
    if table is None:
        raise RuntimeError("Evidently report has no drift_by_columns result")

    columns = [
        {
            "column_name": info["column_name"],
            "drift_score": float(info["drift_score"]),
            "drifted": bool(info["drift_detected"]),
            "stat_test": info.get("stattest_name"),
        }
        for info in table["drift_by_columns"].values()
    ]
    return (
        columns,
        int(table["number_of_drifted_columns"]),
        int(table["number_of_columns"]),
        bool(table["dataset_drift"]),
    )


def run_group_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical: list[str],
    categorical: list[str],
    html_path: Path,
) -> tuple[list[dict], int, int, bool]:
    """Run an Evidently DataDriftPreset for one column group and save its HTML."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.pipeline.column_mapping import ColumnMapping
    from evidently.report import Report

    mapping = ColumnMapping()
    mapping.numerical_features = numerical
    mapping.categorical_features = categorical
    # No ground truth in production traffic; keep drift purely feature-based.
    mapping.target = None
    mapping.prediction = None

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=mapping)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(html_path))
    return _extract_drift(report)


def _align(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to their shared columns (needed for channel stats,
    where reference and current could in principle see different channel sets)."""
    shared = [c for c in reference_df.columns if c in current_df.columns]
    return reference_df[shared], current_df[shared]


def main():
    parser = argparse.ArgumentParser(description="Run a drift report for a served model.")
    parser.add_argument("--run-id", default=None,
                        help="MLflow run_id of the served model. Defaults to reading it off "
                             "the running API's /model endpoint (see --api-url).")
    parser.add_argument("--api-url", default=None,
                        help="Base URL of the running API, used to resolve run_id from /model "
                             "when --run-id is not given (default: API_BASE_URL / "
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
        print("ERROR: No database URI configured. Set API_DB_URI.")
        return

    window_end = datetime.fromisoformat(args.window_end) if args.window_end else datetime.now()
    window_start = (
        datetime.fromisoformat(args.window_start)
        if args.window_start
        else window_end - timedelta(days=args.window_days)
    )
    if window_start >= window_end:
        print(f"ERROR: window_start ({window_start}) must be before window_end ({window_end}).")
        return

    run_id = resolve_run_id(args.run_id, args.api_url or settings.base_url)
    if not run_id:
        print("ERROR: could not resolve a run_id. Pass --run-id, or ensure the API "
              "at --api-url is running and serving a model.")
        return

    print(f"Drift report for run_id={run_id}")
    print(f"Window: [{window_start}, {window_end})")

    db_logger = DBLogger(db_uri=settings.db_uri)
    try:
        db_logger.connect()
    except Exception as e:  # noqa: BLE001
        print(f"Failed to connect to database: {e}")
        return

    try:
        # ── Pull reference + current data for each group ──────────────────
        ref_img = pd.DataFrame(db_logger.fetch_reference_image_level(run_id) or [])
        cur_img = pd.DataFrame(db_logger.fetch_current_image_level(run_id, window_start, window_end) or [])

        ref_tile_rows = db_logger.fetch_reference_tile_level(run_id) or []
        cur_tile_rows = db_logger.fetch_current_tile_level(run_id, window_start, window_end) or []
        ref_tile = pd.DataFrame(ref_tile_rows)
        cur_tile = pd.DataFrame(cur_tile_rows)

        if ref_img.empty or ref_tile.empty:
            print("ERROR: no reference data for this run_id. Run compute-reference first.")
            return
        if cur_img.empty and cur_tile.empty:
            print("No live predictions in the window -- nothing to compare. Exiting.")
            return

        # Build the (up to) three group DataFrames.
        groups: dict[str, dict] = {}

        if not cur_img.empty:
            groups["image_level"] = {
                "reference": ref_img[["p_label", "vote_fraction", "avg_confidence"]],
                "current": cur_img[["p_label", "vote_fraction", "avg_confidence"]],
                "numerical": ["vote_fraction", "avg_confidence"],
                "categorical": ["p_label"],
            }

        if not cur_tile.empty:
            ref_conf = ref_tile[["tile_pred_id", "p_label", "confidence"]].drop_duplicates("tile_pred_id")
            cur_conf = cur_tile[["tile_pred_id", "p_label", "confidence"]].drop_duplicates("tile_pred_id")
            groups["tile_level"] = {
                "reference": ref_conf[["p_label", "confidence"]].reset_index(drop=True),
                "current": cur_conf[["p_label", "confidence"]].reset_index(drop=True),
                "numerical": ["confidence"],
                "categorical": ["p_label"],
            }

            ref_chan = pivot_channel_stats(ref_tile_rows)
            cur_chan = pivot_channel_stats(cur_tile_rows)
            if not ref_chan.empty and not cur_chan.empty:
                ref_chan, cur_chan = _align(ref_chan, cur_chan)
                if not ref_chan.columns.empty:
                    groups["channel_stats"] = {
                        "reference": ref_chan,
                        "current": cur_chan,
                        "numerical": list(ref_chan.columns),
                        "categorical": [],
                    }

        # ── Output directory ──────────────────────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.reports_dir) / f"drift_{run_id[:8]}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Run each group's report ───────────────────────────────────────
        all_columns: list[tuple] = []       # for DB insert
        metrics_summary: dict = {}          # for metrics.json
        total_drifted = 0
        total_columns = 0
        dataset_drift = False

        for group_name, g in groups.items():
            html_path = out_dir / f"{group_name}.html"
            columns, n_drifted, n_total, grp_drift = run_group_report(
                g["reference"], g["current"], g["numerical"], g["categorical"], html_path,
            )
            print(f"  [{group_name}] {n_drifted}/{n_total} columns drifted "
                  f"(dataset_drift={grp_drift}) -> {html_path.name}")

            total_drifted += n_drifted
            total_columns += n_total
            dataset_drift = dataset_drift or grp_drift
            metrics_summary[group_name] = {
                "n_columns_drifted": n_drifted,
                "n_columns_total": n_total,
                "dataset_drift": grp_drift,
                "columns": columns,
            }
            for col in columns:
                all_columns.append((
                    None,  # drift_report_id filled in after the parent insert
                    col["column_name"],
                    group_name,
                    col["drift_score"],
                    col["drifted"],
                    col["stat_test"],
                ))

        if total_columns == 0:
            print("No comparable columns across any group -- nothing to report.")
            return

        (out_dir / "metrics.json").write_text(json.dumps(metrics_summary, indent=2, default=str))

        # ── Persist to Postgres ───────────────────────────────────────────
        report_path = str(out_dir)
        drift_report_id = db_logger.log_drift_report((
            run_id, window_start, window_end, dataset_drift,
            total_drifted, total_columns, report_path,
        ))
        rows = [(drift_report_id, *col[1:]) for col in all_columns]
        db_logger.log_drift_report_column(rows)

        print(f"Done. dataset_drift={dataset_drift}, {total_drifted}/{total_columns} "
              f"columns drifted. drift_report.id={drift_report_id}, report_path={report_path}")
    finally:
        db_logger.close()


if __name__ == "__main__":
    main()
