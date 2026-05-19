"""
Compare MLflow runs: extract differing parameters and key metrics.

Usage:
    python scripts/compare_Mlflow_params.py --run-names "run1" "run2" "run3"
    python scripts/compare_Mlflow_params.py --run-ids "abc123" "def456"
    python scripts/compare_Mlflow_params.py --all

    # Filter with MLflow SQL syntax (params., metrics., tags., attributes.)
    python scripts/compare_Mlflow_params.py --all --filter "params.model/backbone_name = 'vit_small'"
    python scripts/compare_Mlflow_params.py --all --filter "metrics.`val/acc` > 0.6"
    python scripts/compare_Mlflow_params.py --all --filter "params.dataset_name LIKE '%moredata%'"

    # Convenience shortcuts (combined with AND)
    python scripts/compare_Mlflow_params.py --all --model vit_small
    python scripts/compare_Mlflow_params.py --all --dataset 4wells_per_label_16samples
    python scripts/compare_Mlflow_params.py --all --status FINISHED
    python scripts/compare_Mlflow_params.py --all --model vit_small --min-val-acc 0.5

    # Combine filter + shortcuts
    python scripts/compare_Mlflow_params.py --all --model vit_small --filter "metrics.`val/acc` > 0.6"

Output: Table with run names as rows, differing params + metrics as columns.
"""

import argparse
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Defaults
TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "image_classifier"


def build_filter_string(filter_str=None, model=None, dataset=None, status=None, min_val_acc=None):
    """Build MLflow filter string from convenience args + raw filter."""
    parts = []
    if filter_str:
        parts.append(filter_str)
    if model:
        parts.append(f"params.`model/backbone_name` = '{model}'")
    if dataset:
        parts.append(f"params.dataset_name LIKE '%{dataset}%'")
    if status:
        parts.append(f"attributes.status = '{status}'")
    if min_val_acc is not None:
        parts.append(f"metrics.`val/acc` > {min_val_acc}")
    return " AND ".join(parts) if parts else None


def get_runs(experiment_id, run_names=None, run_ids=None, all_runs=False,
             filter_string=None):
    """Fetch runs by name, ID, or all, with optional MLflow SQL filter."""
    if all_runs:
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string or "",
            order_by=["start_time DESC"],
        )
    elif run_ids:
        filter_parts = [f"run_id = '{rid}'" for rid in run_ids]
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=" or ".join(filter_parts),
        )
    elif run_names:
        frames = []
        for name in run_names:
            name_filter = f"run_name = '{name}'"
            if filter_string:
                name_filter = f"{name_filter} AND {filter_string}"
            r = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string=name_filter,
                max_results=1,
            )
            if not r.empty:
                frames.append(r)
            else:
                print(f"WARNING: Run '{name}' not found, skipping.")
        if not frames:
            print("No runs found.")
            sys.exit(1)
        runs = pd.concat(frames, ignore_index=True)
    else:
        print("Provide --run-names, --run-ids, or --all")
        sys.exit(1)

    if runs.empty:
        print("No runs found.")
        sys.exit(1)

    return runs


def extract_params_and_metrics(runs_df):
    """Extract params and metrics into a clean DataFrame indexed by run name."""
    # Identify param and metric columns
    param_cols = [c for c in runs_df.columns if c.startswith("params.")]
    metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]

    # Build a DataFrame with run_name as index
    records = []
    for _, row in runs_df.iterrows():
        run_name = row.get("tags.mlflow.runName") if "tags.mlflow.runName" in row.index else None
        run_id = row["run_id"]
        if pd.isna(run_name) or run_name is None:
            run_name = run_id[:8]
        record = {"run_id": run_id, "run_name": run_name}

        # All params
        for col in param_cols:
            param_name = col.replace("params.", "")
            record[param_name] = row[col]

        # Key metrics
        for col in metric_cols:
            metric_name = col.replace("metrics.", "")
            record[metric_name] = row[col]

        records.append(record)

    df = pd.DataFrame(records).set_index("run_id")

    ## clean params column names
    param_cols = [col.replace("params.", "") for col in param_cols]
    if "run_name" not in param_cols:
        param_cols.append("run_name")
    return df, param_cols, metric_cols


def find_differing_params(df, param_names):
    """Find params that differ across runs (or are missing in some)."""
    differing = []
    for param in param_names:
        values = df[param].dropna().unique()
        has_missing = df[param].isna().any()
        if len(values) > 1 or has_missing:
            differing.append(param)
    return differing


def get_best_metrics_from_history(run_ids_and_names):
    """For each run, find the step with max val/acc, then get val/f1 and train/acc at that step.
    
    Returns a DataFrame indexed by run_name with columns:
        best_val_acc, best_val_f1, train_acc_at_best, best_step
    """
    client = MlflowClient()
    records = []

    for run_id, run_name in run_ids_and_names:
        record = {"run_id": run_id}

        # Get val/acc history
        try:
            val_acc_history = client.get_metric_history(run_id, "val/acc")
        except Exception:
            val_acc_history = []

        if not val_acc_history:
            record["best_val_acc"] = None
            record["best_val_f1"] = None
            record["train_acc_at_best"] = None
            record["best_step"] = None
            records.append(record)
            continue

        # Find step with max val/acc
        best_metric = max(val_acc_history, key=lambda m: m.value)
        best_step = best_metric.step
        record["best_val_acc"] = best_metric.value
        record["best_step"] = best_step

        # Get val/f1 at that step
        try:
            val_f1_history = client.get_metric_history(run_id, "val/f1")
            f1_at_step = [m for m in val_f1_history if m.step == best_step]
            record["best_val_f1"] = f1_at_step[0].value if f1_at_step else None
        except Exception:
            record["best_val_f1"] = None

        # Get train/acc at that step
        try:
            train_acc_history = client.get_metric_history(run_id, "train/acc")
            # train/acc might be logged per step or per epoch; find closest step <= best_step
            train_at_step = [m for m in train_acc_history if m.step == best_step]
            if train_at_step:
                record["train_acc_at_best"] = train_at_step[0].value
            else:
                # Find the closest step that is <= best_step
                candidates = [m for m in train_acc_history if m.step <= best_step]
                if candidates:
                    record["train_acc_at_best"] = max(candidates, key=lambda m: m.step).value
                else:
                    record["train_acc_at_best"] = None
        except Exception:
            record["train_acc_at_best"] = None
        

        records.append(record)

    return pd.DataFrame(records).set_index("run_id")


def main():
    parser = argparse.ArgumentParser(description="Compare MLflow runs")
    parser.add_argument("--run-names", nargs="+", help="Run names to compare")
    parser.add_argument("--run-ids", nargs="+", help="Run IDs to compare")
    parser.add_argument("--all", action="store_true", help="Compare all runs")
    parser.add_argument("--filter", type=str, default=None,
                        help="MLflow SQL filter, e.g. \"params.`model/backbone_name` = 'vit_small'\"")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model backbone name (e.g. vit_small, efficientnet_b3)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter by dataset name (substring match)")
    parser.add_argument("--status", type=str, default=None,
                        help="Filter by run status (FINISHED, FAILED, RUNNING)")
    parser.add_argument("--min-val-acc", type=float, default=None,
                        help="Filter runs with val/acc above this threshold")
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--output", default="run_comparison.csv", help="Output CSV file path")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        print(f"Experiment '{args.experiment}' not found")
        sys.exit(1)

    # Build filter from convenience args + raw filter
    combined_filter = build_filter_string(
        filter_str=args.filter,
        model=args.model,
        dataset=args.dataset,
        status=args.status,
        min_val_acc=args.min_val_acc,
    )
    if combined_filter:
        print(f"Filter: {combined_filter}\n")

    # Fetch runs
    runs_df = get_runs(
        experiment.experiment_id,
        run_names=args.run_names,
        run_ids=args.run_ids,
        all_runs=args.all,
        filter_string=combined_filter,
    )
    print(f"Found {len(runs_df)} runs\n")

    # Extract all params and metrics
    df, param_cols, metric_cols = extract_params_and_metrics(runs_df)

    # Find differing params
    differing_params = find_differing_params(df, param_cols)
    print(f"Parameters that differ across runs ({len(differing_params)}):")
    for p in differing_params:
        print(f"  - {p}")
    print()

    # Get best metrics from metric history (step with max val/acc)
    print("Fetching metric histories to find best step per run...")
    run_ids_and_names = []
    for _, row in runs_df.iterrows():
        name = row.get("tags.mlflow.runName") if "tags.mlflow.runName" in row.index else None
        if pd.isna(name) or name is None:
            name = row["run_id"][:8]
        run_ids_and_names.append((row["run_id"], name))
    best_metrics_df = get_best_metrics_from_history(run_ids_and_names)

    # Merge best metrics into main df
    df = df.join(best_metrics_df, how="left")

    # Metrics columns to show
    available_metrics = [m for m in ["best_val_acc", "best_val_f1", "train_acc_at_best", "best_step"]
                         if m in df.columns]

    # Build final comparison table
    if "run_name" not in differing_params:
        columns_to_show = ["run_name"] + differing_params + available_metrics
    else:
        columns_to_show = differing_params + available_metrics
    columns_to_show = [c for c in columns_to_show if c in df.columns]

    result = df[columns_to_show].copy()

    # Sort by val accuracy descending if available
    sort_col = next((c for c in ["val/acc", "best_val_acc"] if c in result.columns), None)
    if sort_col:
        result[sort_col] = pd.to_numeric(result[sort_col], errors="coerce")
        result = result.sort_values(sort_col, ascending=False)

    # Display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 30)
    print("=" * 80)
    print("RUN COMPARISON TABLE")
    print("=" * 80)
    print(result.to_string())
    print()

    # Also save to CSV
    output_path = Path(args.output)
    result.to_csv(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
