#!/usr/bin/env python
"""Compare Hydra config artifacts between two MLflow runs.

Usage:
    python scripts/compare_runs.py <run_name_1> <run_name_2>
    python scripts/compare_runs.py --run-id <run_id_1> <run_id_2>
    
Examples:
    python scripts/compare_runs.py "efficientnet_b3_v1" "efficientnet_b3_v2"
    python scripts/compare_runs.py --run-id abc123 def456
"""
import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from mlflow.tracking import MlflowClient


def get_run_by_name(client: MlflowClient, run_name: str, experiment_name: str = "image_classifier") -> Optional[str]:
    """Find run ID by run name."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )
    
    if not runs:
        print(f"Run with name '{run_name}' not found")
        return None
    
    return runs[0].info.run_id


def load_config_artifact(client: MlflowClient, run_id: str, artifact_name: str = "config.yaml", temp_dir: str = None) -> Dict[str, Any]:
    """Download and load config artifact from a run to a temp directory."""
    try:
        local_path = client.download_artifacts(run_id, artifact_name, dst_path=temp_dir)
        with open(local_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config from run {run_id}: {e}")
        return {}


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dict with dot notation keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compare_configs(cfg1: Dict[str, Any], cfg2: Dict[str, Any], name1: str, name2: str) -> Tuple[int, int, int]:
    """Compare two configs and print differences."""
    flat1 = flatten_dict(cfg1)
    flat2 = flatten_dict(cfg2)
    
    all_keys = set(flat1.keys()) | set(flat2.keys())
    
    added = []      # In run2 but not run1
    removed = []    # In run1 but not run2
    changed = []    # Different values
    
    for key in sorted(all_keys):
        v1 = flat1.get(key)
        v2 = flat2.get(key)
        
        if v1 is None and v2 is not None:
            added.append((key, v2))
        elif v1 is not None and v2 is None:
            removed.append((key, v1))
        elif v1 != v2:
            changed.append((key, v1, v2))
    
    # Print results (flush to clear any progress bar artifacts)
    sys.stdout.flush()
    print("\n" * 2)  # Clear space after progress bars
    print("=" * 70)
    print(f"Config Comparison: {name1} vs {name2}")
    print("=" * 70)
    
    if not added and not removed and not changed:
        print("\n✓ Configs are identical!")
        return 0, 0, 0
    
    if changed:
        print(f"\n[CHANGED] ({len(changed)} parameters)")
        print("-" * 40)
        for key, v1, v2 in changed:
            print(f"  {key}:")
            print(f"    {name1}: {v1}")
            print(f"    {name2}: {v2}")
    
    if added:
        print(f"\n[ADDED in {name2}] ({len(added)} parameters)")
        print("-" * 40)
        for key, v in added:
            print(f"  {key}: {v}")
    
    if removed:
        print(f"\n[REMOVED in {name2}] ({len(removed)} parameters)")
        print("-" * 40)
        for key, v in removed:
            print(f"  {key}: {v}")
    
    print("\n" + "=" * 70)
    print(f"Summary: {len(changed)} changed, {len(added)} added, {len(removed)} removed")
    
    return len(changed), len(added), len(removed)


def main():
    parser = argparse.ArgumentParser(description="Compare Hydra configs between MLflow runs")
    parser.add_argument("run1", help="First run name or ID")
    parser.add_argument("run2", help="Second run name or ID")
    parser.add_argument("--run-id", action="store_true", help="Treat arguments as run IDs instead of names")
    parser.add_argument("--experiment", default="image_classifier", help="MLflow experiment name")
    parser.add_argument("--artifact", default="hydra_config.yaml", help="Config artifact filename")
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    # Initialize client
    client = MlflowClient(tracking_uri=args.tracking_uri)
    
    # Get run IDs
    if args.run_id:
        run_id_1, run_id_2 = args.run1, args.run2
        name1, name2 = args.run1[:8], args.run2[:8]  # Truncate for display
    else:
        run_id_1 = get_run_by_name(client, args.run1, args.experiment)
        run_id_2 = get_run_by_name(client, args.run2, args.experiment)
        name1, name2 = args.run1, args.run2
        
        if not run_id_1 or not run_id_2:
            sys.exit(1)
    
    # Load configs in temp directory (auto-cleaned after use)
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg1 = load_config_artifact(client, run_id_1, args.artifact, temp_dir)
        cfg2 = load_config_artifact(client, run_id_2, args.artifact, temp_dir)
        
        if not cfg1 or not cfg2:
            sys.exit(1)
        
        # Compare
        compare_configs(cfg1, cfg2, name1, name2)


if __name__ == "__main__":
    main()
