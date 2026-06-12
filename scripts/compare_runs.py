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

# ANSI color codes
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colored(text: str, *styles: str) -> str:
    """Apply ANSI color codes to text."""
    return "".join(styles) + str(text) + Colors.RESET


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


def deep_diff(v1: Any, v2: Any, path: str = "") -> list:
    """Recursively compare two values and return list of differences.
    
    Returns list of tuples: (path, val1, val2) for changed values.
    """
    diffs = []
    
    if type(v1) != type(v2):
        # Different types - report as changed
        diffs.append((path, v1, v2))
    elif isinstance(v1, dict) and isinstance(v2, dict):
        # Compare dicts recursively
        all_keys = set(v1.keys()) | set(v2.keys())
        for k in sorted(all_keys):
            sub_path = f"{path}.{k}" if path else k
            if k not in v1:
                diffs.append((sub_path, "<missing>", v2[k]))
            elif k not in v2:
                diffs.append((sub_path, v1[k], "<missing>"))
            else:
                diffs.extend(deep_diff(v1[k], v2[k], sub_path))
    elif isinstance(v1, list) and isinstance(v2, list):
        # Compare lists - try to match by _target_ for callback-like dicts
        if all(isinstance(x, dict) and "_target_" in x for x in v1 + v2):
            # Match list items by _target_ key
            targets1 = {x.get("_target_"): x for x in v1}
            targets2 = {x.get("_target_"): x for x in v2}
            all_targets = set(targets1.keys()) | set(targets2.keys())
            for target in sorted(all_targets):
                # Use short name for path (last part of _target_)
                short_name = target.split(".")[-1] if target else "unknown"
                sub_path = f"{path}[{short_name}]"
                if target not in targets1:
                    diffs.append((sub_path, "<missing>", targets2[target]))
                elif target not in targets2:
                    diffs.append((sub_path, targets1[target], "<missing>"))
                else:
                    diffs.extend(deep_diff(targets1[target], targets2[target], sub_path))
        else:
            # Simple list comparison by index
            max_len = max(len(v1), len(v2))
            for i in range(max_len):
                sub_path = f"{path}[{i}]"
                if i >= len(v1):
                    diffs.append((sub_path, "<missing>", v2[i]))
                elif i >= len(v2):
                    diffs.append((sub_path, v1[i], "<missing>"))
                elif v1[i] != v2[i]:
                    if isinstance(v1[i], (dict, list)) and isinstance(v2[i], (dict, list)):
                        diffs.extend(deep_diff(v1[i], v2[i], sub_path))
                    else:
                        diffs.append((sub_path, v1[i], v2[i]))
    elif v1 != v2:
        diffs.append((path, v1, v2))
    
    return diffs


def compare_configs(cfg1: Dict[str, Any], cfg2: Dict[str, Any], name1: str, name2: str) -> Tuple[int, int, int]:
    """Compare two configs and print differences using deep recursive comparison."""
    # Use deep_diff for recursive comparison
    all_diffs = deep_diff(cfg1, cfg2)
    
    added = []      # In run2 but not run1
    removed = []    # In run1 but not run2
    changed = []    # Different values
    
    for path, v1, v2 in all_diffs:
        if v1 == "<missing>":
            added.append((path, v2))
        elif v2 == "<missing>":
            removed.append((path, v1))
        else:
            changed.append((path, v1, v2))
    
    # Print results (flush to clear any progress bar artifacts)
    sys.stdout.flush()
    print("\n" * 2)  # Clear space after progress bars
    
    # Header
    print(colored("═" * 70, Colors.BOLD, Colors.CYAN))
    print(colored("  CONFIG COMPARISON", Colors.BOLD, Colors.CYAN))
    print(colored(f"  {name1}", Colors.BLUE) + colored(" vs ", Colors.DIM) + colored(name2, Colors.HEADER))
    print(colored("═" * 70, Colors.BOLD, Colors.CYAN))
    
    if not added and not removed and not changed:
        print(colored("\n  ✓ Configs are identical!", Colors.GREEN, Colors.BOLD))
        return 0, 0, 0
    
    if changed:
        print(colored(f"\n  ⚡ CHANGED ({len(changed)})", Colors.YELLOW, Colors.BOLD))
        print(colored("  " + "─" * 50, Colors.DIM))
        for key, v1, v2 in changed:
            print(colored(f"    {key}", Colors.BOLD))
            print(colored(f"      ◀ ", Colors.RED) + colored(f"{name1}: ", Colors.DIM) + str(v1))
            print(colored(f"      ▶ ", Colors.GREEN) + colored(f"{name2}: ", Colors.DIM) + str(v2))
    
    if added:
        print(colored(f"\n  ✚ ADDED in {name2} ({len(added)})", Colors.GREEN, Colors.BOLD))
        print(colored("  " + "─" * 50, Colors.DIM))
        for key, v in added:
            print(colored(f"    {key}: ", Colors.BOLD) + colored(str(v), Colors.GREEN))
    
    if removed:
        print(colored(f"\n  ✖ REMOVED in {name2} ({len(removed)})", Colors.RED, Colors.BOLD))
        print(colored("  " + "─" * 50, Colors.DIM))
        for key, v in removed:
            print(colored(f"    {key}: ", Colors.BOLD) + colored(str(v), Colors.RED))
    
    # Summary
    print(colored("\n" + "═" * 70, Colors.BOLD, Colors.CYAN))
    summary_parts = []
    if changed:
        summary_parts.append(colored(f"{len(changed)} changed", Colors.YELLOW))
    if added:
        summary_parts.append(colored(f"{len(added)} added", Colors.GREEN))
    if removed:
        summary_parts.append(colored(f"{len(removed)} removed", Colors.RED))
    print(colored("  Summary: ", Colors.BOLD) + ", ".join(summary_parts))
    
    return len(changed), len(added), len(removed)


def main():
    parser = argparse.ArgumentParser(description="Compare Hydra configs between MLflow runs")
    parser.add_argument("run1", help="First run name or ID")
    parser.add_argument("run2", help="Second run name or ID")
    parser.add_argument("--run-id", action="store_true", help="Treat arguments as run IDs instead of names")
    parser.add_argument("--experiment", default="image_classifier", help="MLflow experiment name")
    parser.add_argument("--artifact", default="hydra_config.yaml", help="Config artifact filename")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db", help="MLflow tracking URI")
    
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
