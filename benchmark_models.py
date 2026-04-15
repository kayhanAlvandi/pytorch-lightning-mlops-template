"""Benchmark script to train multiple model configurations using Hydra overrides."""
import subprocess
import time


# Define experiment configurations using Hydra model config groups
# Format: (model_config_name, description, extra_tags)
EXPERIMENTS = [
    # Baseline (original architecture)
    ("simplecnn", "baseline", ["baseline"]),
    
    # Depth experiments (shallow vs deep)
    ("shallow", "shallow", ["depth_experiment"]),
    ("deep", "deep", ["depth_experiment"]),
    
    # Width experiments (narrow vs wide)
    ("narrow", "narrow", ["width_experiment"]),
    ("wide", "wide", ["width_experiment"]),
    
    # Channel multiplier experiments
    ("slow_growth", "slow_growth", ["multiplier_experiment"]),
    ("fast_growth", "fast_growth", ["multiplier_experiment"]),
    
    # Hidden dimension experiments
    ("small_classifier", "small_classifier", ["hidden_dim_experiment"]),
    ("large_classifier", "large_classifier", ["hidden_dim_experiment"]),
]

# Training configuration
MAX_EPOCHS = 300


def run_training(model_config, experiment_name, tags):
    """Run the training script with Hydra overrides."""
    print(f"\n{'='*80}")
    print(f"Starting training: {experiment_name}")
    print(f"{'='*80}\n")
    
    # Build Hydra command with overrides
    cmd = [
        "python", "train.py",
        f"model={model_config}",
        f"training.max_epochs={MAX_EPOCHS}",
        f"run_name={experiment_name}",
        f"tags=[{','.join(tags)}]",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
        )
        print(f"\n✓ Training completed successfully: {experiment_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed: {experiment_name}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Training interrupted by user: {experiment_name}")
        raise


def main():
    """Run all benchmark experiments."""
    print("=" * 80)
    print("MODEL ARCHITECTURE BENCHMARK")
    print("=" * 80)
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Max epochs per experiment: {MAX_EPOCHS}")
    print(f"Estimated total time: ~{len(EXPERIMENTS) * 2} hours (assuming 2h per experiment)")
    print("=" * 80)
    
    results = []
    start_time = time.time()
    
    try:
        for i, (model_config, description, extra_tags) in enumerate(EXPERIMENTS, 1):
            exp_name = f"exp{i:02d}_{description}"
            tags = ["benchmark", description] + extra_tags
            
            print(f"\n\n{'#'*80}")
            print(f"EXPERIMENT {i}/{len(EXPERIMENTS)}: {exp_name}")
            print(f"  Model config: model={model_config}")
            print(f"  Tags: {tags}")
            print(f"{'#'*80}")
            
            # Run training with Hydra overrides
            exp_start = time.time()
            success = run_training(model_config, exp_name, tags)
            exp_duration = time.time() - exp_start
            
            results.append({
                'experiment': exp_name,
                'model_config': model_config,
                'success': success,
                'duration_minutes': exp_duration / 60,
            })
            
            print(f"\nExperiment duration: {exp_duration/60:.1f} minutes")
            
            # Brief pause between experiments
            if i < len(EXPERIMENTS):
                print("\nWaiting 10 seconds before next experiment...")
                time.sleep(10)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user!")
    
    finally:
        # Print summary
        total_duration = time.time() - start_time
        print("\n\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"Total time: {total_duration/3600:.2f} hours")
        print(f"\nResults:")
        print(f"{'Experiment':<25} {'Model Config':<20} {'Status':<12} {'Time (min)':<12}")
        print("-" * 70)
        
        for r in results:
            status = "✓ Success" if r['success'] else "✗ Failed"
            print(f"{r['experiment']:<25} {r['model_config']:<20} {status:<12} {r['duration_minutes']:<12.1f}")
        
        successful = sum(1 for r in results if r['success'])
        print(f"\nCompleted: {successful}/{len(results)} experiments")
        print("\n✓ Check MLflow UI to compare results:")
        print("  mlflow ui")
        print("  http://localhost:5000")


if __name__ == "__main__":
    main()
