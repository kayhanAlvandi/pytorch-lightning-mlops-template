"""Benchmark script to compare transfer learning models with frozen backbones vs SimpleCNN baseline."""
import subprocess
import time


# Define experiment configurations
# Format: (model_config_name, description, extra_overrides, extra_tags)
EXPERIMENTS = [
    # Baseline - SimpleCNN (best performing config from previous benchmarks)
    ("simplecnn", "baseline_simplecnn", [], ["baseline", "simplecnn"]),
    
    # ResNet variants (frozen backbone - feature extraction)
    ("resnet18", "resnet18_frozen", ["model.freeze_backbone=true"], ["transfer_learning", "resnet", "frozen"]),
    ("resnet50", "resnet50_frozen", ["model.freeze_backbone=true"], ["transfer_learning", "resnet", "frozen"]),
    
    # EfficientNet variants (frozen backbone - feature extraction)
    ("efficientnet_b0", "efficientnet_b0_frozen", ["model.freeze_backbone=true"], ["transfer_learning", "efficientnet", "frozen"]),
    ("efficientnet_b3", "efficientnet_b3_frozen", ["model.freeze_backbone=true"], ["transfer_learning", "efficientnet", "frozen"]),
    
    # Vision Transformer variants (frozen backbone - feature extraction)
    # Note: ViT configs already have freeze_backbone=true by default
    ("vit_small", "vit_small_frozen", [], ["transfer_learning", "vit", "frozen"]),
    ("vit_base", "vit_base_frozen", [], ["transfer_learning", "vit", "frozen"]),
]

# Training configuration
MAX_EPOCHS = 300

# Early stopping configuration (uncomment in callbacks config to enable)
# This allows reducing training time if model converges early
EARLY_STOPPING_PATIENCE = 15


def run_training(model_config, experiment_name, extra_overrides, tags):
    """Run the training script with Hydra overrides."""
    print(f"\n{'='*80}")
    print(f"Starting training: {experiment_name}")
    print(f"Model config: {model_config}")
    print(f"{'='*80}\n")
    
    # Build Hydra command with overrides
    cmd = [
        "python", "train.py",
        f"model={model_config}",
        f"trainer.max_epochs={MAX_EPOCHS}",
        f"run_name={experiment_name}",
        f"tags=[{','.join(tags)}]",
    ] + extra_overrides
    
    print(f"Command: {' '.join(cmd)}")
    
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
    print("TRANSFER LEARNING BENCHMARK")
    print("Comparing frozen backbone transfer learning vs SimpleCNN baseline")
    print("=" * 80)
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Max epochs per experiment: {MAX_EPOCHS}")
    print(f"\nModels to benchmark:")
    for model_config, desc, _, tags in EXPERIMENTS:
        print(f"  - {desc} (model={model_config})")
    print("=" * 80)
    print("\nNOTE: To enable early stopping and reduce training time,")
    print("uncomment the EarlyStopping callback in configs/callbacks/default.yaml")
    print("=" * 80)
    
    results = []
    start_time = time.time()
    
    try:
        for i, (model_config, description, extra_overrides, extra_tags) in enumerate(EXPERIMENTS, 1):
            exp_name = f"tl_{i:02d}_{description}"
            tags = ["transfer_learning_benchmark"] + extra_tags
            
            print(f"\n\n{'#'*80}")
            print(f"EXPERIMENT {i}/{len(EXPERIMENTS)}: {exp_name}")
            print(f"  Model config: model={model_config}")
            print(f"  Extra overrides: {extra_overrides}")
            print(f"  Tags: {tags}")
            print(f"{'#'*80}")
            
            # Run training with Hydra overrides
            exp_start = time.time()
            success = run_training(model_config, exp_name, extra_overrides, tags)
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
        print("TRANSFER LEARNING BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"Total time: {total_duration/3600:.2f} hours")
        print(f"\nResults:")
        print(f"{'Experiment':<30} {'Model Config':<20} {'Status':<12} {'Time (min)':<12}")
        print("-" * 75)
        
        for r in results:
            status = "✓ Success" if r['success'] else "✗ Failed"
            print(f"{r['experiment']:<30} {r['model_config']:<20} {status:<12} {r['duration_minutes']:<12.1f}")
        
        successful = sum(1 for r in results if r['success'])
        print(f"\nCompleted: {successful}/{len(results)} experiments")
        print("\n✓ Check MLflow UI to compare results:")
        print("  mlflow ui")
        print("  http://localhost:5000")
        print("\nFilter by tag 'transfer_learning_benchmark' to see all runs from this benchmark.")


if __name__ == "__main__":
    main()
