"""Quick benchmark script with specific configurations."""
import subprocess
import sys
from pathlib import Path
import yaml
import time

# Define your specific experiment configurations
# Format: (num_blocks, base_channels, channel_multiplier, hidden_dim, description)
EXPERIMENTS = [
    (2, 32, 2, 128, "shallow_vs_deep"),
    (4, 64, 2, 128, "wide_vs_narrow"),
    (4, 32, 3, 128, "fast_growth"),
    (4, 32, 2, 64, "small_classifier"),
]

# Training configuration
MAX_EPOCHS = 300
CONFIG_FILE = "config/config.yaml"
BACKUP_CONFIG = "config/config_backup.yaml"


def backup_config():
    """Backup the original config file."""
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    with open(BACKUP_CONFIG, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✓ Backed up config to {BACKUP_CONFIG}")


def restore_config():
    """Restore the original config file."""
    if Path(BACKUP_CONFIG).exists():
        with open(BACKUP_CONFIG, 'r') as f:
            config = yaml.safe_load(f)
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✓ Restored config from {BACKUP_CONFIG}")


def update_config(num_blocks, base_channels, channel_multiplier, hidden_dim):
    """Update the config file with new model parameters."""
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update model parameters
    config['model']['num_blocks'] = num_blocks
    config['model']['base_channels'] = base_channels
    config['model']['channel_multiplier'] = float(channel_multiplier)
    config['model']['hidden_dim'] = hidden_dim
    
    # Update max epochs
    config['training']['max_epochs'] = MAX_EPOCHS
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_training(experiment_name, tags=None):
    """Run the training script."""
    print(f"\n{'='*80}")
    print(f"Starting training: {experiment_name}")
    print(f"{'='*80}\n")
    
    # Run training (assumes conda environment is already activated)
    cmd = ["python", "train.py", "--config", CONFIG_FILE, "--run-name", experiment_name]
    
    # Add tags if provided
    if tags:
        cmd.extend(["--tags"] + tags)
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
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
    print("="*80)
    print("QUICK MODEL BENCHMARK")
    print("="*80)
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print("="*80)
    
    # Show all experiments
    print("\nExperiments to run:")
    for i, (nb, bc, cm, hd, desc) in enumerate(EXPERIMENTS, 1):
        channels = [int(bc * (cm ** j)) for j in range(nb)]
        print(f"  {i}. {desc}: blocks={nb}, base_ch={bc}, mult={cm}, hidden={hd}")
        print(f"     Channels: {' → '.join(map(str, channels))}")
    
    input("\nPress Enter to start benchmark (or Ctrl+C to cancel)...")
    
    # Backup original config
    backup_config()
    
    results = []
    start_time = time.time()
    
    try:
        for i, (num_blocks, base_channels, channel_multiplier, hidden_dim, description) in enumerate(EXPERIMENTS, 1):
            exp_name = f"exp{i:02d}_{description}"
            
            print(f"\n\n{'#'*80}")
            print(f"EXPERIMENT {i}/{len(EXPERIMENTS)}: {exp_name}")
            print(f"{'#'*80}")
            print(f"Config: blocks={num_blocks}, base_ch={base_channels}, mult={channel_multiplier}, hidden={hidden_dim}")
            
            # Update config
            update_config(num_blocks, base_channels, channel_multiplier, hidden_dim)
            
            # Add descriptive tags
            tags = ["quick_benchmark", description]
            
            # Run training
            exp_start = time.time()
            success = run_training(exp_name, tags=tags)
            exp_duration = time.time() - exp_start
            
            results.append({
                'experiment': exp_name,
                'config': f"{num_blocks},{base_channels},{channel_multiplier},{hidden_dim}",
                'success': success,
                'duration_minutes': exp_duration / 60
            })
            
            print(f"\nExperiment duration: {exp_duration/60:.1f} minutes")
            
            # Brief pause between experiments
            if i < len(EXPERIMENTS):
                print("\nWaiting 10 seconds before next experiment...")
                time.sleep(10)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user!")
    
    finally:
        # Restore original config
        restore_config()
        
        # Print summary
        total_duration = time.time() - start_time
        print("\n\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"Total time: {total_duration/3600:.2f} hours")
        print(f"\nResults:")
        print(f"{'Experiment':<30} {'Config (B,C,M,H)':<20} {'Status':<12} {'Time (min)':<12}")
        print("-"*80)
        
        for r in results:
            status = "✓ Success" if r['success'] else "✗ Failed"
            print(f"{r['experiment']:<30} {r['config']:<20} {status:<12} {r['duration_minutes']:<12.1f}")
        
        successful = sum(1 for r in results if r['success'])
        print(f"\nCompleted: {successful}/{len(results)} experiments")
        print("\n✓ Compare results in MLflow UI:")
        print("  conda activate DL-project")
        print("  python -m mlflow ui --backend-store-uri file:./mlruns --port 5000")
        print("  http://localhost:5000")


if __name__ == "__main__":
    main()
