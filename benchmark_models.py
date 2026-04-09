"""Benchmark script to train multiple model configurations."""
import subprocess
import sys
from pathlib import Path
import yaml
import time
from datetime import datetime

# Define experiment configurations
# Format: (num_blocks, base_channels, channel_multiplier, hidden_dim, description)
EXPERIMENTS = [
    # Baseline (original architecture)
    (4, 32, 2, 128, "baseline"),
    
    # Depth experiments (shallow vs deep)
    (2, 32, 2, 128, "shallow"),
    (5, 32, 2, 128, "deep"),
    
    # Width experiments (narrow vs wide)
    (4, 16, 2, 128, "narrow"),
    (4, 64, 2, 128, "wide"),
    
    # Channel multiplier experiments
    (4, 32, 1.5, 128, "slow_growth"),
    (4, 32, 3, 128, "fast_growth"),
    
    # Hidden dimension experiments
    (4, 32, 2, 64, "small_classifier"),
    (4, 32, 2, 256, "large_classifier"),
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
    config['model']['channel_multiplier'] = channel_multiplier
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
    print("MODEL ARCHITECTURE BENCHMARK")
    print("="*80)
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Max epochs per experiment: {MAX_EPOCHS}")
    print(f"Estimated total time: ~{len(EXPERIMENTS) * 2} hours (assuming 2h per experiment)")
    print("="*80)
    
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
            print(f"Architecture:")
            print(f"  - num_blocks: {num_blocks}")
            print(f"  - base_channels: {base_channels}")
            print(f"  - channel_multiplier: {channel_multiplier}")
            print(f"  - hidden_dim: {hidden_dim}")
            
            # Calculate expected channels per block
            channels = [int(base_channels * (channel_multiplier ** j)) for j in range(num_blocks)]
            print(f"  - Channel progression: {' → '.join(map(str, channels))}")
            print(f"  - Total parameters: ~{sum(channels)} feature channels")
            
            # Update config
            update_config(num_blocks, base_channels, channel_multiplier, hidden_dim)
            
            # Determine tags based on experiment type
            tags = ["benchmark", description]
            if "baseline" in description:
                tags.append("baseline")
            if "shallow" in description or "deep" in description:
                tags.append("depth_experiment")
            if "narrow" in description or "wide" in description:
                tags.append("width_experiment")
            if "growth" in description:
                tags.append("multiplier_experiment")
            if "classifier" in description:
                tags.append("hidden_dim_experiment")
            
            # Run training
            exp_start = time.time()
            success = run_training(exp_name, tags=tags)
            exp_duration = time.time() - exp_start
            
            results.append({
                'experiment': exp_name,
                'num_blocks': num_blocks,
                'base_channels': base_channels,
                'channel_multiplier': channel_multiplier,
                'hidden_dim': hidden_dim,
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
        print(f"{'Experiment':<25} {'Blocks':<8} {'Base Ch':<10} {'Mult':<8} {'Hidden':<8} {'Status':<10} {'Time (min)':<12}")
        print("-"*80)
        
        for r in results:
            status = "✓ Success" if r['success'] else "✗ Failed"
            print(f"{r['experiment']:<25} {r['num_blocks']:<8} {r['base_channels']:<10} "
                  f"{r['channel_multiplier']:<8} {r['hidden_dim']:<8} {status:<10} {r['duration_minutes']:<12.1f}")
        
        successful = sum(1 for r in results if r['success'])
        print(f"\nCompleted: {successful}/{len(results)} experiments")
        print("\n✓ Check MLflow UI to compare results:")
        print("  mlflow ui")
        print("  http://localhost:5000")


if __name__ == "__main__":
    main()
