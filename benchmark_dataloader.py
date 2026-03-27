"""Benchmark script to find optimal DataLoader configuration.

Tests combinations of:
- num_workers: [0, 2, 4, 8]
- batch_size: [16, 32, 64, 128]
- cache_size: [4, 8, 16, 32]
- pin_memory: [True, False]

Measures: samples/second throughput
"""
import sys
import time
import itertools
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.datamodule import MultiChannelDataModule
from src.config import Config


def benchmark_config(
    root_dir: str,
    channels: list,
    crop_size: int,
    num_workers: int,
    batch_size: int,
    cache_size: int,
    pin_memory: bool,
    use_tiling: bool = True,
    num_batches: int = 20,
    warmup_batches: int = 5,
    max_tiles: int = 2000,
    num_runs: int = 2,
) -> dict:
    """Benchmark a specific DataLoader configuration.
    
    Returns dict with throughput metrics.
    """
    try:
        datamodule = MultiChannelDataModule(
            root_dir=root_dir,
            channels=channels,
            crop_size=crop_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_mongodb=True,
            use_tiling=use_tiling,
            cache_size=cache_size,
        )
        datamodule.setup()
        
        # Limit dataset to max_tiles for faster benchmark
        if len(datamodule.train_dataset) > max_tiles:
            datamodule.train_dataset.tiles = datamodule.train_dataset.tiles[:max_tiles]
        
        train_loader = datamodule.train_dataloader()
        
        # Warmup
        loader_iter = iter(train_loader)
        for _ in range(warmup_batches):
            try:
                batch = next(loader_iter)
                # Move to GPU if available to simulate real training
                if torch.cuda.is_available():
                    batch[0].cuda()
            except StopIteration:
                loader_iter = iter(train_loader)
        
        # Benchmark with multiple runs
        throughputs = []
        for run in range(num_runs):
            total_samples = 0
            start_time = time.perf_counter()
            
            loader_iter = iter(train_loader)
            for _ in range(num_batches):
                try:
                    batch = next(loader_iter)
                    if torch.cuda.is_available():
                        batch[0].cuda()
                    total_samples += batch[0].shape[0]
                except StopIteration:
                    loader_iter = iter(train_loader)
            
            elapsed = time.perf_counter() - start_time
            throughputs.append(total_samples / elapsed)
        
        throughput = sum(throughputs) / len(throughputs)
        
        return {
            "num_workers": num_workers,
            "batch_size": batch_size,
            "cache_size": cache_size,
            "pin_memory": pin_memory,
            "throughput": throughput,
            "samples_per_sec": f"{throughput:.1f}",
            "elapsed": f"{elapsed:.2f}s",
            "total_samples": total_samples,
            "status": "success",
        }
        
    except Exception as e:
        return {
            "num_workers": num_workers,
            "batch_size": batch_size,
            "cache_size": cache_size,
            "pin_memory": pin_memory,
            "throughput": 0,
            "status": f"error: {str(e)[:50]}",
        }


def run_benchmark(config_path: str = "config/config.yaml"):
    """Run full benchmark suite."""
    config = Config.from_yaml(config_path)
    
    # Parameter grid to search
    # With 5000 tiles (~50 images), test cache sizes that matter
    param_grid = {
        "num_workers": [0],
        "batch_size": [64],
        "cache_size": [16, 32, 64, 128],  # 50 images needed, so these will show cache effect
        "pin_memory": [True],
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    
    print("=" * 70)
    print("DataLoader Benchmark")
    print(f"Data root: {config.data.root_dir}")
    print(f"Crop size: {config.data.crop_size}")
    print(f"Channels: {config.data.channels}")
    print(f"Testing {len(combinations)} configurations...")
    print("=" * 70)
    
    results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        result = benchmark_config(
            root_dir=config.data.root_dir,
            channels=config.data.channels,
            crop_size=config.data.crop_size,
            **params,
        )
        results.append(result)
        
        if result["status"] == "success":
            print(f"  → {result['samples_per_sec']} samples/sec ({result['elapsed']})")
        else:
            print(f"  → {result['status']}")
    
    # Sort by throughput
    successful = [r for r in results if r["status"] == "success"]
    successful.sort(key=lambda x: x["throughput"], reverse=True)
    
    print("\n" + "=" * 70)
    print("RESULTS (sorted by throughput)")
    print("=" * 70)
    print(f"{'workers':<8} {'batch':<8} {'cache':<8} {'pin_mem':<8} {'samples/sec':<12}")
    print("-" * 70)
    
    for r in successful[:10]:  # Top 10
        print(f"{r['num_workers']:<8} {r['batch_size']:<8} {r['cache_size']:<8} "
              f"{str(r['pin_memory']):<8} {r['samples_per_sec']:<12}")
    
    if successful:
        best = successful[0]
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION:")
        print(f"  num_workers: {best['num_workers']}")
        print(f"  batch_size: {best['batch_size']}")
        print(f"  cache_size: {best['cache_size']}")
        print(f"  pin_memory: {best['pin_memory']}")
        print(f"  throughput: {best['samples_per_sec']} samples/sec")
        print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_benchmark()
