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
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig


def benchmark_config(
    datamodule_cfg: DictConfig,
    num_workers: int,
    batch_size: int,
    pin_memory: bool,
    num_batches: int = 20,
    warmup_batches: int = 5,
    max_tiles: int = 2000,
    num_runs: int = 2,
) -> dict:
    """Benchmark a specific DataLoader configuration.
    
    Returns dict with throughput metrics.
    """
    try:
        datamodule = instantiate(
            datamodule_cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            _recursive_=False,
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
        
        cache_size = OmegaConf.select(datamodule_cfg, "dataset.cache_size")
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
        cache_size = OmegaConf.select(datamodule_cfg, "dataset.cache_size")
        return {
            "num_workers": num_workers,
            "batch_size": batch_size,
            "cache_size": cache_size,
            "pin_memory": pin_memory,
            "throughput": 0,
            "status": f"error: {str(e)[:50]}",
        }


def run_benchmark():
    """Run full benchmark suite."""
    from hydra import compose, initialize_config_dir
    
    config_dir = str(Path(__file__).parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
    
    datamodule_cfg = cfg.datamodule
    
    # Parameter grid to search
    param_grid = {
        "num_workers": [0],
        "batch_size": [64],
        "pin_memory": [True],
        "cache_size": [4, 8, 16, 32],
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    
    print("=" * 70)
    print("DataLoader Benchmark")
    print(f"Data root: {datamodule_cfg.dataset.root_dir}")
    print(f"Crop size: {datamodule_cfg.dataset.crop_size}")
    print(f"Channels: {list(datamodule_cfg.dataset.channels)}")
    print(f"Dataset: {datamodule_cfg.dataset._target_}")
    print(f"Testing {len(combinations)} configurations...")
    print("=" * 70)
    
    results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        # Override nested dataset params (e.g. cache_size) via OmegaConf
        cfg_override = OmegaConf.create(OmegaConf.to_container(datamodule_cfg, resolve=True))
        if "cache_size" in params:
            OmegaConf.update(cfg_override, "dataset.cache_size", params.pop("cache_size"))
        
        result = benchmark_config(
            datamodule_cfg=cfg_override,
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
        print(f"{r['num_workers']:<8} {r['batch_size']:<8} "
              f"{r.get('cache_size', '-'):<8} "
              f"{str(r['pin_memory']):<8} {r['samples_per_sec']:<12}")
    
    if successful:
        best = successful[0]
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION:")
        print(f"  num_workers: {best['num_workers']}")
        print(f"  batch_size: {best['batch_size']}")
        print(f"  cache_size: {best.get('cache_size', '-')}")
        print(f"  pin_memory: {best['pin_memory']}")
        print(f"  throughput: {best['samples_per_sec']} samples/sec")
        print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_benchmark()
