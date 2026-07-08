"""Inspect checkpoint contents to find OmegaConf types leaking in."""
import torch
import sys
from pathlib import Path
from omegaconf import ListConfig, DictConfig


def find_omegaconf_types(obj, path="", depth=0, max_depth=10):
    """Recursively search for OmegaConf types in a nested structure."""
    if depth > max_depth:
        return
    
    obj_type = type(obj).__module__ + "." + type(obj).__qualname__
    
    # Check if this object is an OmegaConf type
    if isinstance(obj, (ListConfig, DictConfig)):
        print(f"  ❌ FOUND OmegaConf type at [{path}]: {type(obj).__name__} = {repr(obj)[:200]}")
        return
    
    if "omegaconf" in obj_type.lower():
        print(f"  ❌ FOUND OmegaConf type at [{path}]: {obj_type} = {repr(obj)[:200]}")
        return
    
    # Recurse into dicts
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_omegaconf_types(v, f"{path}.{k}" if path else str(k), depth + 1, max_depth)
    
    # Recurse into lists/tuples
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            find_omegaconf_types(v, f"{path}[{i}]", depth + 1, max_depth)


def inspect_checkpoint(ckpt_path):
    print(f"\n{'='*70}")
    print(f"Inspecting: {ckpt_path}")
    print(f"{'='*70}")
    
    # Must load with weights_only=False to actually see the contents
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    print(f"\nTop-level keys: {list(ckpt.keys())}")
    
    for key in ckpt.keys():
        print(f"\n--- [{key}] type={type(ckpt[key]).__name__} ---")
        
        if key == "state_dict":
            # Just check types, don't print tensor values
            print(f"  {len(ckpt[key])} parameters (skipping tensor inspection)")
            continue
        
        if key == "optimizer_states":
            print(f"  {len(ckpt[key])} optimizer states (skipping)")
            continue
        
        # For everything else, search recursively
        find_omegaconf_types(ckpt[key], key)
        
        # Also print structure for hyper_parameters
        if key == "hyper_parameters":
            print("\n  Hyper parameters detail:")
            for k, v in ckpt[key].items():
                vtype = type(v).__module__ + "." + type(v).__qualname__
                print(f"    {k}: type={vtype}, value={repr(v)[:200]}")
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        v2type = type(v2).__module__ + "." + type(v2).__qualname__
                        print(f"      {k2}: type={v2type}, value={repr(v2)[:200]}")
        
        # Print callbacks if present
        if key == "callbacks":
            for cb_key, cb_val in ckpt[key].items():
                print(f"  callback: {cb_key}")
                find_omegaconf_types(cb_val, f"callbacks.{cb_key}")


if __name__ == "__main__":
    ckpt_dir = Path("checkpoints")
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    
    if not ckpt_files:
        print("No checkpoint files found in checkpoints/")
        sys.exit(1)
    
    for ckpt_path in ckpt_files:
        inspect_checkpoint(str(ckpt_path))
