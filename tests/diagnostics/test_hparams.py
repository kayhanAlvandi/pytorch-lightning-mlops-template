"""Test what gets saved in save_hyperparameters with OmegaConf inputs."""
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from hydra.utils import instantiate
import os


class TestModel(pl.LightningModule):
    def __init__(self, param1, param2, param3):
        super().__init__()
        self.save_hyperparameters()
        print("\n=== Inside __init__ ===")
        print(f"param1 type: {type(param1)}, value: {param1}")
        print(f"param2 type: {type(param2)}, value: {param2}")
        print(f"param3 type: {type(param3)}, value: {param3}")
        
    def forward(self, x):
        return x


# Test 1: Direct instantiation with OmegaConf types
print("\n" + "="*60)
print("TEST 1: Direct instantiation with OmegaConf types")
print("="*60)

cfg = OmegaConf.create({
    "_target_": "__main__.TestModel",
    "param1": [1, 2, 3],
    "param2": {"a": 1, "b": 2},
    "param3": "hello"
})

model1 = instantiate(cfg)
print("\n=== Saved hparams ===")
for k, v in model1.hparams.items():
    print(f"{k}: type={type(v)}, value={v}")

# Save and reload
torch.save(model1.state_dict(), "test_checkpoint1.ckpt")
print("\n✓ Checkpoint saved")

try:
    loaded = torch.load("test_checkpoint1.ckpt", weights_only=True)
    print("✓ Loaded with weights_only=True")
except Exception as e:
    print(f"✗ Failed with weights_only=True: {e}")


# Test 2: With _convert_="all"
print("\n" + "="*60)
print("TEST 2: With _convert_='all'")
print("="*60)

model2 = instantiate(cfg, _convert_="all")
print("\n=== Saved hparams ===")
for k, v in model2.hparams.items():
    print(f"{k}: type={type(v)}, value={v}")

torch.save(model2.state_dict(), "test_checkpoint2.ckpt")
print("\n✓ Checkpoint saved")

try:
    loaded = torch.load("test_checkpoint2.ckpt", weights_only=True)
    print("✓ Loaded with weights_only=True")
except Exception as e:
    print(f"✗ Failed with weights_only=True: {e}")


# Test 3: Manually convert kwargs before passing
print("\n" + "="*60)
print("TEST 3: Manually convert kwargs")
print("="*60)

model3 = TestModel(
    param1=list(cfg.param1),
    param2=dict(cfg.param2),
    param3=str(cfg.param3)
)
print("\n=== Saved hparams ===")
for k, v in model3.hparams.items():
    print(f"{k}: type={type(v)}, value={v}")

torch.save(model3.state_dict(), "test_checkpoint3.ckpt")
print("\n✓ Checkpoint saved")

try:
    loaded = torch.load("test_checkpoint3.ckpt", weights_only=True)
    print("✓ Loaded with weights_only=True")
except Exception as e:
    print(f"✗ Failed with weights_only=True: {e}")


# Cleanup
for f in ["test_checkpoint1.ckpt", "test_checkpoint2.ckpt", "test_checkpoint3.ckpt"]:
    if os.path.exists(f):
        os.remove(f)
