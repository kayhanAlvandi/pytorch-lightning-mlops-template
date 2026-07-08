"""Test Lightning checkpoint saving with OmegaConf types."""
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from hydra.utils import instantiate
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset


class TestModel(pl.LightningModule):
    def __init__(self, dropout, num_blocks, base_channels, class_names, learning_rate=0.001, weight_decay=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(10, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
    def forward(self, x):
        return self.linear(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


# Test with OmegaConf types
print("="*60)
print("TEST: Lightning checkpoint with OmegaConf types")
print("="*60)

cfg = OmegaConf.create({
    "_target_": "__main__.TestModel",
    "dropout": 0.5,
    "num_blocks": 4,
    "base_channels": 32,
    "class_names": ["class1", "class2", "class3"]
})

print("\n=== Instantiating with _convert_='all' ===")
model = instantiate(cfg, _convert_="all")

print("\n=== Saved hparams ===")
for k, v in model.hparams.items():
    print(f"{k}: type={type(v)}, value={v}")

# Save Lightning checkpoint using proper Lightning mechanism
ckpt_path = "test_lightning.ckpt"


checkpoint_callback = ModelCheckpoint(dirpath=".", filename="test_lightning", save_top_k=1)
trainer = pl.Trainer(
    max_epochs=1, 
    logger=False, 
    callbacks=[checkpoint_callback],
    enable_progress_bar=False,
    enable_model_summary=False
)
# Create a proper DataLoader
dummy_x = torch.randn(10, 10)
dummy_y = torch.randint(0, 2, (10,))
dummy_dataset = TensorDataset(dummy_x, dummy_y)
dummy_loader = DataLoader(dummy_dataset, batch_size=2)

trainer.fit(model, dummy_loader)
ckpt_path = checkpoint_callback.best_model_path
print(f"\n✓ Lightning checkpoint saved to {ckpt_path}")

# Try to load with weights_only=True
try:
    loaded = TestModel.load_from_checkpoint(ckpt_path, weights_only=True)
    print("✓ Loaded with weights_only=True")
except Exception as e:
    print("✗ Failed with weights_only=True:")
    print(f"   {type(e).__name__}: {str(e)[:2000]}")

# Try to load with weights_only=False
try:
    loaded = torch.load(ckpt_path, weights_only=False)
    print("✓ Loaded with weights_only=False")
except Exception as e:
    print(f"✗ Failed with weights_only=False: {e}")

# Cleanup
if Path(ckpt_path).exists():
    Path(ckpt_path).unlink()
    print(f"\n✓ Cleaned up {ckpt_path}")


# Test 2: Direct Python instantiation without Hydra
print("\n" + "="*60)
print("TEST 2: Direct Python instantiation (no Hydra)")
print("="*60)

print("\n=== Creating model directly ===")
model2 = TestModel(dropout=0.5, num_blocks=4, base_channels=32, class_names=["class1", "class2", "class3"])

print("\n=== Saved hparams ===")
for k, v in model2.hparams.items():
    print(f"{k}: type={type(v)}, value={v}")

# Save Lightning checkpoint using proper Lightning mechanism
ckpt_path2 = "test_lightning2.ckpt"
checkpoint_callback2 = ModelCheckpoint(dirpath=".", filename="test_lightning2", save_top_k=1)
trainer2 = pl.Trainer(
    max_epochs=1, 
    logger=False, 
    callbacks=[checkpoint_callback2],
    enable_progress_bar=False,
    enable_model_summary=False
)
trainer2.fit(model2, dummy_loader)
ckpt_path2 = checkpoint_callback2.best_model_path
print(f"\n✓ Lightning checkpoint saved to {ckpt_path2}")

# Try to load with weights_only=True
try:
    loaded2 = TestModel.load_from_checkpoint(ckpt_path2, weights_only=True)
    print("✓ Loaded with weights_only=True")
except Exception as e:
    print("✗ Failed with weights_only=True:")
    print(f"   {type(e).__name__}: {str(e)[:2000]}")

# Try to load with weights_only=False
try:
    loaded2 = TestModel.load_from_checkpoint(ckpt_path2, weights_only=False)
    print("✓ Loaded with weights_only=False")
except Exception as e:
    print(f"✗ Failed with weights_only=False: {e}")

# Cleanup
if Path(ckpt_path2).exists():
    Path(ckpt_path2).unlink()
    print(f"\n✓ Cleaned up {ckpt_path2}")


# Test 3: Simulate actual train.py pattern with optimizer params from config
print("\n" + "="*60)
print("TEST 3: Actual train.py pattern (optimizer params from config)")
print("="*60)

# Create a config that simulates the actual train.py setup
optimizer_cfg = OmegaConf.create({
    "learning_rate": 0.001,
    "weight_decay": 0.0001
})

model_cfg = OmegaConf.create({
    "_target_": "__main__.TestModel",
    "dropout": 0.5,
    "num_blocks": 4,
    "base_channels": 32,
    "class_names": ["class1", "class2", "class3"]
})

print("\n=== Instantiating with params from separate configs (like train.py) ===")
# This simulates how train.py passes optimizer params from cfg.optimizer
model3 = instantiate(
    model_cfg,
    learning_rate=optimizer_cfg.learning_rate,  # Direct from config (no float())
    weight_decay=optimizer_cfg.weight_decay,    # Direct from config (no float())
    _convert_="all",
)

print("\n=== Saved hparams ===")
for k, v in model3.hparams.items():
    print(f"{k}: type={type(v)}, value={v}")

# Save Lightning checkpoint
ckpt_path3 = "test_lightning3.ckpt"
checkpoint_callback3 = ModelCheckpoint(dirpath=".", filename="test_lightning3", save_top_k=1)
trainer3 = pl.Trainer(
    max_epochs=1, 
    logger=False, 
    callbacks=[checkpoint_callback3],
    enable_progress_bar=False,
    enable_model_summary=False
)
trainer3.fit(model3, dummy_loader)
ckpt_path3 = checkpoint_callback3.best_model_path
print(f"\n✓ Lightning checkpoint saved to {ckpt_path3}")

# Try to load with weights_only=True
try:
    loaded3 = TestModel.load_from_checkpoint(ckpt_path3, weights_only=True)
    print("✓ Loaded with weights_only=True")
except Exception as e:
    print("✗ Failed with weights_only=True:")
    print(f"   {type(e).__name__}: {str(e)[:2000]}")

# Try to load with weights_only=False
try:
    loaded3 = TestModel.load_from_checkpoint(ckpt_path3, weights_only=False)
    print("✓ Loaded with weights_only=False")
except Exception as e:
    print(f"✗ Failed with weights_only=False: {e}")

# Cleanup
if Path(ckpt_path3).exists():
    Path(ckpt_path3).unlink()
    print(f"\n✓ Cleaned up {ckpt_path3}")


# Test 4: With explicit type conversion (recommended pattern)
print("\n" + "="*60)
print("TEST 4: Recommended pattern (explicit type conversion)")
print("="*60)

print("\n=== Instantiating with explicit type conversion ===")
model4 = instantiate(
    model_cfg,
    learning_rate=float(optimizer_cfg.learning_rate),  # Explicit float()
    weight_decay=float(optimizer_cfg.weight_decay),    # Explicit float()
    _convert_="all",
)

print("\n=== Saved hparams ===")
for k, v in model4.hparams.items():
    print(f"{k}: type={type(v)}, value={v}")

# Save Lightning checkpoint
ckpt_path4 = "test_lightning4.ckpt"
checkpoint_callback4 = ModelCheckpoint(dirpath=".", filename="test_lightning4", save_top_k=1)
trainer4 = pl.Trainer(
    max_epochs=1, 
    logger=False, 
    callbacks=[checkpoint_callback4],
    enable_progress_bar=False,
    enable_model_summary=False
)
trainer4.fit(model4, dummy_loader)
ckpt_path4 = checkpoint_callback4.best_model_path
print(f"\n✓ Lightning checkpoint saved to {ckpt_path4}")

# Try to load with weights_only=True
try:
    loaded4 = TestModel.load_from_checkpoint(ckpt_path4, weights_only=True)
    print("✓ Loaded with weights_only=True")
except Exception as e:
    print("✗ Failed with weights_only=True:")
    print(f"   {type(e).__name__}: {str(e)[:2000]}")

# Try to load with weights_only=False
try:
    loaded4 = TestModel.load_from_checkpoint(ckpt_path4, weights_only=False)
    print("✓ Loaded with weights_only=False")
except Exception as e:
    print(f"✗ Failed with weights_only=False: {e}")

# Cleanup
if Path(ckpt_path4).exists():
    Path(ckpt_path4).unlink()
    print(f"\n✓ Cleaned up {ckpt_path4}")
