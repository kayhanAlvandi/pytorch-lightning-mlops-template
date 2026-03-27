# CNN Classifier for Multi-Channel Microscopy Images

A PyTorch Lightning project for classifying multi-channel microscopy images with configurable channel selection and random cropping.

## Project Structure

```
cnn_classifier/
├── config/
│   └── config.yaml        # Configuration file
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration dataclasses
│   ├── dataset.py         # Custom dataset for multi-channel images
│   ├── datamodule.py      # PyTorch Lightning DataModule
│   ├── transforms.py      # Image transformations
│   └── model.py           # CNN model and Lightning module
├── train.py               # Training script
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

### 1. Set up PYTHONPATH for shared tools library

This project uses the shared `tools.loading` module. Add it to your conda environment:

```bash
conda env config vars set PYTHONPATH=L:\GITHUB\LIB_Python
conda activate tools  # reactivate to apply
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` to customize:

- **data.root_dir**: Path to your image directory
- **data.channels**: List of channels to use (1-5)
- **data.crop_size**: Random crop size for training
- **data.num_classes**: Number of classification classes
- **dataloader**: Batch size, workers, etc.
- **training**: Epochs, learning rate, weight decay
- **model**: Dropout rate

## Usage

### Basic Training

```bash
python train.py --config config/config.yaml
```

### Override Configuration via CLI

```bash
# Use specific channels
python train.py --channels 1 2 3

# Custom crop size and batch size
python train.py --crop-size 256 --batch-size 32

# Custom learning rate and epochs
python train.py --lr 0.0005 --epochs 100

# Use MongoDB for labels
python train.py --use-mongodb
```

### Channel Selection Examples

```bash
# Single channel
python train.py --channels 1

# Two channels
python train.py --channels 1 4

# All 5 channels
python train.py --channels 1 2 3 4 5
```

## Data Format

Images should follow this naming pattern:
```
{plate}_{well}_T{time}F{field}L{layer}A{action}Z{z}C{channel}.jxl
```

Example:
```
MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A01Z01C01.jxl
```

- **plate**: Plate identifier
- **well**: Well position (e.g., K07)
- **C**: Channel number (01-05)
- **F**: Field number

## Labels

Labels are retrieved based on plate and well information:

1. **Dummy Labels** (default): Random labels for testing
2. **MongoDB**: Query MongoDB with plate/well to get actual labels

For MongoDB, ensure your database has documents with:
```json
{
    "plate": "MIG-Exp03-CP-40X-bin1X1",
    "well": "K07",
    "label": 0
}
```

## Model Architecture

Simple CNN with:
- 4 convolutional blocks (32 → 64 → 128 → 256 channels)
- BatchNorm + ReLU activation
- MaxPooling after each block
- Global Average Pooling
- Fully connected classifier with dropout

## Outputs

- **checkpoints/**: Model checkpoints
- **logs/**: TensorBoard logs

View training progress:
```bash
tensorboard --logdir logs
```

## Next Steps (Future Complexity)

1. Add more sophisticated architectures (ResNet, EfficientNet)
2. Implement data augmentation strategies
3. Add cross-validation
4. Implement proper MongoDB label loading
5. Add inference script
6. Add attention mechanisms
7. Implement multi-task learning
