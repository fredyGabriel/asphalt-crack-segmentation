# Asphalt Crack Segmentation Configuration

This document provides a comprehensive overview of the configuration settings
used in the asphalt crack segmentation project, ensuring reproducibility of
results.

## Model Architectures

The system supports multiple model architectures:

### UNet Variants
- **Standard UNet**: Basic encoder-decoder architecture with skip connections
- **UNet with Dropout**: Enhanced version with configurable dropout rates at
different network depths
  - Encoder dropout increases with depth (0.1 to 0.4)
  - Bottleneck dropout: 0.5
  - Decoder dropout decreases with upsampling (0.4 to 0.1)

### UNetResNet
- **Architecture**: UNet-like structure with a pre-trained ResNet backbone
- **Available backbones**: resnet18, resnet34, resnet50, resnet101, resnet152
- **Feature channels**: [256, 128, 64, 32]
- **Transfer learning**: Pre-trained weights from ImageNet
- **Gradual unfreezing**: 
  - Starts with frozen encoder
  - Progressively unfreezes layers based on validation performance
  - 5 unfreezing stages (layer4 → layer3+4 → layer2+3+4 → etc.)
  - Controlled by patience parameter (default: 5 epochs)

## Training Configuration

### Basic Parameters
- **Batch size**: 32
- **Learning rate**: 0.001
- **Weight decay**: 0.0001
- **Image size**: 256×256
- **Number of epochs**: 100
- **Early stopping patience**: 10 epochs

### Optimization
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau
  - Mode: min (monitors validation loss)
  - Factor: 0.5 (halves learning rate)
  - Patience: 5 epochs
  - Minimum learning rate: 0.00001
- **Gradient accumulation steps**: 4
- **Mixed precision training**: Enabled (for compatible GPUs)

### Loss Functions
- **Binary Cross Entropy**: Standard for binary segmentation
- **Dice Loss**: Optimizes for intersection over union
- **Combined Loss**: Weighted combination of BCE and Dice

## Data Processing

### Dataset Structure
- **Image directory**: Contains input images
- **Mask directory**: Contains binary segmentation masks
- **Train/Val/Test split**: 70%/15%/15% split with fixed random seed (42)

### Transformations
- **Resize**: Fixed size of 256×256 pixels
- **Random flips**: Horizontal and vertical with 50% probability each
- **Normalization**: ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224,
0.225)

### Augmentations
- **RandomNoise**: Simulates imaging conditions with multiple noise types:
  - Gaussian noise
  - Salt and pepper noise
  - Speckle noise
- **RandomShadow**: Adds realistic shadow patterns to simulate lighting
variations

## Evaluation and Checkpoints

### Metrics
- **IoU**: Intersection over Union (primary metric)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Model Saving Strategy
- **Best IoU model**: Saved when validation IoU improves
- **Best loss model**: Saved when validation loss improves
- **Regular checkpoints**: Every N epochs (default: 5)
- **Final model**: At the end of training

## Reproducibility Features

- **Fixed random seeds**: Set to 42 for Python random, NumPy, and PyTorch
- **Deterministic data splits**: Consistent train/val/test splits
- **Hardware adaptation**: 
  - CUDA acceleration when available
  - CPU fallback option
  - Number of workers: 16 (configurable)
- **TensorBoard logging**: Tracks metrics, losses, and learning rates

## Configuration Management

Configuration is managed through YAML files (`configs/default.yaml`), enabling:
- Easy modification of hyperparameters
- Experiment tracking
- Configuration versioning

This standardized approach ensures experimental results can be faithfully
reproduced.
