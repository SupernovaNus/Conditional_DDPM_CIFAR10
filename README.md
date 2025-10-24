# Conditional Denoising Diffusion Probabilistic Model (DDPM) for CIFAR-10

This repository implements a conditional denoising diffusion probabilistic model (DDPM) for image denoising on the CIFAR-10 dataset. The model is implemented in PyTorch and supports both single-GPU and distributed multi-GPU training using DistributedDataParallel (DDP).

## Features

- Conditional DDPM implementation for image denoising
- UNet architecture with time embeddings
- Support for both deterministic (DDIM) and stochastic sampling
- Distributed training support using PyTorch DDP
- Gradient checkpointing for memory efficiency
- Mixed precision training with AMP
- Configurable training parameters via YAML config files

## Project Structure

```
.
├── cond_diffusion_denoiser.py   # Core model implementation
├── train_ddp.py                 # Distributed training script
├── train_single_gpu.py          # Single GPU training script
├── configs/                     # Configuration files
├── data/                        # Dataset directory
└── logs/                        # Training logs
```

## Installation

1. Ensure you have PyTorch installed with CUDA support if using GPU.
2. Clone this repository
3. Install the required dependencies:
```bash
pip install torch torchvision tqdm pyyaml
```

## Usage

### Single GPU Training

To train on a single GPU:

```bash
python train_single_gpu.py --yaml_config configs/default.yaml
```

### Distributed Training

To train with multiple GPUs using DDP:

```bash
bash submit_ddp.sh
```

### Configuration

Training parameters can be configured either through YAML config files or command line arguments. Command line arguments will override the YAML config settings. Key parameters include:

- Training parameters:
  - `epochs`: Number of training epochs
  - `batch_size`: Training batch size
  - `lr`: Learning rate
  - `weight_decay`: Weight decay for optimization

- Model parameters:
  - `in_channels`: Number of input image channels
  - `base_channels`: Base number of channels in UNet
  - `time_dim`: Time embedding dimension
  - `timesteps`: Number of diffusion timesteps

- Data parameters:
  - `train_sigma_min`: Minimum noise level for training
  - `train_sigma_max`: Maximum noise level for training
  - `val_sigma`: Fixed noise level for validation

## Model Architecture

The model consists of:

1. Conditional UNet with time embeddings
2. Cosine schedule for diffusion process
3. Support for both DDPM and DDIM sampling

### Key Components

- `UNetCond`: Conditional UNet architecture with skip connections
- `CondDDPM`: Main diffusion model implementation
- `DiffusionSchedule`: Handles the diffusion process timing
- `CIFAR10DenoiseDataset`: Custom dataset for noisy image pairs

## Training Process

The training process involves:

1. Adding noise to clean images according to a schedule
2. Training the model to predict the noise
3. Using DDIM sampling for validation
4. Tracking PSNR (Peak Signal-to-Noise Ratio) as the evaluation metric

## Performance

Model performance is evaluated using PSNR on the validation set. The best model checkpoint is saved based on the highest validation PSNR score.

## License

[MIT License](LICENSE)