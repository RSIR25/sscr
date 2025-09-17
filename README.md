# SSCR: Efficient Multimodal Cloud Removal Framework via Exploiting Structural Semantics in SAR

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## Abstract

This repository contains the official implementation of the method described in Cheng et al.'s paper on **SAR-optical multimodal cloud removal**. Our approach combines Synthetic Aperture Radar (SAR) and optical satellite observations to reconstruct cloud-free optical images.

**Key Contributions:**
- **Novel SAR-Optical Fusion Architecture**: SSCR framework that effectively integrates multi-modal satellite data
- **Ground Structure Learning**: Advanced structural semantic learning for better understanding of ground surface features
- **Efficient Framework**: Efficient training and inference pipeline with optimized computational performance




## Key Features

- 🌟 **Cloud-free Reconstruction**: Provides pixel-wise reconstruction for cloud-covered areas
- 🛰️ **SAR-Optical Fusion**: Effectively combines Sentinel-1 SAR and Sentinel-2 optical data
- 🎯 **Flexible Loss Functions**: Supports deterministic (L1, L2, Structure Loss) and probabilistic (GNLL, MGNLL) losses
- 🔧 **Flexible Architecture**: Multiple backbone options (SSCR, U-Net, U-TAE, UnCRtainTS)

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.9
- CUDA compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/RSIR25/sscr.git
cd sscr

# Create conda environment
conda create -n sscr python=3.8
conda activate sscr

# Install dependencies
pip install torch torchvision torchaudio
pip install rasterio numpy scipy matplotlib tqdm natsort
pip install s2cloudless opencv-python tensorboard torchnet
```

Alternatively, you may install all that's needed via

```bash
pip install -r requirements.txt
```


### Data Preparation

You can download the dataset from https://patricktum.github.io/cloud_removal/sen12mscr/

The implementation supports the SEN12MS-CR dataset:

**Data Structure:**
```
dataset/
├── ROIs1158_spring_s1/
├── ROIs1158_spring_s2/
├── ROIs1158_spring_s2_cloudy/
├── ROIs1868_summer_s1/
├── ROIs1868_summer_s2/
├── ROIs1868_summer_s2_cloudy/
├── ROIs1970_fall_s1/
├── ROIs1970_fall_s2/
├── ROIs1970_fall_s2_cloudy/
├── ROIs2017_winter_s1/
├── ROIs2017_winter_s2/
├── ROIs2017_winter_s2_cloudy/

```

## Training and Evaluation

### Key Training Parameters

**Model Configuration:**
- `--model`: Architecture choice (`sscr`,`uncrtaints`, `utae`, `unet`)
- `--encoder_widths`: Encoder channel dimensions (e.g., `[128]`)
- `--decoder_widths`: Decoder channel dimensions (e.g., `[128,128,128,128,128]`)
- `--block_type`: Convolution block type (`mbconv`, `residual`)

**Training Setup:**
- `--loss`: Loss function (`ml`,`l1`, `l2`, `GNLL`, `MGNLL`)
- `--use_sar`: Enable SAR-optical fusion

### Training and Testing Commands
You can train a new model via
```bash
python model/train_reconstruct.py --experiment_name demo --root3 path/to/SEN12MSCR --model multisarv2 --epochs 20 --lr 0.001 --batch_size 8 --gamma 1.0 --scale_by 10.0 --loss ml --use_sar --block_type mbconv --n_head 16 --device cuda --res_dir ./results --rdm_seed 1 --pretrain --num_workers 32
```
and you can test a (pre-)trained model via
```bash
python model/test_reconstruct.py --root3 path/to/SEN12MSCR --export_every -1 --load_config path/to/test_model_config --plot_every 1
```

## Evaluation Metrics and Results

### Standard Image Quality Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **SAM**: Spectral Angle Mapper  
- **MAE**: Mean Absolute Error

### Results Visualization and Logging

## Repository Structure

```
├── data/
│   ├── dataLoader.py              # SEN12MS-CR dataset loaders
│   └── __init__.py
├── model/
│   ├── train_reconstruct.py       # Main training script
│   ├── test_reconstruct.py        # Model evaluation
│   ├── inference.py               # Single image inference
│   ├── ensemble_reconstruct.py    # Multi-model ensemble
│   ├── parse_args.py              # Command line argument parsing
│   └── src/
│       ├── model_utils.py         # Model instantiation and utilities
│       ├── losses.py              # Loss function implementations
│       ├── utils.py               # General utility functions
│       ├── backbones/             # Neural network architectures
│       └── learning/
│           ├── metrics.py         # Evaluation metrics implementation
│           └── weight_init.py     # Weight initialization schemes
├── util/
│   ├── detect_cloudshadow.py      # Cloud and shadow detection algorithms
│   ├── utils.py                   # Data processing utilities
│   ├── pre_compute_data_samples.py # Dataset preprocessing
└── README.md                      # This documentation
```


## Related Work and References

This implementation builds upon several key contributions:

- **UnCRtainTS**: [Ebel et al.](https://github.com/PatrickTUM/UnCRtainTS) - "UnCRtainTS: Uncertainty Quantification for Cloud Removal in Optical Satellite Time Series"
- **SEN12MS-CR Dataset**: [Ebel et al.](https://patricktum.github.io/cloud_removal/sen12mscr/) - Cloud removal benchmark dataset
- **s2cloudless**: [Sentinel Hub](https://github.com/sentinel-hub/sentinel2-cloud-detector) - Cloud detection algorithm
