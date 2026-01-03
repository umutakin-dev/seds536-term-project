# SEDS536 Term Project: Skin Tone Detection for Skincare Recommendations

> **Course**: SEDS536 - Image Understanding (Master's Level)
> **Author**: Umut Akin
> **Academic Year**: 2024-2025

## Project Overview

This project develops a mobile application that analyzes facial images to detect skin tones and provide personalized skincare or makeup recommendations based on individual skin types.

### Objectives

- Identify and classify skin tones from facial images
- Provide personalized skincare/makeup recommendations
- Implement on-device image processing (preferred) or backend solution
- Build a Flutter-based mobile application

### Key Features

- Facial image analysis
- Skin tone classification
- Personalized product recommendations
- Privacy-focused design (on-device processing preferred)

## Project Status

**Current Phase**: ML Training

See [docs/project-management/timeline.md](docs/project-management/timeline.md) for detailed milestones.

## Repository Structure

```
seds536-term-project/
├── README.md                          # This file
├── pyproject.toml                     # Python dependencies (uv)
├── docs/                              # Documentation
│   ├── research/                      # Research findings
│   │   ├── literature-review.md       # Papers and articles reviewed
│   │   ├── datasets.md                # Available datasets
│   │   ├── models.md                  # Pre-trained models and approaches
│   │   └── skin-tone-scales.md        # Comparison of classification schemes
│   ├── architecture/                  # Technical design documents
│   │   ├── system-design.md           # Overall system architecture
│   │   ├── ml-approach.md             # ML model selection and design
│   │   └── mobile-integration.md      # Flutter integration approach
│   └── project-management/            # Project planning
│       ├── timeline.md                # Project timeline and milestones
│       └── requirements.md            # Functional requirements
├── training/                          # ML training pipeline
│   ├── configs/
│   │   └── config.yaml                # Training configuration
│   ├── scripts/
│   │   ├── check_env.py               # Environment & GPU check
│   │   ├── preprocess_ccv2.py         # Dataset preprocessing
│   │   ├── dataset.py                 # Dataset & dataloader utilities
│   │   ├── model.py                   # Model architectures
│   │   ├── train.py                   # Training script
│   │   └── evaluate.py                # Evaluation & metrics
│   ├── data/                          # Preprocessed datasets (gitignored)
│   ├── checkpoints/                   # Model checkpoints (gitignored)
│   └── logs/                          # Training logs
├── research-papers/                   # Academic papers (gitignored)
└── app/                               # Flutter application
```

## Technical Approach

### Considerations

**Skin Tone Classification:**
- Fitzpatrick Scale (6 types, medical standard)
- Monk Skin Tone Scale (10 types, more inclusive)
- Custom classification system

**Detection Methods:**
- Color space analysis (LAB color space)
- Traditional computer vision (face detection + color extraction)
- Machine learning (CNN-based classification)
- Hybrid approaches

**Implementation Options:**
- On-device ML: TensorFlow Lite, Google ML Kit, PyTorch Mobile
- Backend processing: Cloud-based ML inference
- Hybrid: Preprocessing on-device, inference on backend

### Ethical Considerations

- **Bias and Fairness**: Ensuring model performs equally well across all skin tones
- **Privacy**: Preference for on-device processing to protect user data
- **Lighting Normalization**: Handling diverse lighting conditions
- **Inclusivity**: Using diverse datasets and evaluation metrics

## Research Areas

1. **Skin Tone Detection Algorithms**
   - State-of-the-art methods
   - Fairness-aware approaches
   - Performance across diverse skin tones

2. **Mobile ML Integration**
   - On-device inference capabilities
   - Model optimization for mobile
   - Flutter ML integration patterns

3. **Datasets & Pre-trained Models**
   - Available diverse facial datasets
   - Pre-trained models for skin analysis
   - Transfer learning opportunities

4. **Skincare Recommendation Systems**
   - Product databases
   - Recommendation algorithms
   - User experience design

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- CUDA-compatible GPU (recommended for training)
- Flutter SDK (for mobile app)

### Python Environment Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/umutakin-dev/seds536-term-project.git
cd seds536-term-project

# Create virtual environment and install dependencies
uv sync

# Verify environment and GPU availability
uv run check-env
```

### Dataset Preparation

The project uses the [Meta Casual Conversations v2](https://ai.meta.com/datasets/casual-conversations-v2-dataset/) dataset with Monk Skin Tone annotations.

1. Request access to the dataset from Meta
2. Download the frame archives (`CCv2_frames_part_1.zip` through `CCv2_frames_part_5.zip`)
3. Download annotations (`CCv2_annotations.zip`)
4. Extract to your downloads folder
5. Run preprocessing:

```bash
uv run python -m training.scripts.preprocess_ccv2
```

This creates a balanced dataset in `training/data/ccv2_balanced/` with train/val/test splits.

## Training

### Quick Start

```bash
# Check environment and GPU
uv run check-env

# Start training with default config
uv run train

# Or with custom config
uv run train --config training/configs/config.yaml
```

### Training Configuration

Edit `training/configs/config.yaml` to customize:

- **Model**: `efficientnet_b0`, `resnet18`, `resnet50`, `mobilenet_v3_small`
- **Batch size**: Adjust based on GPU memory
- **Learning rate**: Default 0.001 with cosine annealing
- **Class weights**: Handles imbalanced Monk scale distribution
- **Augmentation**: Color jitter, rotation, flipping

### Resume Training

```bash
uv run train --resume training/checkpoints/checkpoint_epoch_10.pth
```

### Monitor Progress

Training logs are saved to `training/logs/`. Checkpoints saved to `training/checkpoints/`.

## Evaluation

```bash
# Evaluate best model on test set
uv run evaluate --checkpoint training/checkpoints/best_model.pth --split test

# Save detailed results to JSON
uv run evaluate --checkpoint training/checkpoints/best_model.pth --output results.json
```

Evaluation provides:
- Per-class precision, recall, F1, accuracy
- Confusion matrix
- Fairness metrics (accuracy parity across skin tones)

## Flutter App

See [app/README.md](app/README.md) for mobile application setup.

## Testing

(To be populated once testing framework is established)

## References

See [docs/research/literature-review.md](docs/research/literature-review.md) for academic references and related work.

## License

This is an academic project for SEDS536 course.

## Acknowledgments

Course: SEDS536 - Image Understanding
