# Training Experiments Log

This document tracks ML training experiments for the Monk Skin Tone classifier.

---

## Experiment 1: Baseline EfficientNet-B0

**Date**: January 3, 2026
**Duration**: ~6.2 hours (30 epochs)
**Status**: Completed

### Objective

Train a baseline 10-class Monk Skin Tone classifier using the preprocessed CCv2 dataset.

### Dataset

| Split | Images | Notes |
|-------|--------|-------|
| Train | 104,510 | Weighted sampling enabled |
| Val | 22,280 | |
| Test | 22,730 | |

**Class Distribution (Training Set)**:
| Scale | Samples | Percentage |
|-------|---------|------------|
| 1 | 570 | 0.5% |
| 2 | 5,520 | 5.3% |
| 3 | 14,410 | 13.8% |
| 4 | 17,870 | 17.1% |
| 5 | 47,260 | **45.2%** |
| 6 | 13,230 | 12.7% |
| 7 | 2,970 | 2.8% |
| 8 | 1,850 | 1.8% |
| 9 | 710 | 0.7% |
| 10 | 120 | 0.1% |

### Configuration

```yaml
Model:
  architecture: efficientnet_b0
  pretrained: true (ImageNet)
  dropout: 0.3
  total_parameters: 4,020,358

Training:
  batch_size: 32
  num_epochs: 30
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler: cosine annealing
  optimizer: AdamW

Class Imbalance Handling:
  class_weights: inverse_frequency
  use_oversampling: true (WeightedRandomSampler)

Augmentation:
  resize: 256
  crop_size: 224
  horizontal_flip: true
  rotation: 15 degrees
  color_jitter: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
  normalize: ImageNet stats
```

### Results

#### Overall Metrics

| Metric | Value |
|--------|-------|
| Train Accuracy | 93.0% |
| Val Accuracy | 40.9% |
| **Test Accuracy** | **38.7%** |
| Macro F1 | 0.230 |
| Weighted F1 | 0.407 |

#### Per-Class Performance

| Scale | Precision | Recall | F1 | Accuracy | Support |
|-------|-----------|--------|-----|----------|---------|
| 1 (Lightest) | 0.111 | 0.007 | 0.013 | **0.7%** | 150 |
| 2 | 0.169 | 0.138 | 0.152 | 13.8% | 1,230 |
| 3 | 0.264 | 0.359 | 0.304 | 35.9% | 3,150 |
| 4 | 0.237 | 0.352 | 0.283 | 35.2% | 3,860 |
| 5 | 0.741 | 0.492 | 0.591 | **49.2%** | 10,180 |
| 6 | 0.228 | 0.338 | 0.272 | 33.8% | 2,850 |
| 7 | 0.102 | 0.050 | 0.067 | **5.0%** | 680 |
| 8 | 0.332 | 0.188 | 0.240 | 18.8% | 420 |
| 9 | 0.433 | 0.339 | 0.380 | 33.9% | 180 |
| 10 (Darkest) | 0.000 | 0.000 | 0.000 | **0.0%** | 30 |

#### Fairness Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Accuracy Range | 49.2% | <5% | FAIL |
| Accuracy Std Dev | 16.4% | - | - |
| Worst-Case Ratio | 0.0% | >85% | FAIL |
| Best Class Accuracy | 49.2% (Scale 5) | - | - |
| Worst Class Accuracy | 0.0% (Scale 10) | - | - |

#### Confusion Matrix

```
Predicted →   1     2     3     4     5     6     7     8     9    10
True ↓
  1           1    49    61    39     0     0     0     0     0     0
  2           2   170   527   373    79    58    14     2     5     0
  3           0   343  1130  1172   227   217    39    16     6     0
  4           0   203  1262  1360   435   481    73    31    15     0
  5           4   171   821  1804  5005  2242   102    27     4     0
  6           2    46   322   690   744   963    54    25     4     0
  7           0     7    98   151   173   167    34    41     7     2
  8           0    17    61   101    45    72     6    79    39     0
  9           0     0     2    41    24    24    11    17    61     0
 10           0     0     0     9    20     1     0     0     0     0
```

### Analysis

#### What Worked
1. Model learned to distinguish middle skin tones (scales 3-6) reasonably well
2. Training pipeline ran successfully end-to-end
3. Weighted sampling and class weights helped somewhat with imbalance
4. Training accuracy reached 93%, showing model has capacity to learn

#### What Didn't Work
1. **Severe overfitting**: 93% train vs 39% test accuracy
2. **Minority class failure**: Scales 1, 7, 10 essentially not learned
3. **Scale 10 complete failure**: 0% accuracy (only 30 test samples, 120 train samples)
4. **Model biased to majority class**: Predictions cluster around scales 3-6
5. **Fairness targets not met**: 49% accuracy gap between best and worst classes

#### Root Causes
1. **Extreme class imbalance**: Scale 5 has 394x more samples than Scale 10
2. **Insufficient minority samples**: Even with oversampling, rare classes underrepresented
3. **Dataset quality**: Mix of face close-ups and full-body shots may confuse model
4. **Task difficulty**: 10-class fine-grained classification is inherently challenging

### Artifacts

| File | Description |
|------|-------------|
| `training/checkpoints/best_model.pth` | Best model checkpoint (epoch 26) |
| `training/logs/training_20260103_020256.log` | Training log |
| `training/results.json` | Detailed evaluation metrics |
| `training/configs/config.yaml` | Training configuration |

### Next Steps (Potential Improvements)

1. **Reduce number of classes**
   - Group into 3 classes: Light (1-3), Medium (4-7), Dark (8-10)
   - Or 5 classes: merge adjacent scales
   - Should significantly improve accuracy and fairness

2. **Face-crop preprocessing**
   - Use face detection to crop images to faces only
   - Remove full-body shots or non-face regions
   - May improve feature learning for skin tone

3. **More aggressive data augmentation**
   - Mixup or CutMix for regularization
   - AutoAugment policies
   - May help with overfitting

4. **Alternative loss functions**
   - Focal loss (down-weight easy examples)
   - Label smoothing
   - May help with class imbalance

5. **Ensemble or larger models**
   - Try EfficientNet-B2 or ResNet50
   - Ensemble multiple models
   - May improve overall accuracy

6. **Collect more minority class data**
   - Not feasible for this project timeline
   - But would be the ideal solution

### Decision

For the project demo, consider:
- **Option A**: Use this model as-is, acknowledge limitations in presentation
- **Option B**: Retrain with 3-class grouping for better demo results
- **Option C**: Add face-crop preprocessing and retrain

---

## Experiment 2: 3-Class with Face-Cropped Images

**Date**: January 3, 2026
**Duration**: ~2.4 hours (30 epochs)
**Status**: Completed

### Objective

Improve on Experiment 1 by:
1. Grouping 10 Monk scales into 3 classes (Light/Medium/Dark)
2. Using face-cropped images (removing full-body shots and backgrounds)

### Changes from Experiment 1

| Aspect | Experiment 1 | Experiment 2 |
|--------|-------------|--------------|
| Classes | 10 (Monk scales 1-10) | 3 (Light/Medium/Dark) |
| Dataset | Original CCv2 frames | Face-cropped CCv2 |
| Num classes | 10 | 3 |

### Preprocessing: Face Extraction

Before training, ran face extraction on all images:
- **Tool**: OpenCV Haar Cascade detector
- **Time**: 48.8 minutes (12 workers, parallel)
- **Results**: 92.9% faces detected, 7.1% kept original
- **Output**: `training/data/ccv2_faces/`

### Dataset (3-Class Grouping)

| Class | Scales | Train | Val | Test |
|-------|--------|-------|-----|------|
| Light | 1-3 | 20,500 | 4,370 | 4,530 |
| Medium | 4-7 | 81,330 | 17,340 | 17,570 |
| Dark | 8-10 | 2,680 | 570 | 630 |

**Imbalance**: Medium class still dominates (~78%), but much better than 10-class.

### Configuration

```yaml
Model: efficientnet_b0 (pretrained)
Classes: 3 (Light, Medium, Dark)
Data: training/data/ccv2_faces/
Epochs: 30
Batch size: 32
Class weights: inverse_frequency
Oversampling: enabled
```

### Results

#### Overall Metrics

| Metric | Experiment 1 | Experiment 2 | Improvement |
|--------|-------------|--------------|-------------|
| Train Accuracy | 93.0% | 98.2% | +5.2% |
| Val Accuracy | 40.9% | **80.3%** | **+39.4%** |
| Test Accuracy | 38.7% | **78.6%** | **+39.9%** |
| Macro F1 | 0.230 | **0.625** | **+0.395** |
| Weighted F1 | 0.407 | **0.790** | **+0.383** |
| Training Time | 6.2 hours | **2.4 hours** | **-61%** |

#### Per-Class Performance

| Class | Precision | Recall | F1 | Accuracy | Support |
|-------|-----------|--------|-----|----------|---------|
| Light (1-3) | 0.517 | 0.609 | 0.560 | **60.9%** | 4,530 |
| Medium (4-7) | 0.881 | 0.844 | 0.862 | **84.4%** | 17,570 |
| Dark (8-10) | 0.486 | 0.424 | 0.453 | **42.4%** | 630 |

#### Fairness Metrics

| Metric | Experiment 1 | Experiment 2 | Target |
|--------|-------------|--------------|--------|
| Accuracy Range | 49.2% | **42.1%** | <5% |
| Worst-Case Ratio | 0.0% | **50.2%** | >85% |
| Best Class Acc | 49.2% | 84.4% | - |
| Worst Class Acc | 0.0% | **42.4%** | - |

**Fairness targets still FAIL**, but significant improvement:
- Worst class now actually learns (42% vs 0%)
- All classes have meaningful predictions

#### Confusion Matrix

```
              Light  Medium  Dark
Light          2760    1721    49
Medium         2503   14834   233
Dark             73     290   267
```

**Key observations**:
- Medium class well-classified (84%)
- Light often confused with Medium
- Dark class struggles but learns (42% vs 0% before)

### Analysis

#### What Worked
1. **3-class grouping**: Massive accuracy improvement (78% vs 39%)
2. **Face cropping**: Cleaner data, faster training
3. **Faster training**: 2.4 hours vs 6.2 hours (smaller images)
4. **All classes learn**: Even Dark class (42%) vs Scale 10 (0%) before

#### What Still Needs Work
1. **Dark class underperforms**: Only 42% (630 test samples)
2. **Fairness gap**: 42% range between best/worst classes
3. **Light/Medium confusion**: Many Light classified as Medium

#### Root Causes
1. **Class imbalance persists**: Medium has 78% of data
2. **Dark samples scarce**: Only 2,680 training samples
3. **Adjacent class confusion**: Light and Medium overlap visually

### Artifacts

| File | Description |
|------|-------------|
| `training/checkpoints_3class/best_model.pth` | Best model (epoch 25) |
| `training/logs/training_20260103_113325.log` | Training log |
| `training/results_3class.json` | Detailed evaluation metrics |
| `training/configs/config_3class_faces.yaml` | Training configuration |
| `training/data/ccv2_faces/` | Face-cropped dataset |

### Conclusion

**Experiment 2 is a success**:
- ~2x accuracy improvement (78% vs 39%)
- ~2.5x faster training
- All classes now learn meaningfully
- Ready for deployment in demo app

**For production use**, consider:
- Collecting more Dark skin tone samples
- Fine-tuning with focal loss for class imbalance
- Confidence thresholding for uncertain predictions

---

## Summary: Experiment Comparison

| Metric | Exp 1 (10-class) | Exp 2 (3-class) |
|--------|-----------------|-----------------|
| Test Accuracy | 38.7% | **78.6%** |
| Macro F1 | 0.230 | **0.625** |
| Training Time | 6.2 hr | **2.4 hr** |
| Worst Class | 0.0% | **42.4%** |
| Fairness | FAIL | FAIL (improved) |

**Recommendation**: Use Experiment 2 model (3-class) for the demo application.

---

## Future Improvements Roadmap

These improvements are documented for future iterations after the initial app deployment.

### Priority 1: Focal Loss + Class Weighting

**Goal**: Improve Dark class accuracy (currently 42.4%)

**Approach**:
- Replace CrossEntropyLoss with Focal Loss (gamma=2.0)
- Combine with inverse frequency class weights
- Expected improvement: 5-15% on underrepresented classes

**Implementation**:
```python
# In training/scripts/train.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

**Effort**: ~2-3 hours (retrain + evaluate)

---

### Priority 2: Confidence Thresholding

**Goal**: Improve user experience by detecting uncertain predictions

**Approach**:
- After softmax, check if max probability > threshold
- If below threshold, prompt user to retake photo
- Better to say "uncertain" than give wrong recommendation

**Implementation**:
```python
def predict_with_confidence(model, image, threshold=0.7):
    probs = F.softmax(model(image), dim=1)
    confidence, predicted = probs.max(dim=1)

    if confidence < threshold:
        return None, confidence  # Uncertain
    return predicted, confidence
```

**Threshold selection**:
- Plot accuracy vs coverage at thresholds 0.5, 0.6, 0.7, 0.8
- Find optimal tradeoff (e.g., 0.7 might give 85%+ accuracy on accepted samples)

**Effort**: ~1-2 hours (analysis + implementation)

---

### Priority 3: Additional Dark Skin Tone Data

**Goal**: Address root cause of class imbalance (only 2,680 Dark training samples)

**Options**:
1. **Merge additional datasets**: UTKFace, FFHQ, Chicago Face Database
2. **Targeted augmentation**: More aggressive transforms for Dark class only
3. **Synthetic data**: Style transfer or diffusion models (complex)

**Practical approach for this project**:
- Weighted sampling already enabled
- Add more aggressive augmentation for Dark class
- Consider test-time augmentation (TTA)

**Effort**: 4-8 hours depending on approach

---

### Implementation Order

1. Deploy current model to app (in progress)
2. Confidence thresholding (quick win, improves UX)
3. Focal loss experiment (medium effort, may improve fairness)
4. Additional data/augmentation (if time permits)

---

## Model Conversion: PyTorch to TFLite

### Platform Requirements

**IMPORTANT**: TFLite conversion must be done on Linux/WSL because:
- `ai-edge-torch` (Google's official converter) only provides Linux wheels
- TensorFlow conversion tools have limited Windows support
- Python 3.11-3.13 required (TensorFlow doesn't support 3.14+)

### Conversion Steps (WSL on Windows)

```bash
# Navigate to project (from WSL)
cd /mnt/c/Users/ydran/workspace/seds/seds536-image-understanding/seds536-term-project

# Install dependencies with Python 3.12
uv sync --extra conversion --python 3.12

# Run conversion
uv run --python 3.12 python training/scripts/convert_to_tflite.py
```

### Output

- **TFLite model**: `training/models/skin_tone_classifier.tflite`
- **Expected size**: ~15-20 MB (EfficientNet-B0)
- **Input**: RGB image tensor `[1, 3, 224, 224]` (NCHW format)
- **Output**: Logits tensor `[1, 3]` (Light, Medium, Dark)

### Conversion Script

Located at: `training/scripts/convert_to_tflite.py`

Uses `ai-edge-torch` for direct PyTorch → TFLite conversion (no ONNX intermediate).

---
