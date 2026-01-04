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

| Metric | Exp 1 (10-class) | Exp 2 (3-class) | Exp 3 (ITA Raw) | Exp 3b (Color) | Exp 3c (Oval) | Exp 3d (Small Oval) |
|--------|------------------|-----------------|-----------------|----------------|---------------|---------------------|
| Test Accuracy | 38.7% | **78.6%** | 19.3% | 52.3% | 52.2% | 52.3% |
| Macro F1 | 0.230 | **0.625** | 0.188 | 0.337 | 0.343 | 0.358 |
| Training Time | 6.2 hr | 2.4 hr | **0 hr** | **0 hr** | **0 hr** | **0 hr** |
| Preprocessing | None | Face crop | None | Skin mask | Oval 0.7×0.85 | Oval 0.5×0.7 |
| Skin Ratio | N/A | N/A | N/A | 55.1% | 34.2% | 22.8% |
| Method | CNN | CNN | Classical | Classical | Classical | Classical |

**Key Findings**:
1. **CNN (3-class) is best**: 78.6% accuracy, learns features automatically
2. **ITA raw fails**: 19.3% - literature thresholds don't match real-world data
3. **ITA + preprocessing + tuning**: ~52% ceiling regardless of segmentation method
4. **Segmentation method doesn't matter**: Color-only ≈ Oval+color ≈ Small oval (all ~52%)
5. **ITA is the bottleneck**: High variance (std 37-42°), overlapping distributions
6. **Accuracy-fairness tradeoff**: Can push to 69% with extreme thresholds but Dark class dies (1% recall)

**Classical Techniques Demonstrated** (for SEDS536 course):
- Color space conversion (RGB → YCbCr, RGB → LAB)
- Thresholding for skin segmentation
- Morphological operations (erosion, dilation)
- Geometric masking (ellipse/oval)
- Grid search optimization

**Recommendation**: Use Experiment 2 model (3-class CNN) for production. ITA serves as educational baseline.

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

## Experiment 3: ITA Baseline (Classical Method)

**Date**: January 4, 2026
**Duration**: ~4.5 minutes (inference only, no training)
**Status**: Completed

### Objective

Implement Individual Typology Angle (ITA) as a classical baseline for skin tone classification, comparing against the CNN approach.

### Background: What is ITA?

ITA is a dermatology standard for measuring skin color objectively using the LAB color space:

```
ITA = arctan((L* - 50) / b*) × (180/π)
```

Where:
- **L*** = Lightness (0-100)
- **b*** = Blue-Yellow axis (skin has yellow/warm undertones)

**Standard ITA Thresholds** (from literature):
| ITA Range | Skin Tone | Our 3-Class |
|-----------|-----------|-------------|
| > 55° | Very Light | Light |
| 41°-55° | Light | Light |
| 28°-41° | Intermediate | Medium |
| 10°-28° | Tan | Medium |
| < 10° | Dark | Dark |

### Configuration

```yaml
Method: ITA (Individual Typology Angle)
Color Space: CIE LAB
Thresholds:
  Light: ITA > 41°
  Medium: 10° ≤ ITA ≤ 41°
  Dark: ITA < 10°
Preprocessing: None (raw face crops)
Skin Segmentation: None
```

### Results

#### Overall Metrics

| Metric | ITA Baseline | CNN (Exp 2) | Difference |
|--------|-------------|-------------|------------|
| Test Accuracy | **19.3%** | 78.6% | -59.3% |
| Macro F1 | 0.188 | 0.625 | -0.437 |
| Weighted F1 | 0.243 | 0.790 | -0.547 |

#### Per-Class Performance

| Class | Precision | Recall | F1 | Accuracy | Support |
|-------|-----------|--------|-----|----------|---------|
| Light | 0.227 | 0.325 | 0.268 | 32.5% | 4,530 |
| Medium | 0.721 | 0.146 | 0.243 | 14.6% | 17,570 |
| Dark | 0.027 | 0.548 | 0.052 | 54.8% | 630 |

#### Confusion Matrix

```
Pred →    Light     Medium    Dark
Light     1473      886       2171
Medium    4833      2574      10163
Dark      173       112       345
```

#### ITA Statistics by Ground Truth Class

| Class | Mean ITA | Std Dev | Min | Max |
|-------|----------|---------|-----|-----|
| Light | 7.3° | 51.6° | -90° | 90° |
| Medium | -5.1° | 56.4° | -90° | 90° |
| Dark | 0.0° | 54.3° | -90° | 90° |

### Analysis

#### Why ITA Failed

1. **No skin segmentation**: ITA computed over entire face crop (hair, eyes, lips, background)
2. **High variance**: All classes have ITA std dev > 50° (essentially random)
3. **Overlapping distributions**: Class means are nearly identical (7.3°, -5.1°, 0.0°)
4. **Lighting variation**: Different white balance shifts LAB values unpredictably

#### Key Insight

The CNN (78.6%) dramatically outperforms ITA (19.3%) on the same data because:
- CNN **learns** to focus on relevant skin regions
- CNN **learns** lighting invariance from augmented training data
- ITA needs **explicit preprocessing** (skin segmentation, lighting normalization)

#### Academic Value

This result demonstrates:
1. Why deep learning has replaced classical methods for many vision tasks
2. The importance of preprocessing for classical approaches
3. A fair baseline comparison for the course presentation

### Artifacts

| File | Description |
|------|-------------|
| `training/scripts/ita_baseline.py` | ITA implementation script |
| `training/results_ita_baseline.json` | Detailed results |

### Next Steps: Improving ITA

To make ITA competitive, consider:

1. **Skin segmentation** (Priority 1) ✅ DONE
   - Use color thresholding in YCbCr space
   - Apply morphological cleanup
   - Calculate ITA only on skin pixels

2. **Threshold tuning** ✅ DONE
   - Grid search on validation set
   - Dataset-specific thresholds

3. **ROI selection** (Optional)
   - Extract forehead/cheek regions only
   - Avoid eyes, lips, hair

4. **Lighting normalization** (Optional)
   - CLAHE on L channel before ITA
   - White balance correction

---

## Experiment 3b: ITA with Skin Segmentation + Tuned Thresholds

**Date**: January 4, 2026
**Duration**: ~15 minutes (preprocessing) + ~2 minutes (evaluation)
**Status**: Completed

### Objective

Improve ITA baseline by:
1. Adding skin segmentation (YCbCr color thresholding + morphological cleanup)
2. Tuning ITA thresholds on validation set

### Preprocessing Pipeline

Classical image processing techniques applied:

```
Face Image → RGB to YCbCr → Threshold Skin → Morphological Cleanup → Skin Mask
                              (Cb: 77-127)      (Open + Close)
                              (Cr: 133-173)
```

**Preprocessing Statistics** (149,520 images):
- Mean skin ratio: 55.1%
- Processing time: ~13 minutes (12 workers)
- Output: `ccv2_faces_preprocessed/` with `_masked.jpg` and `_mask.png` per image

### Configuration

```yaml
Method: ITA with Skin Segmentation
Skin Detection: YCbCr color thresholding
  Cb_range: [77, 127]
  Cr_range: [133, 173]
Morphological Cleanup: Opening + Closing (5x5 ellipse kernel)
Threshold Tuning: Grid search on validation set
  Light range: [-30°, 30°] step 5°
  Dark range: [-50°, 0°] step 5°
```

### Results

#### Threshold Tuning (Validation Set)

| Thresholds | Source | Val Accuracy |
|------------|--------|--------------|
| Light > 41°, Dark < 10° | Literature | ~19% |
| Light > 25°, Dark < -50° | **Tuned** | **51.5%** |

**Key insight**: Literature thresholds assume controlled lighting. Real-world data has much lower ITA values.

#### Test Set Performance

| Metric | Exp 3 (Raw ITA) | Exp 3b (Tuned) | Improvement |
|--------|-----------------|----------------|-------------|
| Test Accuracy | 19.3% | **52.3%** | **+33.0%** |
| Macro F1 | 0.188 | **0.337** | **+0.149** |
| Weighted F1 | 0.243 | **0.578** | **+0.335** |

#### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Light | 0.300 | 0.284 | 0.292 | 4,530 |
| Medium | 0.768 | 0.596 | 0.671 | 17,570 |
| Dark | 0.028 | 0.213 | 0.049 | 630 |

#### Confusion Matrix

```
Pred →    Light     Medium    Dark
Light     1288      2822      420
Medium    2852      10470     4248
Dark      157       339       134
```

### Analysis

#### What Worked
1. **Skin segmentation reduced variance**: ITA std dev dropped from 51-56° to 36-41°
2. **Threshold tuning was critical**: 19% → 52% accuracy
3. **Medium class performs well**: 59.6% recall (majority class)

#### What Didn't Work
1. **Dark class still struggles**: Only 21.3% recall (limited samples, high variance)
2. **Light/Medium confusion**: Many Light samples classified as Medium
3. **ITA distributions still overlap**: Even with skin-only pixels

#### Classical Techniques Demonstrated

| Technique | Course Topic | Implementation |
|-----------|--------------|----------------|
| RGB → YCbCr | Color spaces | `skin_segmentation.py` |
| Thresholding | Segmentation | Skin pixel detection |
| Erosion/Dilation | Morphology | Mask cleanup |
| RGB → LAB | Color spaces | ITA calculation |
| Grid search | Optimization | Threshold tuning |

### Artifacts

| File | Description |
|------|-------------|
| `training/scripts/skin_segmentation.py` | Skin segmentation module |
| `training/scripts/preprocess_skin.py` | Batch preprocessing script |
| `training/scripts/ita_baseline.py` | Updated with `--use-preprocessed`, `--tune`, `--workers` |
| `training/data/ccv2_faces_preprocessed/` | Preprocessed dataset |
| `training/results_ita_preprocessed.json` | Detailed results |

---

## Experiment 3c: ITA with Face Oval + Color Segmentation

**Date**: January 4, 2026
**Duration**: ~12 minutes (preprocessing) + ~2 minutes (evaluation)
**Status**: Completed

### Objective

Test whether combining a face oval mask with color-based skin segmentation improves ITA accuracy. The hypothesis was that:
1. Oval restricts detection to face region (excludes hair, background)
2. Color filtering within oval excludes non-skin areas (eyes, lips)

### Preprocessing Pipeline

```
Face Image → Create Oval Mask → YCbCr Skin Detection → Intersection → Final Mask
               (70% x 85%)        (same as 3b)          (both must match)
```

**Oval Parameters** (defaults from `segment_skin_with_oval`):
- width_ratio: 0.7 (70% of image width)
- height_ratio: 0.85 (85% of image height)
- center_y_offset: -0.05 (5% above center)

**Preprocessing Statistics** (149,520 images):
- Mean skin ratio: 34.2% (vs 55.1% in Exp 3b)
- Processing time: ~12 minutes (12 workers)
- Output: `ccv2_faces_preprocessed_v3/`

### Results

#### Threshold Tuning (Validation Set)

| Thresholds | Val Accuracy |
|------------|--------------|
| Light > 25°, Dark < -50° | **52.2%** |

Same optimal thresholds as Exp 3b.

#### Test Set Performance

| Metric | Exp 3b (Color Only) | Exp 3c (Oval+Color) | Change |
|--------|---------------------|---------------------|--------|
| Test Accuracy | 52.3% | **52.2%** | -0.1% |
| Macro F1 | 0.337 | **0.343** | +0.006 |
| Weighted F1 | 0.578 | **0.580** | +0.002 |
| Skin Ratio | 55.1% | **34.2%** | -20.9% |

#### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Light | 0.320 | 0.298 | 0.308 | 4,530 |
| Medium | 0.771 | 0.590 | 0.668 | 17,570 |
| Dark | 0.029 | 0.230 | 0.051 | 630 |

#### Confusion Matrix

```
Pred →    Light     Medium    Dark
Light     1348      2728      454
Medium    2741      10370     4459
Dark      128       357       145
```

#### ITA Statistics by Class

| Class | Mean ITA | Std Dev | Range |
|-------|----------|---------|-------|
| Light | 3.8° | 37.6° | [-90°, 90°] |
| Medium | -18.3° | 40.0° | [-90°, 90°] |
| Dark | -13.8° | 42.7° | [-90°, 90°] |

### Analysis

#### Key Finding: Oval Didn't Help Accuracy

Despite reducing skin ratio from 55% to 34% (more focused mask), accuracy remained essentially identical (52.2% vs 52.3%).

**Why?**
1. **ITA is the bottleneck**: Even with perfect segmentation, ITA distributions overlap significantly
2. **Standard deviations still high**: 37-42° across all classes (vs 36-41° in Exp 3b)
3. **Class means poorly separated**: Light=3.8°, Medium=-18.3°, Dark=-13.8°

#### Conclusion

The segmentation method matters less than the fundamental ITA formula. The CNN's 78.6% accuracy represents what's achievable with learned features versus the ~52% ceiling of classical ITA-based classification.

### Artifacts

| File | Description |
|------|-------------|
| `training/data/ccv2_faces_preprocessed_v3/` | Oval+color preprocessed dataset |
| `training/scripts/skin_segmentation.py` | `segment_skin_with_oval()` function |
| `training/results_ita_preprocessed.json` | Results (overwritten) |

---

## Experiment 3d: Smaller Oval + Threshold Tuning

**Date**: January 5, 2026
**Duration**: ~12 minutes (preprocessing) + ~5 minutes (threshold experiments)
**Status**: Completed

### Objective

1. Test smaller oval mask (0.5 × 0.7) to focus on cheeks/forehead
2. Explore threshold tuning to find optimal accuracy-fairness balance

### Preprocessing: Smaller Oval

**Oval Parameters**:
- width_ratio: 0.5 (vs 0.7 in v3)
- height_ratio: 0.7 (vs 0.85 in v3)

**Results**:
- Mean skin ratio: 22.8% (vs 34.2% in v3)
- Tighter mask focusing on central face region

### Threshold Experiments

Tested multiple threshold combinations on the smaller oval (v4) data:

| Thresholds (Light, Dark) | Val Acc | Test Acc | Light Recall | Medium Recall | Dark Recall | Macro F1 |
|--------------------------|---------|----------|--------------|---------------|-------------|----------|
| > 25°, < -50° (baseline) | 52.9% | 52.3% | 38.9% | 56.8% | 23.3% | 0.358 |
| > 33°, < -60° | 59.2% | 58.8% | 30.4% | 67.6% | 16.8% | 0.367 |
| > 35°, < -80° (extreme) | 70.0% | 69.2% | 28.4% | 82.1% | 1.3% | 0.378 |

### Analysis

#### Key Finding: Accuracy-Fairness Tradeoff

As thresholds become more extreme:
- **Accuracy increases** (52% → 69%) by classifying more samples as Medium
- **Dark class collapses** (23% → 1% recall)
- **Light class suffers** (39% → 28% recall)

This is a classic "majority class takeover" - the model games accuracy by predicting Medium (77% of data) for almost everything.

#### Recommendation for Production

**Use balanced thresholds (25°, -50°)** for a real skincare app:
- More equitable across skin tones
- Avoids telling dark-skinned users they're "medium"
- Fairness > raw accuracy for this use case

### ITA Distribution Analysis

Even with optimized segmentation, ITA distributions overlap significantly:

| Class | Mean ITA | Std Dev | Range |
|-------|----------|---------|-------|
| Light | 9.7° | 39.1° | [-90°, 90°] |
| Medium | -13.9° | 41.7° | [-90°, 90°] |
| Dark | -11.9° | 42.5° | [-90°, 90°] |

**Problem**: Dark mean (-11.9°) is actually higher than Medium mean (-13.9°)! The distributions are essentially indistinguishable, making reliable classification impossible with ITA alone.

### Artifacts

| File | Description |
|------|-------------|
| `training/data/ccv2_faces_preprocessed_v4/` | Small oval preprocessed dataset |
| `training/scripts/ita_baseline.py` | Updated with `--light-range`, `--dark-range` |

---

## Classical Methods: Final Conclusions

### Summary Table

| Experiment | Method | Accuracy | Macro F1 | Notes |
|------------|--------|----------|----------|-------|
| Exp 3 | ITA raw | 19.3% | 0.188 | Literature thresholds fail |
| Exp 3b | ITA + color mask | 52.3% | 0.337 | YCbCr segmentation |
| Exp 3c | ITA + oval+color (0.7×0.85) | 52.2% | 0.343 | Larger oval |
| Exp 3d | ITA + oval+color (0.5×0.7) | 52.3% | 0.358 | Smaller oval |
| Exp 3d | ITA + extreme thresholds | 69.2% | 0.378 | Unfair to minorities |
| **Exp 2** | **CNN (3-class)** | **78.6%** | **0.625** | **Best overall** |

### ITA in Literature vs Our Results

Recent research shows ITA **can** achieve high accuracy under the right conditions:

| Study | Method | Accuracy | Balanced Acc |
|-------|--------|----------|--------------|
| [Nature 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12179258/) | ITA + DensePose + OpenFace | 89-92% | 58-75% |
| [Nature 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12179258/) | ITA → Fitzpatrick | 0.5-20% | 17-65% |
| **Our study** | ITA + Haar Cascade + YCbCr | **52%** | ~35% |

**Why the gap?**

The literature's 89-92% used:
- **DensePose/OpenFace** for face detection (vs our Haar Cascade)
- **Clinical-quality images** with controlled lighting
- **Direct ITA → Monk mapping** (not threshold-based classification)

ITA was originally designed for **colorimeter measurements** in dermatology labs, not smartphone photos. Key limitations from literature:
- "Sensitive to illuminants" - lighting variation degrades accuracy
- "Less stable at higher melanin levels"
- Per-class accuracy varies **0% to 99.66%** even in best conditions

**Potential improvement**: Better face detection (MediaPipe, MTCNN, or RetinaFace) could significantly improve our results. See [Issue #33](https://github.com/umutakin-dev/seds536-term-project/issues/33).

### Why Our ITA Implementation Underperforms

1. **Poor face detection**: Haar Cascade produces noisy crops with hair/background
2. **Uncontrolled lighting**: CCv2 dataset has variable smartphone lighting
3. **No calibration**: Cannot calibrate ITA without reference color patches
4. **Threshold-based**: Binary thresholds lose information vs learned mappings

### Classical Techniques Demonstrated (SEDS536)

| Technique | Implementation |
|-----------|---------------|
| Color space conversion | RGB → YCbCr, RGB → LAB |
| Thresholding | Skin pixel detection |
| Morphological operations | Opening, closing (5×5 ellipse) |
| Geometric masking | Face oval creation |
| Grid search optimization | Threshold tuning |

### Final Recommendation

**Use CNN (Experiment 2)** for production:
- 78.6% accuracy vs ~52% for ITA
- Better fairness across skin tones
- Learned features handle lighting variation
- Already converted to TFLite for mobile deployment

ITA serves as an educational baseline demonstrating why deep learning has replaced classical methods for many vision tasks.

### Known Data Quality Issues

**Face Detection Quality** (tracked in [Issue #33](https://github.com/umutakin-dev/seds536-term-project/issues/33)):
- Some face crops from Haar Cascade detector are poor quality
- Issues include: partial faces, off-center crops, excessive background/hair
- This contributes to high ITA variance even with good segmentation
- Future work: implement quality filtering or use better detector (MTCNN, MediaPipe)

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
