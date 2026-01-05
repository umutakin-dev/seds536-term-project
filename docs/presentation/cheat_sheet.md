# Presentation Cheat Sheet

Quick reference for terminology and concepts, organized by slide order.

---

## Slide 2: Monk Scale & Fairness

### Fitzpatrick Scale
- 6 categories (I-VI), developed 1975
- Originally for UV sensitivity, not skin color
- Under-represents darker skin tones

### Monk Skin Tone Scale
- 10 categories, developed by Google (2019)
- Designed for ML fairness evaluation
- Better representation across full spectrum

### Our Groupings
| Classes | Monk Scales |
|---------|-------------|
| 3-class | Light (1-3), Medium (4-7), Dark (8-10) |
| 5-class | Very Light (1-2), Light (3-4), Medium (5-6), Dark (7-8), Very Dark (9-10) |

---

## Slide 3: Dataset Overview

### Class Imbalance
Unequal distribution of samples across classes.

Our dataset:
| Scale | % of Data |
|-------|-----------|
| Scale 5 (Medium) | 45% |
| Scale 10 (Darkest) | 0.1% |

**Imbalance ratio:** 394:1 (Scale 5 vs Scale 10)

### Weighted Cross-Entropy
Cross-entropy with class weights. We used **inverse frequency weighting**:
- Class with fewer samples gets higher weight
- Formula: `weight = total_samples / (num_classes × class_count)`

---

## Slide 4: Classical Approach - ITA

### Color Spaces

**RGB (Red, Green, Blue)**
- Standard color space for displays
- Not ideal for skin detection (skin colors spread across all channels)

**YCbCr**
- Y: Luminance (brightness)
- Cb: Blue-difference chroma
- Cr: Red-difference chroma
- Skin colors cluster tightly in Cb-Cr plane
- Our thresholds: Cb=[77,127], Cr=[133,173]

**LAB (CIE LAB)**
- L*: Lightness (0-100)
- a*: Green-Red axis
- b*: Blue-Yellow axis
- Used for ITA calculation
- Perceptually uniform

### ITA (Individual Typology Angle)

**Formula:** `ITA = arctan((L* - 50) / b*) × (180/π)`

**What it measures:** Ratio of lightness to yellow-blue. Higher ITA = lighter skin.

**Standard thresholds (Fitzpatrick):**
| ITA | Classification |
|-----|----------------|
| > 55° | Very Light |
| 41-55° | Light |
| 28-41° | Intermediate |
| 10-28° | Tan |
| < 10° | Dark |

**Why it failed for us:** Non-monotonic with Monk scale. Dark class had HIGHER median ITA than Medium (-6.3° vs -3.4°).

### Morphological Operations

All use a **structuring element** (we used 5x5 ellipse kernel).

**Erosion**
- Shrinks white regions
- Removes small white noise

**Dilation**
- Expands white regions
- Fills small holes

**Opening (Erosion → Dilation)**
- Removes small bright spots (noise)
- Preserves overall shape

**Closing (Dilation → Erosion)**
- Fills small dark holes
- Preserves overall shape

---

## Slide 5: Deep Learning Approach

### Transfer Learning
Use a model pre-trained on large dataset (ImageNet, 1.2M images) and fine-tune for specific task.
- Saves training time
- Works with less data
- Leverages learned features (edges, textures, shapes)

### EfficientNet
Family of CNN architectures that balance depth, width, and resolution through compound scaling.
- B0: Smallest (4M params) - we used this
- B7: Largest (66M params)

### Fine-tuning
Unfreeze pre-trained layers and train with small learning rate. Adapts ImageNet features to skin tone classification.

### Dropout
Randomly zero out neurons during training.
- We used 0.3 = 30% dropout
- Prevents overfitting
- Only active during training, not inference

### Cosine Annealing (Learning Rate Schedule)
Learning rate follows cosine curve:
- Starts at initial LR (0.001)
- Gradually decreases to near-zero
- Smoother than step decay

### Data Augmentation (Training)
Transforms applied during training:
- **Horizontal flip**: Mirror image left-right
- **Rotation**: ±15 degrees
- **Color jitter**: Random brightness, contrast, saturation, hue
- **Random crop**: Crop to 224x224 from 256x256 resize

---

## Slide 7: Key Results

### Quick Stats
| Metric | Value |
|--------|-------|
| Best model accuracy | 78.6% (3-class CNN) |
| Best ITA accuracy | 52.3% (tuned thresholds) |
| CNN improvement over ITA | 2x better |
| Training images | 104,510 |
| Model size | ~15MB TFLite |
| Inference time | <2 seconds |

---

## Slide 8: Fairness Analysis

### Fairness Metrics

**Accuracy Parity**
All groups should have similar accuracy.
- Our gap: 42% (Medium 84% vs Dark 42%)
- Target: <10% gap

**Equalized Odds**
All groups should have similar true positive and false positive rates.

**Worst-Case Accuracy**
Minimum accuracy across all groups.
- Ours: 42.4% (Dark class)

---

## Slide 10: Future Work

### Focal Loss
A modified cross-entropy loss designed for **class imbalance**.

**Problem it solves:** Standard cross-entropy treats all samples equally, so the model focuses on easy (majority) examples and ignores hard (minority) examples.

**How it works:**
- Adds weighting factor `(1-p)^γ` that down-weights easy examples
- Focuses training on hard, misclassified examples
- Formula: `FL = -(1-p)^γ * log(p)`
- γ (gamma) controls focusing strength (typically γ=2)

**Example:**
| Sample Type | Confidence | Cross-Entropy | Focal Loss (γ=2) |
|-------------|------------|---------------|------------------|
| Easy (correct) | 0.95 | 0.05 | 0.0001 (ignored) |
| Hard (wrong) | 0.30 | 1.20 | 0.59 (emphasized) |

### Balanced Augmentation
Augment all classes to have **equal sample counts**.

| Class | Original | After Balancing |
|-------|----------|-----------------|
| Light | 20,500 | 80,000 |
| Medium | 81,330 | 80,000 (subsample) |
| Dark | 2,680 | 80,000 (augment ~30x) |

### Test-Time Augmentation (TTA)
At inference:
1. Create multiple augmented versions of input
2. Run inference on each
3. Average predictions

Improves accuracy at cost of speed.

### MLOps Tools

**MLflow**
- Open-source experiment tracking
- Logs parameters, metrics, models
- Local or remote server

**Weights & Biases (W&B)**
- Cloud-based experiment tracking
- Nice visualizations
- Team collaboration

**What they track:**
- Hyperparameters (learning rate, batch size, epochs)
- Metrics per epoch (loss, accuracy, F1)
- Model checkpoints and versions
- Dataset versions

---

## Bonus: Model Deployment

### TFLite (TensorFlow Lite)
Google's format for on-device ML, optimized for mobile/embedded devices.

### Conversion Pipeline
```
PyTorch Model → ai-edge-torch → TFLite Model
```

### Our Model Stats
- Size: ~15MB
- Inference: <2 seconds on mid-range phone
- Input: 224×224 RGB image
- Output: 3 or 5 class probabilities
