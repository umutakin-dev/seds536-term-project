# SEDS536 Term Project Presentation

**Duration**: 10 minutes total (5 min slides + 5 min demo)
**Slides**: 10 slides (~30 seconds each)

---

## Slide 1: Title + Problem Statement

### **Skin Tone Detection for Inclusive Skincare Recommendations**

*A Comparison of Classical and Deep Learning Approaches*

---

**Umut Akin**

SEDS536 - Image Understanding | January 2026

---

**Problem**: Skincare/makeup products often lack inclusive shade recommendations

**Challenge**: Detecting skin tone accurately across diverse populations

**Goal**: Build a fair, on-device mobile classifier using image understanding techniques

---

**Speaking notes (~30 sec):**
> "Today I'll present my term project on skin tone detection. The skincare industry often fails to provide inclusive recommendations for all skin tones. My goal was to build a mobile app that can accurately classify skin tones across diverse populations, comparing classical image processing techniques with deep learning. I'll show both approaches, our experiments, and a live demo of the app."

---

## Slide 2: Monk Skin Tone Scale & Fairness Motivation

### **Why Monk Scale? Fairness by Design**

---

**Fitzpatrick Scale (Traditional)**
- 6 categories, developed 1975
- Originally for UV sensitivity, not skin color
- Under-represents darker skin tones

**Monk Skin Tone Scale (2019)**
- 10 categories, developed by Google
- Designed for ML fairness evaluation
- Better representation across full spectrum

---

**Our Groupings:**

| Classes | Monk Scales | Purpose |
|---------|-------------|---------|
| 3-class | Light (1-3), Medium (4-7), Dark (8-10) | Best accuracy |
| 5-class | Very Light, Light, Medium, Dark, Very Dark | Finer granularity |

---

**Fairness Goal**: <10% accuracy gap between best and worst performing classes

---

**Speaking notes (~30 sec):**
> "We chose the Monk Skin Tone Scale over the traditional Fitzpatrick scale because it was specifically designed for machine learning fairness. Fitzpatrick has only 6 categories and under-represents darker skin tones. Monk has 10 categories with better coverage. We grouped these into 3 and 5 classes for our experiments. Our fairness goal was to minimize the accuracy gap across all skin tone groups."

---

## Slide 3: Dataset Overview

### **Casual Conversations v2 (CCv2) Dataset**

---

**Source**: Meta AI Research (2024)

**Purpose**: Fairness evaluation in computer vision

**Content**: Video frames with Monk scale annotations (1-10)

---

**Dataset Statistics:**

| Split | Images | Notes |
|-------|--------|-------|
| Train | 104,510 | Weighted sampling |
| Val | 22,280 | |
| Test | 22,730 | |

---

**Class Imbalance Challenge:**

| Scale | % of Data | Challenge |
|-------|-----------|-----------|
| Scale 5 (Medium) | **45%** | Majority class |
| Scale 1 (Lightest) | 0.5% | Under-represented |
| Scale 10 (Darkest) | **0.1%** | Severe scarcity |

*Scale 5 has 394x more samples than Scale 10*

---

**Speaking notes (~30 sec):**
> "We used Meta's Casual Conversations v2 dataset, which contains video frames annotated with Monk skin tone labels. It has about 150,000 images across train, validation, and test splits. The major challenge is severe class imbalance - Scale 5 dominates with 45% of data, while Scale 10 has only 0.1%. This 394x imbalance directly impacts model fairness."

---

## Slide 4: Classical Approach - ITA + Skin Segmentation

### **Individual Typology Angle (ITA)**

*Dermatology standard for skin color measurement*

---

**ITA Formula:**
```
ITA = arctan((L* - 50) / b*) x (180/pi)
```
- **L***: Lightness (LAB color space)
- **b***: Yellow-blue axis

---

**Preprocessing Pipeline (Course Techniques):**

```
RGB Image -> YCbCr Conversion -> Threshold Skin Pixels
                                  (Cb: 77-127, Cr: 133-173)
          -> Morphological Cleanup -> Skin Mask -> ITA Calculation
```

**Morphological Operations (5x5 ellipse kernel):**
- **Opening** (erosion -> dilation): Remove noise
- **Closing** (dilation -> erosion): Fill holes

---

**Segmentation Methods Tested:**
- YCbCr color thresholding
- Face oval + color intersection
- MediaPipe face landmarks (478 points)

---

**Speaking notes (~30 sec):**
> "Our classical approach uses ITA - a dermatology formula that calculates skin tone from LAB color space. The preprocessing pipeline demonstrates key course concepts: color space conversion from RGB to YCbCr for skin detection, thresholding to isolate skin pixels, and morphological operations - opening to remove noise and closing to fill holes - using a 5x5 ellipse kernel. We tested multiple segmentation methods including MediaPipe."

---

## Slide 5: Deep Learning Approach - EfficientNet-B0

### **CNN with Transfer Learning**

*Privacy-first, on-device inference*

---

**Key Requirement: On-Device Processing**
- **No images sent to servers** - user privacy protected
- All inference runs locally on mobile device
- Requires small, efficient model

---

**Why EfficientNet-B0?**
- Pre-trained on ImageNet (transfer learning)
- **Mobile-optimized**: 4M parameters, ~15MB TFLite
- Fast inference: <2 seconds on mid-range devices
- Converts easily to TFLite for Flutter integration

---

**Architecture:**
```
Input (224x224x3) -> EfficientNet-B0 -> Dropout (0.3) -> FC -> Softmax
```

---

**Handling Class Imbalance:**
- Weighted loss (inverse frequency)
- Oversampling (WeightedRandomSampler)
- Augmentation (rotation, flip, color jitter)

---

**Speaking notes (~30 sec):**
> "A top priority was on-device processing - no user images should ever leave the device. This ruled out cloud APIs and required a small, efficient model. We chose EfficientNet-B0 because it's only 4 million parameters, converts to a 15MB TFLite model, and runs inference in under 2 seconds. To handle class imbalance, we used weighted loss, oversampling, and augmentation."

---

## Slide 6: Experiment Summary

### **Experiments Overview**

*6 experiments comparing classical vs deep learning*

---

| Exp | Method | Preprocessing | Classes | Description |
|-----|--------|---------------|---------|-------------|
| 1 | CNN | None | 10 | Baseline (failed) |
| 2 | CNN | Haar face crop | 3 | **Best model** |
| 3 | ITA | None / YCbCr / Oval | 3 | Classical baseline |
| 4 | CNN | MediaPipe masks | 3 | Skin-only images |
| 5a | ITA | MediaPipe raw | 5 | Empirical thresholds |
| 5 | CNN | MediaPipe raw | 5 | 5-class CNN |

---

**Key Variables Tested:**
- **Classes**: 10 vs 3 vs 5
- **Preprocessing**: None -> Haar -> MediaPipe
- **Method**: Classical ITA vs CNN
- **Segmentation**: Color threshold, oval mask, face landmarks

---

**Speaking notes (~30 sec):**
> "We ran 6 experiments systematically varying the method, preprocessing, and number of classes. Experiment 1 tried 10-class CNN and failed due to class imbalance. Experiment 2 used 3-class with Haar face crops and became our best model. Experiments 3 through 5a tested ITA with various segmentation methods. Experiment 5 trained a 5-class CNN on MediaPipe-preprocessed data."

---

## Slide 7: Key Results & Comparison

### **Results: CNN Dramatically Outperforms ITA**

---

| Experiment | Method | Accuracy | Macro F1 |
|------------|--------|----------|----------|
| Exp 1: 10-class CNN | CNN | 38.7% | 0.230 |
| **Exp 2: 3-class Haar** | **CNN** | **78.6%** | **0.625** |
| Exp 3: ITA raw | Classical | 19.3% | 0.188 |
| Exp 3b: ITA + tuned | Classical | 52.3% | 0.337 |
| Exp 4: MediaPipe CNN | CNN | 77.1% | 0.600 |
| Exp 5a: 5-class ITA | Classical | 17.1% | 0.155 |

---

**Key Findings:**

1. **CNN 2x better than ITA**: 78.6% vs 52.3% (best of each)
2. **3-class > 10-class**: Grouping improved accuracy by 40%
3. **ITA ceiling ~52%**: More preprocessing didn't help
4. **Simpler preprocessing wins**: Haar (78.6%) > MediaPipe (77.1%)

---

**Critical Discovery**: ITA is non-monotonic with Monk scale
- Dark class median ITA > Medium class median
- Threshold-based classification fundamentally broken

---

**Speaking notes (~30 sec):**
> "The results clearly show CNN outperforms ITA by a factor of two - 78.6% versus 52%. Reducing from 10 to 3 classes improved accuracy by 40 percentage points. Interestingly, simpler Haar preprocessing beat MediaPipe. Most importantly, we discovered ITA is fundamentally misaligned with the Monk scale - the Dark class actually has higher ITA values than Medium, making threshold-based classification impossible."

---

## Slide 8: Fairness Analysis

### **Per-Class Performance (Best Model: Exp 2)**

---

| Class | Accuracy | Precision | Recall | Support |
|-------|----------|-----------|--------|---------|
| Light (1-3) | 60.9% | 0.517 | 0.609 | 4,530 |
| Medium (4-7) | **84.4%** | 0.881 | 0.844 | 17,570 |
| Dark (8-10) | 42.4% | 0.486 | 0.424 | 630 |

---

**Fairness Metrics:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Accuracy Gap | 42% | <10% | Not met |
| Worst-Case Ratio | 50.2% | >85% | Not met |
| Best Class | 84.4% (Medium) | - | - |
| Worst Class | 42.4% (Dark) | - | - |

---

**Root Cause**: Data imbalance
- Medium: 77% of training data
- Dark: only 2.5% of training data

**Progress**: Worst class improved from **0%** (Exp 1) to **42.4%** (Exp 2)

---

**Speaking notes (~30 sec):**
> "Fairness analysis shows our model still has gaps. Medium class achieves 84% accuracy, but Dark class only 42% - a 42 percentage point gap, missing our 10% target. The root cause is data imbalance: Medium has 77% of training data while Dark has only 2.5%. However, we made significant progress - in Experiment 1, the darkest classes had 0% accuracy. Now Dark class achieves 42%."

---

## Slide 9: Conclusions & Limitations

### **Conclusions**

---

**What Worked:**
- CNN achieves 78.6% accuracy (3-class)
- On-device inference achieved (~15MB TFLite model)
- All skin tone classes now learnable (vs 0% for darkest in baseline)
- Classical techniques demonstrated for educational comparison

**What We Learned:**
- CNN learns features ITA cannot capture (perceptual vs colorimetric)
- Simpler preprocessing often beats complex pipelines
- Data quantity matters more than preprocessing quality
- ITA fundamentally misaligned with Monk scale

---

**Limitations:**

| Limitation | Impact |
|------------|--------|
| Class imbalance | Dark class underperforms (42% vs 84%) |
| Dataset bias | CCv2 skewed toward medium tones |
| Lighting variation | Uncontrolled smartphone lighting |
| Single dataset | Results may not generalize |

---

**Speaking notes (~30 sec):**
> "In conclusion, we achieved our goal of on-device skin tone detection with 78.6% accuracy. The CNN approach dramatically outperformed classical ITA, learning perceptual features that colorimetric formulas cannot capture. Key limitations remain: class imbalance causes the Dark class to underperform, and our results are based on a single dataset with uncontrolled lighting conditions."

---

## Slide 10: Future Work & Questions

### **Future Work**

---

**Addressing Class Imbalance:**
- **Balanced augmentation**: Augment all classes to equal size
- **Focal loss**: Down-weight easy examples, focus on hard cases

**Model Improvements:**
- **Confidence thresholding**: Reject uncertain predictions, prompt retake
- **Test-time augmentation (TTA)**: Average predictions over multiple crops

**MLOps & Experimentation:**
- **Experiment tracking**: MLflow or Weights & Biases for reproducibility
- **Model versioning**: Track model iterations and performance

---

### **Questions?**

---

**Speaking notes (~30 sec):**
> "For future work, we'd address class imbalance through balanced augmentation and focal loss. We'd add confidence thresholding to handle uncertain predictions. For better experimentation, we'd use MLOps tools like MLflow to track experiments and version models. Any questions?"

---

## Slide Summary

**Presentation Flow: Demo First!**

| # | Content | Time |
|---|---------|------|
| - | **DEMO FIRST** (show the app working) | 5:00 |
| 1 | Title + Problem Statement | 0:30 |
| 2 | Monk Scale & Fairness | 0:30 |
| 3 | Dataset Overview | 0:30 |
| 4 | Classical: ITA + Morphological Ops | 0:30 |
| 5 | Deep Learning: EfficientNet-B0 | 0:30 |
| 6 | Experiment Summary | 0:30 |
| 7 | Key Results & Comparison | 0:30 |
| 8 | Fairness Analysis | 0:30 |
| 9 | Conclusions & Limitations | 0:30 |
| 10 | Future Work & Questions | 0:30 |
| - | **Total** | **10:00** |

**Why demo first?**
- Hook the audience with the end result
- Gives context for the technical details
- "Now let me show you how we built this..."
