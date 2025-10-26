# Literature Review: Skin Tone Detection

This document tracks academic papers, articles, and resources reviewed for the skin tone detection project.

## Key Research Areas

1. Skin tone detection algorithms
2. Fairness and bias in skin classification
3. Mobile ML deployment
4. Color space analysis for skin detection

## Papers Reviewed

### Understanding Skin Color Bias in Deep Learning-Based Skin Lesion Segmentation

**Title**: Understanding skin color bias in deep learning-based skin lesion segmentation
**Authors**: Various (PubMed publication)
**Year**: 2024
**Link**: https://pubmed.ncbi.nlm.nih.gov/38290289/

**Key Findings**:
- Pervasive bias in most published lesion segmentation methods
- Due to use of commonly employed neural network architectures and publicly available datasets
- Dataset imbalances: less than 2.1% provide skin tone annotations
- Derived datasets have 3x more light tone images than dark
- Under-representation results in lower performance for darker skin types

**Relevance to Project**:
- Highlights critical importance of balanced datasets
- Shows need for fairness metrics across all skin tones
- Indicates we must ensure diverse representation

**Implementation Notes**:
- Must measure performance separately for each skin tone category
- Need balanced training data or augmentation strategies

### Skin Cancer Machine Learning Model Tone Bias

**Title**: Skin Cancer Machine Learning Model Tone Bias
**Year**: 2024
**Link**: https://arxiv.org/abs/2410.06385

**Key Findings**:
- CNNs perform well for skin cancer but show tone bias
- Open-source datasets from clinical trials in lighter-skin countries
- Models perform well for lighter tones but worse for darker
- Fairness concerns can reduce public trust in AI health field

**Relevance to Project**:
- Even though our app is for skincare (not medical), fairness is critical
- Need to explicitly test and report performance across tones
- Trust is essential for user adoption

### Enhancing Fairness in Machine Learning: Skin Tone Classification Using the Monk Skin Tone Scale

**Title**: Enhancing Fairness in Machine Learning: Skin Tone Classification Using the Monk Skin Tone Scale
**Year**: 2024
**Link**: https://www.researchgate.net/publication/386403126

**Key Findings**:
- SkinTone in The Wild (STW) dataset created
- Merges well-known face recognition datasets
- Labeled according to 10-class Monk Skin Tone scale
- Maintains equal number of images across all color classes to avoid bias

**Relevance to Project**:
- Strong validation for using Monk scale (10 classes)
- STW dataset may be useful for our project
- Balancing strategy: equal samples per class

**Implementation Notes**:
- If using STW dataset, already annotated with Monk scale
- Consider data augmentation to maintain balance

### Skin Tone Estimation under Diverse Lighting Conditions

**Title**: Skin Tone Estimation under Diverse Lighting Conditions
**Year**: 2024
**Link**: https://pubmed.ncbi.nlm.nih.gov/38786563/

**Key Findings**:
- CNN-based skin tone estimation model refined
- Provides consistent accuracy across different skin tones
- Uses 10-point Monk Skin Tone Scale
- Handles various lighting conditions
- Regression model outperforms with estimated-to-target distance of 0.5

**Relevance to Project**:
- Lighting normalization is critical
- Regression approach may be better than classification
- Monk scale works well in practice

**Implementation Notes**:
- Consider regression instead of pure classification
- Need preprocessing for lighting normalization

### Towards Automatic Skin Tone Classification in Facial Images

**Title**: Towards Automatic Skin Tone Classification in Facial Images
**Year**: Not specified (older work)
**Link**: https://link.springer.com/chapter/10.1007/978-3-319-68548-9_28

**Key Findings**:
- Individual Typology Angle (ITA) method
- Uses angle between L* (lightness) and b* (yellow-blue) in LAB color space
- LAB more suitable than RGB for human perception
- Best accuracy (87.06%) using LAB cell histograms with SVM classifier

**Relevance to Project**:
- LAB color space is standard for skin tone
- ITA method is computationally efficient
- Can use as baseline or complementary approach

**Implementation Notes**:
- Convert RGB to LAB: separate luminance from color
- Use A and B channels for skin detection
- Could combine with CNN: LAB features + deep learning

---

## To Review

- [x] Recent papers on fairness in skin tone detection
- [x] Monk Skin Tone Scale documentation
- [x] LAB color space applications in skin analysis
- [ ] Specific TensorFlow Lite skin classification examples
- [ ] Flutter ML Kit integration detailed tutorials
- [ ] Google's Monk Skin Tone Examples (MST-E) dataset documentation

## Related Work

### Existing Applications
- **Google Image Search**: Uses Monk Skin Tone Scale for makeup search refinement
- **Various skincare apps**: Most lack transparency about algorithms
- **Medical imaging apps**: Focus on disease detection, showing significant bias issues

### Open Source Projects
- **SkinTone in The Wild (STW) dataset**: Monk-scale annotated faces
- **Google ML Kit**: Face detection for Flutter/mobile
- **TensorFlow Lite examples**: General image classification

### Flutter Integration Resources
- **face_recognition_auth package**: TFLite + ML Kit integration
- **2024-2025 tutorials**: Updated for latest Flutter/TFLite versions
- **ML Kit + TFLite pattern**: ML Kit detects faces → crop → TFLite classifies

## Summary of Findings

### Critical Insights

**1. Fairness is Paramount**
- Existing skin tone detection systems show significant bias
- Models perform 3x better on lighter skin tones (dataset imbalance)
- Less than 2.1% of datasets include skin tone annotations
- Fairness issues reduce trust and adoption

**2. Monk Skin Tone Scale is Preferred**
- 10 categories vs 6 (Fitzpatrick)
- Designed specifically for computer vision
- Better representation of darker tones
- Growing industry adoption (Google, research community)

**3. Technical Approach: Hybrid Methods Work Best**
- LAB color space superior to RGB for skin tone
- Individual Typology Angle (ITA) method: simple, effective baseline (87% accuracy)
- CNN-based approaches: higher accuracy with proper training
- Regression may outperform classification
- Lighting normalization is critical

**4. Implementation Path**
- Flutter + TensorFlow Lite is well-supported
- ML Kit for face detection (fast, reliable)
- TFLite for custom classification model
- On-device processing is feasible
- Real-time performance achievable

**5. Dataset Requirements**
- Need balanced representation across all skin tones
- STW dataset: Monk-scale annotated, balanced
- Casual Conversations v2: Diverse, multiple scales (Fitzpatrick + Monk)
- Must evaluate separately for each tone category

### Recommended Approach

**Phase 1: Baseline**
- Use ML Kit for face detection
- Implement ITA method (LAB color space)
- Simple classification into 3-5 categories
- Validate feasibility

**Phase 2: ML Model**
- Train/fine-tune CNN on balanced dataset (STW or Casual Conversations)
- Use Monk 10-class scale
- Convert to TensorFlow Lite
- Integrate with Flutter app

**Phase 3: Optimization**
- Lighting normalization preprocessing
- Model quantization for mobile
- Fairness evaluation across all tones
- Consider regression vs classification

### Key Risks & Mitigations

**Risk**: Model bias toward lighter tones
- *Mitigation*: Use balanced dataset, measure performance per category, report fairness metrics

**Risk**: Poor performance in varied lighting
- *Mitigation*: Lighting normalization, diverse training data, user guidance for image capture

**Risk**: On-device model insufficient accuracy
- *Mitigation*: Start with baseline (ITA), validate early, have backend fallback option

### Next Steps

1. Review specific datasets (STW, Casual Conversations v2)
2. Access/download chosen dataset
3. Implement ITA baseline in Python (proof of concept)
4. Test Flutter + ML Kit face detection
5. Evaluate TFLite model options
