# Datasets for Skin Tone Detection

This document catalogs available datasets for training and evaluating skin tone detection models.

## Dataset Requirements

- Diverse representation across skin tones
- Facial images with clear skin visibility
- Varied lighting conditions
- Licensing compatible with academic use
- Sufficient size for training/evaluation

## Available Datasets

### Dataset Template
```
**Name**: [Dataset Name]
**Source**: [URL]
**Size**: [Number of images]
**Diversity**: [Description of skin tone representation]
**Labels**: [What annotations are included]
**License**: [Usage rights]
**Access**: [How to obtain]

**Pros**:
-

**Cons**:
-

**Relevance**: [How we might use this]
```

---

## Reviewed Datasets

### SkinTone in The Wild (STW) Dataset

**Name**: SkinTone in The Wild (STW)
**Source**: https://www.researchgate.net/publication/386403126
**Size**: Size not specified (merged from multiple datasets)
**Diversity**: Balanced across all 10 Monk Skin Tone categories
**Labels**: 10-class Monk Skin Tone (MST) annotations
**License**: Research/academic use (check specific terms)
**Access**: Through research publication

**Pros**:
- Specifically designed for fairness in skin tone classification
- Balanced: equal number of images across all Monk scale classes
- Already annotated with Monk scale
- Purpose-built for ML training

**Cons**:
- Relatively new (2024)
- Access may require contacting authors
- Size unclear from search results

**Relevance**: **HIGH** - Ideal for our project. Monk-scale annotations, balanced dataset addressing bias concerns.

---

### Meta Casual Conversations v2 Dataset

**Name**: Casual Conversations v2
**Source**: https://ai.meta.com/research/publications/the-casual-conversations-v2-dataset/
**Size**: 26,467 videos of 5,567 unique participants (~5 videos per person)
**Diversity**: Highly diverse - recorded in 7 countries (Brazil, India, Indonesia, Mexico, Vietnam, Philippines, USA)
**Labels**:
- Dual annotations: Fitzpatrick (6-tone) AND Monk (10-tone) scales
- Age, gender, language/dialect, disability status, physical attributes, geo-location
**License**: Participants consented for fairness assessment research
**Access**: Available for fairness research (https://socialmediaarchive.org/record/18)

**Pros**:
- Extremely diverse global representation
- Both Fitzpatrick and Monk annotations
- Rich metadata beyond skin tone
- Specifically designed for fairness evaluation
- Video data (multiple frames per person)
- Ethically sourced with explicit consent

**Cons**:
- Video format may need preprocessing
- Large download size
- May be overkill for our simpler use case

**Relevance**: **VERY HIGH** - Best for comprehensive fairness evaluation. Dual-scale annotations perfect for academic rigor.

---

### IBM Diversity in Faces (DiF) Dataset

**Name**: Diversity in Faces (DiF)
**Source**: https://research.ibm.com/blog/diversity-in-faces
**Size**: 1 million annotated facial images
**Diversity**: Designed to address bias in light-skinned, male-dominated datasets
**Labels**: 10 coding schemes including craniofacial features, age, gender predictions
**License**: Originally open-source (2019)
**Access**: **CAUTION** - Dataset caused legal backlash and class action lawsuit

**Pros**:
- Very large scale (1M images)
- Multiple annotation types
- Publicly available images from YFCC-100M Creative Commons

**Cons**:
- Legal and ethical concerns (class action lawsuit)
- No explicit skin tone scale annotations (has race/ethnicity instead)
- **NOT RECOMMENDED** due to ethical/legal issues

**Relevance**: **LOW** - Despite size, ethical concerns and lack of skin tone scale make it unsuitable.

---

### FairFace Dataset

**Name**: FairFace
**Source**: https://github.com/joojs/fairface (papers on OpenAccess CVF)
**Size**: Balanced across race, gender, and age
**Diversity**: Specifically designed for balanced representation
**Labels**: Race, gender, age attributes
**License**: Research use
**Access**: Publicly available on GitHub

**Pros**:
- Designed explicitly for fairness
- Balanced representation
- Well-documented
- Actively used in research

**Cons**:
- Race labels, not skin tone scale
- Would need manual re-annotation for Monk/Fitzpatrick

**Relevance**: **MEDIUM** - Good for diversity, but lacks skin tone annotations. Could use for transfer learning base.

---

### UTKFace Dataset

**Name**: UTKFace
**Source**: https://susanqq.github.io/UTKFace/ | Kaggle
**Size**: 20,000+ face images
**Diversity**: Annotations for age, gender, and ethnicity (White, Black, Asian, Indian, Others)
**Labels**: Age, gender, race/ethnicity
**License**: Research use
**Access**: Publicly available

**Pros**:
- Well-known benchmark dataset
- Moderate size
- Easy access (Kaggle)

**Cons**:
- Ethnicity labels, not skin tone
- No Monk/Fitzpatrick annotations
- Less diverse than newer datasets

**Relevance**: **LOW-MEDIUM** - Could use for pre-training, but needs skin tone annotations.

---

### CelebA Dataset

**Name**: CelebA (CelebFaces Attributes)
**Source**: TensorFlow Datasets catalog
**Size**: 202,599 face images
**Diversity**: **Poor** - mainly lighter skin tones
**Labels**: 5 landmarks, 40 binary attributes (no race/skin tone)
**License**: Research use
**Access**: TensorFlow Datasets, widely available

**Pros**:
- Very large
- Many attributes
- Easy integration with TensorFlow

**Cons**:
- **Biased toward lighter skin tones**
- No skin tone annotations
- Not suitable for fairness-focused work

**Relevance**: **LOW** - Bias issues make it unsuitable for our fairness-critical application.

---

### Google Monk Skin Tone Examples (MST-E)

**Name**: Monk Skin Tone Examples (MST-E)
**Source**: Google Research
**Size**: Example dataset (size not specified)
**Diversity**: Specifically designed to represent all 10 Monk scale tones
**Labels**: Monk Skin Tone Scale (10 classes)
**License**: Publicly available for training annotators
**Access**: https://research.google/blog/

**Pros**:
- Official Monk scale reference
- Designed for training human annotators
- High quality examples
- Direct from scale creator (Google + Dr. Monk)

**Cons**:
- Likely small (example set, not training set)
- May not be sufficient alone for model training

**Relevance**: **MEDIUM-HIGH** - Essential for understanding Monk scale. Use for validation/reference, possibly combine with larger dataset.

---

## Candidate Datasets Not Yet Reviewed

### Still to Investigate
- [ ] FFHQ (Flickr-Faces-HQ) - High quality faces, check diversity
- [ ] Labeled Faces in the Wild (LFW) - Well-known, but likely biased
- [ ] Additional dermatology datasets with skin tone

## Evaluation Strategy

### Test Set Composition
- **Balanced representation**: Equal samples from each Monk scale category (or as close as possible)
- **Stratified split**: Ensure train/val/test maintain same distribution
- **Minimum samples**: At least 50-100 images per category in test set
- **Lighting diversity**: Include varied lighting conditions in test set

### Metrics for Each Skin Tone Category
- **Per-category accuracy**: Report accuracy for each of 10 Monk scale categories
- **Confusion matrix**: Identify which categories are confused with each other
- **Precision/Recall/F1**: Per-category performance metrics
- **Fairness metrics**:
  - Performance parity: variance in accuracy across categories should be <5%
  - Worst-category accuracy should be >85% of best-category accuracy
  - False positive/negative rates per category

### Evaluation Protocol
1. Split dataset: 70% train, 15% validation, 15% test
2. Ensure stratification by skin tone category
3. Report aggregate metrics AND per-category breakdown
4. Visualize performance with confusion matrix
5. Test on held-out diverse lighting conditions

## Data Collection Considerations

### Privacy and Consent
- Use only datasets with explicit participant consent
- **Preference**: Datasets designed for fairness research (consented use)
- Avoid scrapped/non-consensual datasets
- No storage/transmission of user images in our app (on-device only)

### Ethical Data Sourcing
- **Recommended**: Casual Conversations v2, STW (ethically sourced, consented)
- **Avoid**: IBM DiF (legal issues), CelebA (non-diverse)
- Prefer datasets that compensated participants
- Academic use only - not commercial

### Bias Mitigation Strategies
- **Balanced training**: Equal representation across all categories
- **Augmentation**: Data augmentation if category imbalanced
- **Validation**: Continuous monitoring of per-category performance during training
- **Fairness constraints**: Consider fairness-aware training techniques
- **Documentation**: Transparent reporting of dataset composition and limitations

## Dataset Recommendation

### Primary Choice: Meta Casual Conversations v2

**Rationale**:
1. **Dual annotations**: Both Fitzpatrick and Monk scales allow comparison
2. **Highly diverse**: 7 countries, 5,567 participants
3. **Ethically sourced**: Explicit consent for fairness research
4. **Well-documented**: Meta research publication, active support
5. **Academic credibility**: From reputable source, widely cited

**Implementation**:
- Download Casual Conversations v2 dataset
- Extract frames from videos (sample 1-3 frames per video)
- Use Monk scale annotations (10 classes)
- Balance dataset if needed through sampling or augmentation

### Secondary Choice: SkinTone in The Wild (STW)

**Rationale**:
1. **Purpose-built**: Specifically for Monk-scale classification
2. **Pre-balanced**: Equal samples per category
3. **Already annotated**: Ready to use
4. **Recent research**: 2024 publication

**Challenge**: May need to contact authors for access

### Hybrid Approach (Recommended)

- **Primary training**: Casual Conversations v2 (larger, diverse)
- **Validation/calibration**: MST-E (official Monk examples)
- **Additional data if needed**: STW (balanced) or FairFace (diverse base)

## Summary

After reviewing 7 major datasets, **Meta Casual Conversations v2** emerges as the best choice for our project:

**Strengths**:
- Largest diverse dataset with both Monk and Fitzpatrick annotations
- Ethically sourced with explicit consent
- Global representation (7 countries)
- Video format provides multiple angles per person
- Designed specifically for fairness evaluation

**Next Actions**:
1. ✅ Access Casual Conversations v2 dataset
2. ✅ Download MST-E for Monk scale reference
3. Preprocess videos → extract frames
4. Verify annotation quality
5. Create balanced train/val/test splits
6. Document dataset statistics (distribution, lighting, etc.)

**Fallback Options**:
- If CC v2 too large: Use STW (contact authors)
- If need more data: Combine with FairFace (but re-annotate with Monk scale)
- For pre-training: Could use FairFace then fine-tune on CC v2
