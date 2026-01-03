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

---

## Dataset Investigation Log (January 2026)

This section documents our hands-on investigation of dataset options for the SEDS536 term project.

### Investigation Timeline

**Date**: January 2, 2026  
**Deadline**: January 6, 2026 (presentation)  
**Goal**: Find a practical dataset for training a Monk scale skin tone classifier

### Datasets Investigated

#### 1. Meta Casual Conversations v2 - Access Granted ✅

**Access Status**: Approved (same-day approval)  
**Download Page**: https://ai.meta.com/datasets/casual-conversations-v2-dataset/

**Actual Download Contents**:
- `CCv2_annotations.zip` - Monk/Fitzpatrick annotations
- `CCv2_samples.zip` - Sample subset
- `CCv2_frames_part_1.zip` through `CCv2_frames_part_5.zip` - Pre-extracted frames
- `CCv2_part_1.zip` through `CCv2_part_80.zip` - Full video dataset

**Critical Finding**:
- **80 video zip files**, each 30-60 GB = **2.4 - 4.8 TB total**
- Far too large for a 4-day timeline
- However, annotations and samples are manageable

**Decision**: Download only `CCv2_annotations.zip` + `CCv2_samples.zip` for minimal viable approach.

---

#### 2. Google MST-E Dataset - Not Suitable ❌

**URL**: https://skintone.google/mste-dataset  
**Contents**: 1,515 images + 31 videos of 19 subjects across all 10 Monk tones

**Critical Finding**:
> **"The images cannot be used to train machine learning models"**

The dataset is licensed **only for training human annotators**, not ML models. This makes it unsuitable for our use case despite being perfectly aligned with Monk scale.

**Decision**: Cannot use for ML training. Could reference for scale understanding only.

---

#### 3. UTKFace Dataset - Limited Usefulness ⚠️

**URL**: https://www.kaggle.com/datasets/jangedoo/utkface-new  
**Size**: 20,000+ face images (~200MB)  
**Labels**: Age, gender, ethnicity (5 classes: White, Black, Asian, Indian, Others)

**Finding**:
- Has **ethnicity labels**, not skin tone labels
- Easy to download from Kaggle
- Could compute ITA (Individual Typology Angle) values to approximate skin tone

**Decision**: Not ideal - ethnicity ≠ skin tone. Could use as fallback with ITA computation.

---

#### 4. FFHQ (Flickr-Faces-HQ) - No Labels ❌

**URL**: https://github.com/NVlabs/ffhq-dataset  
**Size**: 70,000 high-quality faces  
**Download**: 89GB for 1024x1024 images, 1.95GB for 128x128 thumbnails

**Finding**:
- **No skin tone labels** at all
- Beautiful high-resolution images but requires manual annotation
- Inherits Flickr's demographic biases

**Decision**: Not suitable - no labels and too large for our timeline.

---

#### 5. SCIN Dataset - Wrong Domain ❌

**URL**: https://github.com/google-research-datasets/scin  
**Size**: 10,000+ images  
**Labels**: Includes estimated Monk Skin Tone (eMST)

**Finding**:
- Has Monk scale labels ✅
- But images are of **dermatology conditions** (skin diseases)
- Not suitable for general face/skin tone detection

**Decision**: Wrong domain - dermatology images, not face photos.

---

### Alternative Approaches Considered

#### ITA-Based Approach (No Dataset Needed)

**Method**: Individual Typology Angle calculation using LAB color space  
**Formula**: `ITA = arctan((L* - 50) / b*) × 180/π`

**Pros**:
- No training data required
- Scientifically validated in dermatology research
- Can map ITA values to Monk scale using published thresholds
- Works immediately on existing face extraction pipeline

**Cons**:
- Less accurate than ML approach
- Sensitive to lighting conditions
- Rule-based, not learned

**Decision**: Keep as fallback option for demo if ML approach hits issues.

---

### Final Decision

**Chosen Approach**: Meta Casual Conversations v2 (Minimal)

**What to Download**:
1. `CCv2_annotations.zip` - Contains Monk scale labels for all 26,467 videos
2. `CCv2_samples.zip` - Sample subset for quick experimentation

**Rationale**:
1. **Has actual Monk scale annotations** - gold standard labels
2. **Ethically sourced** - participant consent for ML fairness research
3. **Manageable size** - annotations + samples are small downloads
4. **Academic credibility** - published dataset from Meta AI
5. **Already have access** - approved same-day

**Training Strategy**:
- Use sample subset for initial model development
- If more data needed, can download 1-2 frame parts
- Document full dataset access for future work

**Fallback Plan**:
- If samples insufficient: Implement ITA-based approach for demo
- Mention ML approach as validated methodology in presentation
- Show dataset access and research as evidence of thorough approach

---

### Lessons Learned

1. **Dataset size matters** - TB-scale datasets not practical for short timelines
2. **License restrictions vary** - MST-E looked perfect but can't be used for ML
3. **Domain specificity** - SCIN has Monk labels but wrong image type
4. **Access time** - Meta approved same-day, but others may take weeks
5. **Annotations vs Images** - Sometimes annotations alone are valuable

---

### References

- Meta Casual Conversations v2: https://ai.meta.com/datasets/casual-conversations-v2-dataset/
- Google MST-E: https://skintone.google/mste-dataset
- UTKFace: https://www.kaggle.com/datasets/jangedoo/utkface-new
- FFHQ: https://github.com/NVlabs/ffhq-dataset
- SCIN: https://github.com/google-research-datasets/scin

---

## Dataset Preparation Log (January 2, 2026)

This section documents the actual dataset preparation work completed.

### What We Actually Downloaded

After initial analysis revealed the samples were insufficient (only 6 videos), we downloaded:

| File | Size | Contents |
|------|------|----------|
| `CCv2_annotations.zip` | 11.7 MB | JSON annotations for all 26,467 videos |
| `CCv2_frames_part_1.zip` | ~10 GB | Pre-extracted frames for subjects 0000-1113 |
| `CCv2_frames_part_2.zip` | ~10 GB | Pre-extracted frames for subjects 1114-2227 |
| `CCv2_frames_part_3.zip` | ~10 GB | Pre-extracted frames for subjects 2228-3340 |
| `CCv2_frames_part_4.zip` | ~10 GB | Pre-extracted frames for subjects 3341-4453 |
| `CCv2_frames_part_5.zip` | ~10 GB | Pre-extracted frames for subjects 4454-5566 |
| **Total** | **~50 GB** | **264,670 frames from 5,567 subjects** |

**Rationale for downloading all 5 parts**: The Monk scale distribution is heavily imbalanced, with rare tones (Scale 1, 9, 10) spread across different parts. Downloading only Part 1 would miss critical minority class samples.

### Dataset Statistics

#### Annotation Structure

Each video annotation contains:
```json
{
  "video_name": "0000_portuguese_nonscripted_1.mp4",
  "subject_id": "0000",
  "monk_skin_tone": {
    "scale": "scale 5",
    "confidence": "medium"
  },
  "fitzpatrick_skin_tone": {
    "type": "type iii",
    "confidence": "medium"
  },
  // ... additional metadata (age, gender, location, etc.)
}
```

#### Frame Structure

Frames are organized by subject ID:
```
ccv2-frames-part-1/
├── 0000/
│   ├── 0000_portuguese_nonscripted_1_raw_frame00000381.jpg
│   ├── 0000_portuguese_nonscripted_1_raw_frame00000425.jpg
│   └── ...
├── 0001/
└── ...
```

#### Monk Scale Distribution (All 5 Parts)

| Scale | Subjects | Percentage | Notes |
|-------|----------|------------|-------|
| 1 | 30 | 0.5% | Very limited |
| 2 | 300 | 5.4% | |
| 3 | 747 | 13.4% | |
| 4 | 962 | 17.3% | |
| 5 | 2,502 | 44.9% | **Dominant class** |
| 6 | 723 | 13.0% | |
| 7 | 167 | 3.0% | |
| 8 | 91 | 1.6% | |
| 9 | 39 | 0.7% | Very limited |
| 10 | 6 | 0.1% | **Extremely limited** |

**Key Challenge**: Severe class imbalance - Scale 5 has 417x more subjects than Scale 10.

#### Confidence Distribution

| Confidence | Videos | Percentage |
|------------|--------|------------|
| High | 10,680 | 40.3% |
| Medium | 13,385 | 50.6% |
| Low | 2,402 | 9.1% |

### Preprocessing Pipeline

Created `training/scripts/preprocess_ccv2.py` with the following strategy:

1. **Load annotations** from JSON
2. **Filter by confidence** (keep high/medium only, ~91% of data)
3. **Group by Monk scale** (1-10)
4. **Split by subject** (not by frame) to prevent data leakage:
   - Train: 70% of subjects
   - Validation: 15% of subjects
   - Test: 15% of subjects
5. **Sample frames** (max 30 per subject to prevent dominance)
6. **Copy to organized structure**:
   ```
   training/data/ccv2_balanced/
   ├── train/
   │   ├── scale_1/
   │   ├── scale_2/
   │   └── ... (scale_10/)
   ├── val/
   └── test/
   ```

### Environment Setup

Created Python environment with uv:
```bash
uv venv --python 3.12
uv pip install torch torchvision pillow pandas numpy scikit-learn tqdm
```

### Files Created

| File | Purpose |
|------|---------|
| `training/scripts/preprocess_ccv2.py` | Dataset preprocessing pipeline |
| `training/configs/` | (For training configuration) |
| `training/notebooks/` | (For experimentation) |
| `.venv/` | Python virtual environment |

### Next Steps

1. ✅ **Run preprocessing** - Execute `preprocess_ccv2.py` to create balanced dataset
2. ✅ **Train classifier** - CNN model for 10-class Monk scale classification (see [training-experiments.md](training-experiments.md))
3. ✅ **Improve model** - 3-class grouping + face cropping → 78.6% accuracy (see [Experiment 2](training-experiments.md#experiment-2-3-class-with-face-cropped-images))
4. **Convert to TFLite** - For on-device inference in Flutter app
5. **Integrate with app** - Connect model to existing face extraction pipeline

---

## Preprocessing Results (January 3, 2026)

The preprocessing pipeline completed successfully, creating the balanced dataset structure.

### Output Location

```
training/data/ccv2_balanced/
├── train/
│   ├── scale_1/  ... scale_10/
├── val/
│   ├── scale_1/  ... scale_10/
└── test/
    ├── scale_1/  ... scale_10/
```

### Final Dataset Statistics

**Total frames**: 149,520

#### Training Set (104,510 frames)

| Scale | Frames | % of Train |
|-------|--------|------------|
| 1 | 570 | 0.5% |
| 2 | 5,520 | 5.3% |
| 3 | 14,410 | 13.8% |
| 4 | 17,870 | 17.1% |
| 5 | 47,260 | 45.2% |
| 6 | 13,230 | 12.7% |
| 7 | 2,970 | 2.8% |
| 8 | 1,850 | 1.8% |
| 9 | 710 | 0.7% |
| 10 | 120 | 0.1% |

#### Validation Set (22,290 frames)

| Scale | Frames | % of Val |
|-------|--------|----------|
| 1 | 120 | 0.5% |
| 2 | 1,160 | 5.2% |
| 3 | 3,090 | 13.9% |
| 4 | 3,790 | 17.0% |
| 5 | 10,090 | 45.3% |
| 6 | 2,840 | 12.7% |
| 7 | 620 | 2.8% |
| 8 | 390 | 1.7% |
| 9 | 150 | 0.7% |
| 10 | 30 | 0.1% |

#### Test Set (22,730 frames)

| Scale | Frames | % of Test |
|-------|--------|-----------|
| 1 | 150 | 0.7% |
| 2 | 1,230 | 5.4% |
| 3 | 3,150 | 13.9% |
| 4 | 3,860 | 17.0% |
| 5 | 10,180 | 44.8% |
| 6 | 2,850 | 12.5% |
| 7 | 680 | 3.0% |
| 8 | 420 | 1.8% |
| 9 | 180 | 0.8% |
| 10 | 30 | 0.1% |

### Key Observations

1. **Class imbalance persists**: Scale 5 dominates (~45%), scales 1, 9, 10 are rare (<1%)
2. **Consistent distribution**: Train/val/test splits maintain similar proportions (good stratification)
3. **Subject-level splitting**: Prevents data leakage between splits
4. **Minority classes**: Scale 10 has only 30 samples in val/test - will need careful handling

### Training Considerations

Due to severe class imbalance, training will need:
- **Weighted loss function**: Higher weights for minority classes (1, 9, 10)
- **Oversampling**: Augment minority classes during training
- **Evaluation metrics**: Focus on macro F1, per-class accuracy rather than overall accuracy
- **Class grouping (optional)**: Consider grouping into 3-4 super-classes if 10-class performance is poor

---
