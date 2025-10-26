# Research Papers

This directory contains PDF papers related to skin tone detection, fairness in ML, and mobile ML deployment. The PDFs are not tracked in git (too large), but this README documents which papers we have and where to download them.

## Papers Collection

### Skin Tone Detection & Fairness

**1. Understanding skin color bias in deep learning-based skin lesion segmentation**
- **Authors**: TBD
- **Year**: 2024
- **Link**: https://pubmed.ncbi.nlm.nih.gov/38290289/
- **Local File**: `understanding-skin-bias-2024.pdf`
- **Notes**: Key findings on bias in medical imaging, dataset imbalances

**2. Skin Cancer Machine Learning Model Tone Bias**
- **Authors**: TBD
- **Year**: 2024
- **Link**: https://arxiv.org/abs/2410.06385
- **Local File**: `skin-cancer-ml-bias-2024.pdf`
- **Notes**: Shows 3x performance difference light vs dark tones

**3. Enhancing Fairness in Machine Learning: Skin Tone Classification Using the Monk Skin Tone Scale**
- **Authors**: TBD
- **Year**: 2024
- **Link**: https://www.researchgate.net/publication/386403126
- **Local File**: `monk-scale-classification-2024.pdf`
- **Notes**: SkinTone in The Wild (STW) dataset, balanced Monk scale

**4. Skin Tone Estimation under Diverse Lighting Conditions**
- **Authors**: TBD
- **Year**: 2024
- **Link**: https://pubmed.ncbi.nlm.nih.gov/38786563/
- **Local File**: `skin-tone-lighting-2024.pdf`
- **Notes**: CNN-based, Monk scale, regression approach

**5. Towards Automatic Skin Tone Classification in Facial Images**
- **Authors**: TBD
- **Year**: TBD
- **Link**: https://link.springer.com/chapter/10.1007/978-3-319-68548-9_28
- **Local File**: `automatic-classification-springer.pdf`
- **Notes**: ITA method, LAB color space, 87% accuracy with SVM

### Monk Skin Tone Scale

**6. Consensus and Subjectivity of Skin Tone Annotation for ML Fairness**
- **Authors**: Google Research
- **Year**: 2023
- **Link**: https://arxiv.org/abs/2305.09073
- **Local File**: `monk-consensus-google-2023.pdf`
- **Notes**: Google's research on Monk scale, MST-E dataset

### Datasets

**7. Towards Measuring Fairness in AI: the Casual Conversations Dataset**
- **Authors**: Meta AI
- **Year**: 2021
- **Link**: https://arxiv.org/abs/2104.02821
- **Local File**: `casual-conversations-meta-2021.pdf`
- **Notes**: Original Casual Conversations v1

**8. The Casual Conversations v2 Dataset**
- **Authors**: Meta AI
- **Year**: 2023
- **Link**: https://ai.meta.com/research/publications/the-casual-conversations-v2-dataset/
- **Local File**: `casual-conversations-v2-2023.pdf`
- **Notes**: 26K videos, 7 countries, Monk + Fitzpatrick annotations

**9. FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age**
- **Authors**: TBD
- **Year**: 2021
- **Link**: https://arxiv.org/abs/1908.04913
- **Local File**: `fairface-dataset-2021.pdf`
- **Notes**: Balanced facial dataset

### Mobile ML

**10. [Add mobile ML / TensorFlow Lite papers as needed]**
- **Authors**:
- **Year**:
- **Link**:
- **Local File**:
- **Notes**:

## How to Download

1. Click the link for the paper
2. Download PDF
3. Save to this directory with the filename shown above
4. Update this README with author names and any additional notes

## Organization

Papers are organized by topic:
- **Skin Tone Detection & Fairness**: Core methods and bias research
- **Monk Skin Tone Scale**: Documentation of the scale we're using
- **Datasets**: Papers describing datasets we might use
- **Mobile ML**: TensorFlow Lite, on-device inference, optimization

## Citation Format

When citing in our documentation, use:
```
[Author et al., Year] Title. Journal/Conference. DOI/ArXiv link
```

## Notes

- PDFs are gitignored (too large for git)
- This README is tracked in git for reference
- Both Mac and Windows machines should download papers as needed
- Cross-reference with `docs/research/literature-review.md` for detailed notes
