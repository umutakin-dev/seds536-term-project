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

**Current Phase**: Research & Planning

See [docs/project-management/timeline.md](docs/project-management/timeline.md) for detailed milestones.

## Repository Structure

```
seds536-term-project/
├── README.md                          # This file
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
├── research-papers/                   # Academic papers (gitignored)
└── app/                               # Flutter application (future)
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

(To be populated once development begins)

## Testing

(To be populated once testing framework is established)

## References

See [docs/research/literature-review.md](docs/research/literature-review.md) for academic references and related work.

## License

This is an academic project for SEDS536 course.

## Acknowledgments

Course: SEDS536 - Image Understanding
