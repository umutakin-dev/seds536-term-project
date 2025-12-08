# Project Roadmap & Issue Planning

**Project**: Skin Tone Detection for Skincare Recommendations
**Presentation Date**: January 6, 2026
**Timeline**: 10 weeks (October 27, 2025 - January 6, 2026)
**Current Date**: October 27, 2025
**Days Remaining**: 71 days

## Project Milestones

The project is organized into 4 major milestones:

### M1: App Foundation & Dataset Ready
**Due**: November 24, 2025 (Week 5)
**Description**: Flutter app foundation complete with camera integration. Dataset downloaded and preprocessed. Ready to start ML training.
**Includes**: Epic 1, Epic 2, and all child issues

### M2: ML Pipeline Complete
**Due**: December 15, 2025 (Week 8)
**Description**: Face detection integrated. Skin tone classification model trained and optimized. Model converted to TFLite format.
**Includes**: Epic 3, Epic 4

### M3: Integration Complete
**Due**: December 22, 2025 (Week 9)
**Description**: TFLite model integrated into Flutter app. Complete inference pipeline working on-device. Results UI implemented.
**Includes**: Epic 5, Epic 6

### M4: Final Presentation
**Due**: January 6, 2026 (Week 12) - **Presentation Day**
**Description**: Project complete. All testing done. Documentation finished. Presentation ready.
**Includes**: Epic 7, Epic 8

## Design System

**Using**: [Google Stitch Design System](https://stitch.withgoogle.com)
- Stitch is Google's design system for building inclusive products
- Emphasizes accessibility, diversity, and representation
- Perfect fit for a skin tone detection app focused on fairness

## Project Phases

### Phase 1: Setup & Research (Weeks 1-2) ✅
- [x] Initialize project repository
- [x] Set up Flutter development environment
- [x] Configure GitHub Project board
- [x] Initial research on skin tone detection methods
- [x] Document dataset options and ML approaches

### Phase 2: Design & Flutter Foundation (Weeks 3-4)
**Epic**: Build core Flutter application structure with Stitch design

**Features**:
- Study Stitch design principles and components
- Design app UI/UX following Stitch guidelines
- Camera integration and permissions
- Image capture UI
- Basic navigation structure
- App theming aligned with Stitch

### Phase 3: ML Research & Dataset Preparation (Weeks 3-5)
**Epic**: Prepare ML training pipeline

**Features**:
- Download and analyze Casual Conversations v2 dataset
- Implement data preprocessing pipeline
- Implement baseline ITA method (proof of concept)
- Set up Windows GPU training environment

### Phase 4: Face Detection Integration (Weeks 5-6)
**Epic**: Integrate face detection

**Features**:
- Evaluate face detection libraries (ML Kit, MediaPipe)
- Integrate chosen face detection solution
- Extract face regions from images
- Handle edge cases (no face, multiple faces, poor lighting)

### Phase 5: ML Model Development (Weeks 6-8)
**Epic**: Train and optimize skin tone classification model

**Features**:
- Train CNN model for Monk scale classification
- Implement fairness evaluation across all skin tones
- Optimize model performance
- Convert model to TensorFlow Lite format

### Phase 6: ML Integration (Weeks 8-9)
**Epic**: Integrate ML model into Flutter app

**Features**:
- Add TFLite dependency to Flutter
- Implement on-device inference
- Create prediction pipeline (image → face detection → skin tone classification)
- Performance optimization for mobile

### Phase 7: Results & Recommendations (Week 10)
**Epic**: Display results and provide recommendations (Stitch-compliant UI)

**Features**:
- Design results screen UI following Stitch principles
- Display detected Monk scale value with inclusive visualization
- Provide skincare/makeup recommendations
- Add confidence scores and explanations

### Phase 8: Testing, Documentation & Presentation (Weeks 11-12)
**Epic**: Finalize and document project

**Deadline**: January 6, 2026 (Presentation Day)

**Features**:
- Comprehensive testing across skin tones
- Performance testing on iPhone
- Write final project report
- Create demo video
- Prepare presentation slides
- Code cleanup and documentation

## Epics Breakdown

### Epic 1: Design & Flutter App Foundation 📱
**Priority**: 🔥 High
**Effort**: 🐺 5 (Large)
**Status**: Todo

**Child Issues**:
1. Study Stitch design system and principles (🐭 2 - Small) - comp:flutter 📱, comp:documentation 📝
2. Design app UI/UX mockups following Stitch guidelines (🐱 3 - Medium) - comp:flutter 📱
3. Set up camera integration and permissions (🐱 3 - Medium) - comp:flutter 📱
4. Implement image capture UI with camera preview (🐭 2 - Small) - comp:flutter 📱
5. Create basic navigation structure (🐭 2 - Small) - comp:flutter 📱
6. Implement app theme aligned with Stitch design (🐱 3 - Medium) - comp:flutter 📱

---

### Epic 2: ML Training Pipeline 🎓
**Priority**: 🔥 High
**Effort**: 🐋 13 (Very Large)
**Status**: Todo

**Child Issues**:
1. Download Casual Conversations v2 dataset (🐭 2 - Small) - comp:ml-training 🎓, comp:research 📚
2. Analyze dataset structure and annotations (🐱 3 - Medium) - comp:ml-training 🎓, comp:research 📚
3. Implement data preprocessing pipeline (🐺 5 - Large) - comp:ml-training 🎓
4. Set up Windows GPU training environment (conda, PyTorch, CUDA) (🐱 3 - Medium) - comp:ml-training 🎓
5. Implement baseline ITA method for comparison (🐱 3 - Medium) - comp:ml-training 🎓

---

### Epic 3: Face Detection Integration 🔗
**Priority**: 🔥 High
**Effort**: 🐺 5 (Large)
**Status**: Todo

**Child Issues**:
1. Research face detection libraries (ML Kit vs MediaPipe) (🐭 2 - Small) - comp:ml-integration 🔗, comp:research 📚
2. Integrate ML Kit / MediaPipe into Flutter (🐱 3 - Medium) - comp:ml-integration 🔗
3. Implement face extraction from images (🐭 2 - Small) - comp:ml-integration 🔗
4. Handle edge cases (no face, multiple faces, poor lighting) (🐱 3 - Medium) - comp:ml-integration 🔗

---

### Epic 4: Skin Tone Classification Model 🎓
**Priority**: 🔥 High
**Effort**: 🐋 13 (Very Large)
**Status**: Todo

**Child Issues**:
1. Design CNN architecture for Monk scale classification (🐺 5 - Large) - comp:ml-training 🎓
2. Train initial model on Casual Conversations v2 (🐻 8 - Extra Large) - comp:ml-training 🎓
3. Implement fairness evaluation across all Monk scale classes (🐺 5 - Large) - comp:ml-training 🎓
4. Optimize model performance and size (🐺 5 - Large) - comp:ml-training 🎓
5. Convert trained model to TensorFlow Lite (🐭 2 - Small) - comp:ml-training 🎓

---

### Epic 5: ML Model Integration into Flutter 🔗
**Priority**: 🔸 Medium
**Effort**: 🐺 5 (Large)
**Status**: Backlog

**Child Issues**:
1. Add TFLite Flutter dependency (🐜 1 - Trivial) - comp:ml-integration 🔗
2. Load TFLite model in Flutter app (🐭 2 - Small) - comp:ml-integration 🔗
3. Implement inference pipeline (image → preprocessing → prediction) (🐱 3 - Medium) - comp:ml-integration 🔗
4. Performance optimization for on-device inference (🐱 3 - Medium) - comp:ml-integration 🔗
5. Add loading states and error handling (🐭 2 - Small) - comp:ml-integration 🔗

---

### Epic 6: Results & Recommendations UI (Stitch Design) 📱
**Priority**: 🔸 Medium
**Effort**: 🐺 5 (Large)
**Status**: Backlog

**Child Issues**:
1. Design results screen mockups following Stitch principles (🐭 2 - Small) - comp:flutter 📱
2. Implement results screen UI with Stitch components (🐱 3 - Medium) - comp:flutter 📱
3. Display Monk scale value with inclusive visualization (🐭 2 - Small) - comp:flutter 📱
4. Create skincare/makeup recommendations database (🐱 3 - Medium) - comp:flutter 📱, comp:research 📚
5. Implement recommendation display logic (🐭 2 - Small) - comp:flutter 📱

---

### Epic 7: Testing & Quality Assurance ✅
**Priority**: 🔸 Medium
**Effort**: 🐻 8 (Extra Large)
**Status**: Backlog

**Child Issues**:
1. Test app across different lighting conditions (🐱 3 - Medium) - comp:flutter 📱, comp:ml-integration 🔗
2. Test fairness: ensure equal performance across all skin tones (🐺 5 - Large) - comp:ml-integration 🔗
3. Performance testing on physical iPhone (🐭 2 - Small) - comp:flutter 📱
4. User experience testing and refinement (🐱 3 - Medium) - comp:flutter 📱
5. Code review and cleanup (🐱 3 - Medium) - comp:flutter 📱, comp:ml-integration 🔗

---

### Epic 8: Documentation & Presentation 📝
**Priority**: 🔸 Medium
**Effort**: 🐺 5 (Large)
**Status**: Backlog

**Child Issues**:
1. Write final project report (methodology, results, evaluation) (🐺 5 - Large) - comp:documentation 📝
2. Create demo video showing app functionality (🐭 2 - Small) - comp:documentation 📝
3. Document code and architecture (🐱 3 - Medium) - comp:documentation 📝
4. Prepare presentation slides (🐭 2 - Small) - comp:documentation 📝
5. Document Stitch design implementation and rationale (🐭 2 - Small) - comp:documentation 📝

---

## Standalone Issues (Not Part of Epics)

### Research & Documentation
- Update literature review with new findings (🐭 2 - Small) - comp:research 📚, comp:documentation 📝
- Document Monk vs Fitzpatrick scale comparison (🐭 2 - Small) - comp:research 📚, comp:documentation 📝
- Research Stitch design system best practices (🐭 2 - Small) - comp:research 📚, comp:flutter 📱

### Infrastructure & Tooling
- Set up CI/CD for automated testing (🐱 3 - Medium) - comp:flutter 📱
- Configure code formatting and linting (🐜 1 - Trivial) - comp:flutter 📱

---

## Next Actions

**Immediate (This Week)**:
1. Create Epic issues in GitHub
2. Create child issues for Epic 1 (Design & Flutter App Foundation)
3. Study Stitch design system
4. Create child issues for Epic 2 (ML Training Pipeline)
5. Start work on camera integration

**Short Term (Next 2 Weeks)**:
1. Design app UI/UX following Stitch guidelines
2. Complete Flutter app foundation
3. Download and preprocess dataset
4. Set up Windows ML training environment
5. Implement baseline ITA method

**Medium Term (Weeks 5-8)**:
1. Integrate face detection
2. Train and evaluate ML model
3. Integrate model into Flutter app

**Long Term (Weeks 9-12)**:
1. Complete UI and recommendations (Stitch-compliant)
2. Comprehensive testing
3. Final documentation and presentation

---

## Design Principles (Stitch)

Key principles from Stitch to apply:
- **Inclusive**: Design for diverse users and skin tones
- **Accessible**: Ensure app is usable by everyone
- **Respectful**: Handle sensitive information about appearance with care
- **Clear**: Provide clear explanations of how the app works
- **Empowering**: Help users understand and trust the technology

---

## Issue Creation Checklist

When creating issues, remember to set:
- [ ] Title (clear and descriptive, NO "Epic:" or "Feature:" prefixes)
- [ ] Description (with acceptance criteria)
- [ ] Type (👑 Epic, ✨ Feature, ✅ Task, 🐞 Bug, 🧹 Chore, 📝 Docs)
- [ ] Labels (component labels like comp:flutter 📱)
- [ ] Status (Backlog or Todo)
- [ ] Priority (🔹 Low, 🔸 Medium, 🔥 High, 🚨 Critical)
- [ ] Effort (🐜 1, 🐭 2, 🐱 3, 🐺 5, 🐻 8, 🐋 13, 🤔 ?)
- [ ] Target Date (if applicable)
- [ ] Milestone (M1, M2, M3, or M4)
- [ ] Assignee (umutakin-dev)
- [ ] Parent issue (for child issues - set via web UI)

---

## Notes

- **Design System**: Google Stitch (https://stitch.withgoogle.com)
- **Presentation Date**: January 6, 2026 (71 days from now)
- **Physical iPhone**: Required for camera testing (iOS Simulator inadequate)
- **Windows Desktop**: RTX 4070 Ti Super (16GB VRAM) for ML training
- **MacBook Air M1**: For Flutter development
- **Focus on Fairness**: Equal performance across all skin tones
- **Emphasis on Inclusive Design**: Following Stitch principles
