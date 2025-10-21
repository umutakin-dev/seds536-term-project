# Project Requirements

## Functional Requirements

### Core Features (MVP)

#### FR-1: Image Input
- **FR-1.1**: User can capture image using device camera
- **FR-1.2**: User can upload image from device gallery
- **FR-1.3**: App validates image contains a detectable face
- **FR-1.4**: App provides feedback for poor quality images

#### FR-2: Skin Tone Detection
- **FR-2.1**: System detects face in input image
- **FR-2.2**: System classifies skin tone into predefined categories
- **FR-2.3**: System provides confidence score for classification
- **FR-2.4**: System handles varied lighting conditions

#### FR-3: Recommendations
- **FR-3.1**: System provides skincare/makeup recommendations based on detected skin tone
- **FR-3.2**: Recommendations include at least 3 product categories (e.g., foundation, sunscreen, lipstick)
- **FR-3.3**: User can view detailed product information
- **FR-3.4**: Recommendations consider skin undertone if detected

#### FR-4: User Interface
- **FR-4.1**: Clear instructions for capturing good quality images
- **FR-4.2**: Visual display of detected skin tone
- **FR-4.3**: Easy-to-understand recommendation presentation
- **FR-4.4**: Navigation between app screens

### Extended Features (Post-MVP)

#### FR-5: User Preferences
- **FR-5.1**: User can save preferred brands
- **FR-5.2**: User can filter by price range
- **FR-5.3**: User can specify product preferences (vegan, cruelty-free, etc.)

#### FR-6: History
- **FR-6.1**: User can view past analyses
- **FR-6.2**: User can compare results over time
- **FR-6.3**: User can delete history

#### FR-7: Education
- **FR-7.1**: Information about skin tone scales
- **FR-7.2**: Tips for skincare based on skin type
- **FR-7.3**: About section explaining the technology

## Non-Functional Requirements

### NFR-1: Performance
- **NFR-1.1**: Total analysis time <2 seconds on mid-range devices
- **NFR-1.2**: App launch time <3 seconds
- **NFR-1.3**: Smooth UI (60 fps on supported devices)

### NFR-2: Accuracy
- **NFR-2.1**: Skin tone classification accuracy >85% overall
- **NFR-2.2**: Performance variance <10% across all skin tone categories
- **NFR-2.3**: Face detection success rate >95% for properly captured images

### NFR-3: Privacy
- **NFR-3.1**: All processing happens on-device (preferred)
- **NFR-3.2**: No images uploaded to servers without explicit consent
- **NFR-3.3**: No persistent storage of images without user consent
- **NFR-3.4**: Clear privacy policy presented to user

### NFR-4: Usability
- **NFR-4.1**: App usable without technical knowledge
- **NFR-4.2**: Clear error messages and guidance
- **NFR-4.3**: Accessible design (WCAG AA compliance goal)
- **NFR-4.4**: Responsive design for various screen sizes

### NFR-5: Compatibility
- **NFR-5.1**: Support iOS 12+ and Android 8+
- **NFR-5.2**: Work on variety of device specifications
- **NFR-5.3**: Graceful degradation on lower-end devices

### NFR-6: Maintainability
- **NFR-6.1**: Well-documented code
- **NFR-6.2**: Modular architecture
- **NFR-6.3**: Comprehensive test coverage (unit + integration)
- **NFR-6.4**: Easy model updates

## Constraints

### Technical Constraints
- **TC-1**: Must use Flutter framework (project requirement)
- **TC-2**: App size should be reasonable (<100MB)
- **TC-3**: Must work offline (if on-device approach)
- **TC-4**: Limited to mobile platforms (iOS/Android)

### Academic Constraints
- **AC-1**: Must demonstrate image understanding concepts from SEDS536
- **AC-2**: Must be completed within semester timeline
- **AC-3**: Must be individual work
- **AC-4**: Must include thorough documentation

### Resource Constraints
- **RC-1**: Solo development (one developer)
- **RC-2**: No budget for commercial APIs or services
- **RC-3**: Limited access to diverse testing devices
- **RC-4**: Academic-only dataset licensing

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] App successfully detects faces
- [ ] App classifies skin tones with reasonable accuracy
- [ ] App provides relevant recommendations
- [ ] App runs on both iOS and Android
- [ ] Documentation is complete and clear

### Academic Success
- [ ] Demonstrates course concepts
- [ ] Addresses fairness and bias considerations
- [ ] Shows technical depth
- [ ] Well-documented process and decisions
- [ ] Meets course requirements

### Technical Success
- [ ] Meets performance requirements
- [ ] Passes user testing with diverse users
- [ ] Code quality meets standards
- [ ] Privacy requirements fulfilled
- [ ] Accuracy targets achieved

## Out of Scope

### Explicitly Not Included
- Backend server infrastructure (for MVP)
- E-commerce/purchasing functionality
- Social features (sharing, reviews)
- AR/Virtual try-on
- Skin condition diagnosis (medical)
- Multi-user accounts
- Cloud synchronization

## Dependencies

### External Dependencies
- Flutter SDK and ecosystem
- ML Kit or TensorFlow Lite
- Pre-trained face detection models
- Skin tone classification datasets
- Product recommendation data

### Internal Dependencies
- Research phase completion
- Model training/selection
- Dataset acquisition
- UI/UX design

## Assumptions

1. Users have smartphone with camera
2. Users can capture well-lit selfies
3. Product recommendation data can be curated/sourced
4. On-device ML is feasible for required accuracy
5. Academic use datasets are accessible

## Risks

### Technical Risks
- **R-1**: On-device model insufficient accuracy
  - *Mitigation*: Evaluate early, have backend fallback plan

- **R-2**: Limited diverse training data
  - *Mitigation*: Use multiple datasets, synthetic augmentation

- **R-3**: Performance issues on low-end devices
  - *Mitigation*: Model optimization, progressive feature delivery

### Project Risks
- **R-4**: Timeline constraints
  - *Mitigation*: Clear MVP scope, phased delivery

- **R-5**: Dataset licensing issues
  - *Mitigation*: Identify multiple dataset options early

### Fairness Risks
- **R-6**: Model bias toward certain skin tones
  - *Mitigation*: Balanced evaluation, fairness metrics, diverse testing

## Validation

### Acceptance Testing
- [ ] Functional requirements validated
- [ ] Performance benchmarks met
- [ ] Accuracy targets achieved
- [ ] User testing with diverse participants
- [ ] Privacy audit completed

### Academic Validation
- [ ] Meets course objectives
- [ ] Instructor feedback incorporated
- [ ] Demonstrates required concepts
- [ ] Documentation standards met
