# Project Timeline

## Project Information

- **Course**: SEDS536 - Image Understanding
- **Semester**: Fall 2024 / Spring 2025 (TBD)
- **Project Duration**: TBD (typically 10-14 weeks)
- **Deliverable Date**: TBD

## Phases Overview

```
Phase 1: Research & Planning (Weeks 1-3)
Phase 2: Design & Architecture (Weeks 4-5)
Phase 3: Development - Core (Weeks 6-9)
Phase 4: Testing & Refinement (Weeks 10-12)
Phase 5: Documentation & Presentation (Weeks 13-14)
```

## Detailed Timeline

### Phase 1: Research & Planning (Weeks 1-3)

**Week 1: Literature Review & Requirements**
- [ ] Review recent papers on skin tone detection
- [ ] Survey existing applications and approaches
- [ ] Identify fairness and bias research
- [ ] Document functional requirements
- [ ] Define success criteria

**Week 2: Technical Research**
- [ ] Evaluate skin tone classification scales (Fitzpatrick vs Monk)
- [ ] Research available datasets (UTKFace, FFHQ, etc.)
- [ ] Investigate pre-trained models
- [ ] Assess Flutter ML integration options
- [ ] Compare on-device vs backend approaches

**Week 3: Dataset & Model Selection**
- [ ] Obtain/access chosen dataset(s)
- [ ] Select classification scale (recommend Monk)
- [ ] Choose initial ML approach
- [ ] Finalize technical architecture
- [ ] Create detailed design documents

**Deliverables**:
- Research documentation complete
- Architecture decision made
- Dataset(s) identified and accessible
- Timeline and milestones defined

---

### Phase 2: Design & Architecture (Weeks 4-5)

**Week 4: System Design**
- [ ] Finalize architecture (on-device vs hybrid)
- [ ] Design data pipeline
- [ ] Define ML model architecture
- [ ] Design recommendation logic
- [ ] Create UI/UX mockups

**Week 5: Prototype & Setup**
- [ ] Set up Flutter project
- [ ] Integrate ML framework (TFLite/ML Kit)
- [ ] Create basic UI structure
- [ ] Implement camera functionality
- [ ] Build proof-of-concept pipeline

**Deliverables**:
- System architecture documented
- Flutter project initialized
- Basic prototype working
- UI mockups created

---

### Phase 3: Development - Core (Weeks 6-9)

**Week 6: Face Detection**
- [ ] Implement face detection module
- [ ] Test with various images
- [ ] Handle edge cases (no face, multiple faces)
- [ ] Optimize performance
- [ ] Add user feedback for poor images

**Week 7: Skin Tone Classification**
- [ ] Train/fine-tune classification model
- [ ] Convert model to TFLite (if needed)
- [ ] Integrate model into Flutter app
- [ ] Implement preprocessing pipeline
- [ ] Test accuracy across skin tones

**Week 8: Recommendation Engine**
- [ ] Create product database/dataset
- [ ] Implement recommendation logic
- [ ] Map skin tones to products
- [ ] Add filtering and ranking
- [ ] Test recommendation quality

**Week 9: UI/UX Implementation**
- [ ] Implement all app screens
- [ ] Add navigation and state management
- [ ] Implement results display
- [ ] Add settings and preferences
- [ ] Polish user experience

**Deliverables**:
- Core functionality complete
- All major features implemented
- Initial testing done
- App runs end-to-end

---

### Phase 4: Testing & Refinement (Weeks 10-12)

**Week 10: Testing - Functionality**
- [ ] Unit tests for core functions
- [ ] Integration tests for pipeline
- [ ] Test on both iOS and Android
- [ ] Test with diverse test images
- [ ] Fix critical bugs

**Week 11: Testing - Fairness & Performance**
- [ ] Evaluate accuracy across all skin tones
- [ ] Measure performance variance
- [ ] Test on various devices
- [ ] Optimize model if needed
- [ ] Address bias issues

**Week 12: User Testing & Refinement**
- [ ] Conduct user testing with diverse participants
- [ ] Gather feedback on UX
- [ ] Iterate on recommendations
- [ ] Fix usability issues
- [ ] Final polish and optimization

**Deliverables**:
- Comprehensive testing complete
- Fairness evaluation documented
- User testing feedback incorporated
- App stable and polished

---

### Phase 5: Documentation & Presentation (Weeks 13-14)

**Week 13: Documentation**
- [ ] Complete code documentation
- [ ] Finalize technical reports
- [ ] Document evaluation results
- [ ] Create user guide
- [ ] Prepare demonstration

**Week 14: Final Presentation**
- [ ] Create presentation slides
- [ ] Prepare demonstration video
- [ ] Rehearse presentation
- [ ] Submit final deliverables
- [ ] Course presentation

**Deliverables**:
- All documentation complete
- Final presentation ready
- Project submitted
- Public repository cleaned up

---

## Milestones

### Milestone 1: Research Complete
**Date**: End of Week 3
**Criteria**:
- Architecture decided
- Datasets identified
- Technical approach validated

### Milestone 2: Prototype Working
**Date**: End of Week 5
**Criteria**:
- Basic pipeline functional
- Camera integration working
- Proof of concept demonstrated

### Milestone 3: Core Features Complete
**Date**: End of Week 9
**Criteria**:
- Face detection working
- Classification implemented
- Recommendations functional
- UI complete

### Milestone 4: Testing Complete
**Date**: End of Week 12
**Criteria**:
- All tests passing
- Fairness evaluated
- User testing done
- Performance targets met

### Milestone 5: Project Complete
**Date**: End of Week 14
**Criteria**:
- Documentation complete
- Presentation ready
- All requirements met
- Project submitted

---

## Risk Buffer

- **Contingency Time**: 1-2 weeks built into testing phase
- **Scope Reduction**: Features can be moved to "future work" if needed
- **MVP Definition**: Core features clearly defined for minimum viable submission

## Weekly Commitments

**Recommended Time Investment**:
- Research/Planning: 10-15 hours/week
- Development: 15-20 hours/week
- Testing/Documentation: 10-15 hours/week

**Total Estimated Effort**: 150-200 hours

---

## Tracking

This timeline will be tracked in:
- GitHub Issues (individual tasks)
- GitHub Project (SEDS536-Term-Project)
- Weekly progress updates in documentation

## Updates

**Last Updated**: [Current Date]
**Status**: Planning Phase
**Next Milestone**: Research Complete (Week 3)

---

## Notes

- Timeline assumes standard semester structure
- Adjust based on actual course schedule
- Some phases may overlap
- Regular instructor check-ins recommended
- Keep documentation updated throughout
