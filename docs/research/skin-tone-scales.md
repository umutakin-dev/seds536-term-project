# Skin Tone Classification Scales

This document compares different skin tone classification systems and recommends one for this project.

## Classification Systems

### Fitzpatrick Scale

**Overview**:
- Developed in 1975 by dermatologist Thomas Fitzpatrick
- Medical standard for classifying skin types
- 6 categories based on reaction to sun exposure

**Categories**:
1. Type I: Very fair, always burns, never tans
2. Type II: Fair, usually burns, tans minimally
3. Type III: Medium, sometimes burns, tans uniformly
4. Type IV: Olive, rarely burns, tans easily
5. Type V: Brown, very rarely burns, tans very easily
6. Type VI: Dark brown/black, never burns, tans very easily

**Pros**:
- Medical standard, widely recognized
- Extensive research and documentation
- Correlation with skin health conditions

**Cons**:
- Only 6 categories (limited granularity)
- Based on sun reaction, not just appearance
- Known bias toward lighter skin tones
- May not capture full diversity

### Monk Skin Tone Scale

**Overview**:
- Developed by Dr. Ellis Monk at Harvard (released by Google 2022)
- 10 categories for more inclusive representation
- Designed to address bias in AI/ML systems

**Categories**:
- 10 levels from lightest to darkest
- Based on visual appearance, not sun reaction
- More granular representation of darker tones

**Pros**:
- More inclusive and representative
- Better granularity (10 vs 6 categories)
- Designed specifically for computer vision
- Addresses known biases in skin classification
- Gaining adoption in tech industry

**Cons**:
- Relatively new (less research history)
- May have limited training data availability
- Less familiar to general public

### Custom/Simplified Scales

**Options**:
- 3-category: Light, Medium, Dark
- Continuous scale: 0-100 value
- Regional variations: Asian skin tone scales, etc.

**Considerations**:
- Simpler for MVP/prototype
- May lack nuance
- Risk of oversimplification

## Recommendation for This Project

### Initial Choice: Monk Skin Tone Scale (10 categories)

**Rationale**:
1. **Better representation**: More inclusive across diverse skin tones
2. **AI/ML focus**: Designed for computer vision applications
3. **Fairness**: Addresses known biases in classification
4. **Academic value**: Shows awareness of bias issues
5. **Future-proof**: Industry trend toward more inclusive scales

### Implementation Strategy

**Phase 1 (MVP)**:
- Implement simplified 3-5 category classification
- Validate approach with limited categories
- Focus on getting the pipeline working

**Phase 2 (Full Implementation)**:
- Expand to full 10-category Monk scale
- Fine-tune model for all categories
- Comprehensive evaluation across all tones

**Phase 3 (Optional Enhancement)**:
- Provide both Monk and Fitzpatrick classifications
- Allow user to choose preference
- Compare accuracy across both scales

## Mapping to Recommendations

How skin tone categories map to skincare/makeup recommendations:

### Considerations
- Different undertones (cool, warm, neutral)
- Specific concerns per skin type
- Product suitability ranges

### Recommendation Database Structure
```
Skin Tone → Product Categories → Specific Products
    ├─ Foundation shade ranges
    ├─ Sunscreen recommendations
    ├─ Color cosmetics (blush, lipstick, etc.)
    └─ Skincare actives and formulations
```

## Evaluation Metrics

For chosen scale:
- Accuracy per category
- Inter-category confusion analysis
- Edge case handling (mixed features, lighting)
- User satisfaction with recommendations

## References

- Monk Skin Tone Scale: [Add link]
- Fitzpatrick Scale documentation: [Add link]
- Academic papers comparing scales: [Add links]

## Decision Log

**Date**: [Current Date]
**Decision**: Start with Monk Skin Tone Scale
**Reviewed by**: [Your name]
**Rationale**: See above
