# Pre-trained Models and Approaches

This document tracks available pre-trained models and technical approaches for skin tone detection.

## Model Requirements

- Compatible with mobile deployment (TensorFlow Lite, etc.)
- Reasonable size for on-device inference (<50MB preferred)
- Good performance across diverse skin tones
- Open source or academic-friendly licensing

## Face Detection Models

### Options to Evaluate

**Google ML Kit Face Detection**
- Platform: Flutter plugin available
- Size: Lightweight
- Performance: Fast, on-device
- Limitations: Detection only, no classification

**MediaPipe Face Detection**
- Platform: Cross-platform, has Flutter support
- Size: Efficient
- Performance: Real-time capable
- Limitations: Additional work for skin classification

**MTCNN (Multi-task Cascaded Convolutional Networks)**
- Platform: TensorFlow models available
- Size: Moderate
- Performance: Good accuracy
- Limitations: Need to convert to TFLite

## Skin Tone Classification Models

### Approaches to Investigate

1. **Transfer Learning from Existing Models**
   - Base: MobileNetV2, EfficientNet
   - Fine-tune on skin tone dataset
   - Export to TensorFlow Lite

2. **Color Space Analysis**
   - Face detection + LAB color space
   - Statistical analysis of skin pixels
   - Rules-based classification

3. **Hybrid Approach**
   - Face detection (ML Kit)
   - Skin segmentation (color filtering)
   - Tone classification (lightweight CNN)

## Model Evaluation Criteria

### Technical Metrics
- Accuracy across all skin tone categories
- Inference time on mobile device
- Model size
- Memory usage

### Fairness Metrics
- Performance parity across skin tones
- False positive/negative rates per category
- Confusion matrix analysis

## Conversion and Optimization

### TensorFlow Lite Conversion
- Quantization options (INT8, FP16)
- Model optimization techniques
- Size vs accuracy tradeoffs

### Flutter Integration
- Plugin options (tflite_flutter, ml_kit)
- Implementation patterns
- Performance considerations

## Experiments to Run

- [ ] Benchmark ML Kit face detection speed
- [ ] Test color space analysis accuracy
- [ ] Evaluate pre-trained classification models
- [ ] Compare on-device vs backend inference

## Summary

(To be populated after model evaluation)
