# System Design

High-level architecture for the skin tone detection and recommendation system.

## System Overview

```
[User] → [Flutter App] → [ML Pipeline] → [Recommendation Engine] → [User Interface]
```

## Architecture Options

### Option 1: Full On-Device Processing (Preferred)

```
Flutter App
├── Camera/Image Input
├── Face Detection (ML Kit / TFLite)
├── Skin Tone Classification (TFLite Model)
├── Recommendation Logic (Rule-based / Local DB)
└── UI Presentation
```

**Pros**:
- Complete privacy (no data leaves device)
- Works offline
- Low latency
- No server costs

**Cons**:
- Limited model complexity
- Larger app size
- Device-dependent performance
- Updates require app update

### Option 2: Hybrid Architecture

```
Flutter App (Client)
├── Camera/Image Input
├── Preprocessing
├── API Client
└── UI Presentation

Backend Server
├── ML Inference Service
├── Recommendation Engine
├── Product Database
└── Analytics (optional)
```

**Pros**:
- More powerful models
- Easy updates
- Centralized product database
- Analytics and improvement

**Cons**:
- Privacy concerns
- Requires internet
- Server costs
- Higher latency

### Option 3: Progressive Enhancement

```
Initial Release: On-device with basic model
Future Enhancement: Optional cloud processing for enhanced accuracy
```

## Component Design

### 1. Image Acquisition

**Requirements**:
- Front-facing camera access
- Good lighting guidance
- Face positioning guidelines
- Image quality validation

**Implementation**:
- Flutter camera plugin
- Real-time preview
- Capture validation
- Quality feedback to user

### 2. Face Detection

**Options**:
- Google ML Kit Face Detection
- MediaPipe Face Mesh
- Custom TFLite model

**Selected**: TBD after evaluation

**Requirements**:
- Real-time or near-real-time
- Robust to angle variations
- Minimal false positives

### 3. Skin Tone Classification

**Pipeline**:
```
Input Image
  → Face Detection
  → Skin Region Extraction
  → Color Space Conversion (RGB → LAB)
  → Feature Extraction
  → Classification (Monk Scale)
  → Confidence Score
```

**Model Architecture**: TBD
- Input: Face region (224x224 or similar)
- Output: 10 classes (Monk Scale) + confidence

### 4. Recommendation Engine

**Input**: Skin tone category + optional preferences
**Output**: Ranked list of product recommendations

**Logic**:
```
Skin Tone Category
  → Filter compatible products
  → Apply user preferences (price, brand, etc.)
  → Rank by relevance
  → Return top N recommendations
```

**Data Structure**:
```json
{
  "product_id": "string",
  "name": "string",
  "category": "foundation|lipstick|etc",
  "suitable_tones": [1, 2, 3],
  "undertone": "cool|warm|neutral",
  "price_range": "low|mid|high",
  "description": "string"
}
```

### 5. User Interface

**Screens**:
1. **Home**: Capture/upload image
2. **Analysis**: Processing feedback
3. **Results**: Skin tone + confidence
4. **Recommendations**: Product suggestions
5. **Settings**: Preferences

**UX Considerations**:
- Clear instructions
- Privacy messaging
- Confidence indicators
- Educational content (about skin tones)

## Data Flow

### Primary Flow (On-Device)
```
1. User captures/uploads image
2. App validates image quality
3. Face detection extracts face region
4. Preprocessing normalizes image
5. TFLite model classifies skin tone
6. App queries local recommendation DB
7. Results displayed to user
```

### Error Handling
- No face detected → Guide user
- Low confidence → Request better image
- Poor lighting → Suggest improvements

## Technology Stack

### Mobile App
- **Framework**: Flutter
- **Language**: Dart
- **ML**: TensorFlow Lite / ML Kit
- **State Management**: TBD (Provider, Riverpod, Bloc)
- **Local Storage**: Hive or SQLite
- **Camera**: camera plugin

### ML Pipeline (If Backend)
- **Framework**: TensorFlow/PyTorch
- **API**: FastAPI or Flask
- **Deployment**: TBD (Cloud Run, Lambda, etc.)

## Performance Requirements

### On-Device Inference
- **Target latency**: <500ms total pipeline
- **Face detection**: <100ms
- **Classification**: <300ms
- **UI update**: <100ms

### Model Size
- **Target**: <20MB
- **Maximum**: <50MB

### Accuracy
- **Target**: >90% across all skin tone categories
- **Fairness**: <5% accuracy variance between categories

## Security & Privacy

### Data Handling
- Images processed locally (preferred)
- No image storage without consent
- No data transmission (on-device approach)
- Clear privacy policy

### User Control
- Optional analytics
- Data deletion capability
- Transparent processing

## Scalability Considerations

(Relevant if backend approach chosen)
- Caching strategies
- Rate limiting
- Load balancing
- Database optimization

## Future Enhancements

- Multi-face detection
- Skin condition analysis (acne, etc.)
- Seasonal color analysis
- AR try-on features
- Social sharing
- Product purchasing integration

## Decision Log

**Date**: [Current Date]
**Initial Architecture**: To be decided after research phase
**Key Trade-off**: Privacy vs model complexity
