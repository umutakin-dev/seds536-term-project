# App Vision & Design Decisions

**Document Purpose**: Capture high-level product vision, design philosophy, and key architectural decisions for the skin tone detection app.

**Last Updated**: October 27, 2025

---

## Product Vision

While this is an academic project for SEDS536, the app is designed **in spirit as a commercial, customer-facing beauty/skincare application**. The goal is to create a warm, friendly, and accessible experience that helps users understand their skin tone and get personalized recommendations.

---

## Visual Style & Tone

### Primary Style: Warm & Friendly

- **NOT**: Clinical/medical/sterile aesthetic
- **NOT**: Purely academic/scientific interface
- **YES**: Beauty/personal care app feel
- **YES**: Approachable and welcoming design
- **YES**: Modern, inclusive, and respectful

### Design & Implementation Framework

**Two-Layer Approach**:

#### 1. Design Layer: Google Stitch (AI-Powered UI Generation)
- **Purpose**: Generate UI mockups and design inspiration
- **Tool**: https://stitch.withgoogle.com
- **Features**:
  - AI-powered design generation using Gemini
  - Rapid prototyping and iteration
  - Export to Figma and HTML/CSS
  - Natural language prompting
- **Our Use**: Create initial screen designs, refine layouts, establish visual direction

#### 2. Implementation Layer: shadcn_ui for Flutter
- **Purpose**: Pre-built, customizable Flutter components
- **Package**: `shadcn_ui` v0.38.5 ([pub.dev](https://pub.dev/packages/shadcn_ui))
- **Documentation**: https://flutter-shadcn-ui.mariuti.com
- **Features**:
  - 40+ customizable components (Card, Dialog, Sheet, Input, Progress, etc.)
  - Hybrid Material + shadcn support (critical for camera integration)
  - Built-in theming with 12 color schemes
  - Modern, clean aesthetic (rounded corners, soft shadows)
  - MIT licensed, actively maintained
- **Our Use**: Implement actual Flutter UI using pre-built, customizable components

**Design-to-Implementation Flow**:
```
Stitch (mockups) → Identify shadcn components → Customize with Dracula/Alucard themes → Implement in Flutter
```

**Why This Combination**:
- **Stitch**: Provides warm, friendly design direction aligned with our vision
- **shadcn_ui**: Provides professional, customizable components for rapid implementation
- **Synergy**: Design in Stitch, implement with shadcn, customize to match our aesthetic
- **Inclusive design**: Both tools support accessibility and modern design patterns
- **Flexibility**: shadcn's extreme customizability allows matching Stitch designs

### Color Themes

**Dark Mode: Dracula**
- Background: `#282A36` (dark blue-gray)
- Foreground: `#F8F8F2` (off-white)
- Current Line: `#44475A`
- Comment: `#6272A4`
- Accents:
  - Cyan: `#8BE9FD`
  - Green: `#50FA7B`
  - Orange: `#FFB86C`
  - Pink: `#FF79C6`
  - Purple: `#BD93F9`
  - Red: `#FF5555`
  - Yellow: `#F1FA8C`

**Light Mode: Alucard**
- Background: `#F8F8F2` (warm off-white)
- Sidebar/Secondary: `#E0D8E8` (light lavender-gray)
- Foreground: Dark text for contrast
- Maintains Dracula accent colors (inverted context)
- Warm, approachable aesthetic perfect for beauty/skincare app

**Design Rationale**:
- Dracula/Alucard pair provides consistent design language across modes
- High contrast ensures accessibility
- Warm tones align with "friendly beauty app" vision
- Well-established themes with Flutter/mobile support

### Scientific Transparency (Secondary)

- **Default view**: Friendly, simple recommendations
- **Advanced view**: Accessible via button/toggle
- **Reveals**: Scientific methodology, confidence scores, technical details
- **Rationale**: Don't overwhelm users, but provide transparency for those who want it

**Implementation Idea**: "How does this work?" or "Show technical details" button

---

## User Experience (UX)

### Core User Flow

1. **Welcome/Onboarding** → Explain app purpose, set preferences
2. **Capture** → Take photo with camera
3. **Processing** → Analyze on-device or cloud
4. **Results** → Show Monk scale value with visualization
5. **Recommendations** → Personalized skincare/makeup suggestions
6. **History** → View past analyses

### Comprehensive History Feature

**Requirement**: Users should have complete history of all photos and analyses

**Storage Options** (User Choice):
- **Local-only**: All data stored on device, no internet required
- **Cloud backup**: Sync to Firebase/Firestore for cross-device access
- **Hybrid**: Default local, optional cloud backup

**User Privacy Control**:
- Ask user on first launch about storage preference
- Allow changing preference in settings
- Clear data management (view, export, delete)

**History Features**:
- Grid/list view of past photos
- Date/time stamps
- Ability to re-view analysis
- Compare results over time (seasonal changes, lighting)
- Export/share functionality

### Analysis Options (User Choice)

**On-Device Analysis** (Default, Preferred):
- Pros: Privacy, no internet required, faster
- Cons: Limited model size, device performance dependent
- Implementation: TensorFlow Lite model

**Cloud Analysis** (Optional):
- Pros: More sophisticated models, better accuracy potential
- Cons: Requires internet, privacy concerns, latency
- Implementation: Firebase ML or custom backend
- **Ask user**: "Analyze on your device or use cloud for better accuracy?"

**Implementation Note**:
- Start with on-device only for MVP
- Add cloud option later if time permits
- Always respect user's privacy choice

---

## Technical Decisions

### Face Detection

**Current Phase (MVP)**:
- **Single face only**: App only works if exactly one face detected
- **Error handling**:
  - No face detected → "Please position your face in frame"
  - Multiple faces → "Please ensure only one face is visible"

**Future Enhancement**:
- **Multiple face support**: Detect all faces, ask user to select which to analyze
- **UI**: Highlight detected faces, tap to select

### Data Architecture

**Local Storage**:
- SQLite or Hive for structured data
- Local file system for images
- Offline-first architecture

**Cloud Storage (Optional)**:
- Firebase/Firestore for structured data
- Firebase Storage for images
- User authentication (Firebase Auth)
- Sync strategy: Background sync when connected

---

## Inclusivity & Accessibility

### Current Approach (MVP)

**Generic, Respectful Responses**:
- Use neutral, positive language
- Avoid value judgments about skin tones
- Focus on: "Your skin tone is X on the Monk scale"
- Recommendations: "These products work well for Monk scale X"

**Defer Detailed Inclusivity Work**:
- Fine-tune language and messaging in later phases
- Conduct user testing with diverse participants
- Iterate based on feedback

### Future Inclusivity Goals

- **Language review**: Work with diversity consultants
- **Representation**: Ensure UI shows diverse skin tones
- **Cultural sensitivity**: Consider different cultural contexts
- **Accessibility**: Screen reader support, high contrast mode, haptic feedback

---

## Edge Cases & Error Handling

### Face Detection Edge Cases

1. **No face detected**
   - Message: "Please position your face in the frame"
   - UI: Show guide overlay (oval/frame)

2. **Multiple faces detected**
   - Message: "Please ensure only one face is visible"
   - Future: Allow user to select which face

3. **Poor lighting**
   - Detect low confidence
   - Message: "Try better lighting for more accurate results"
   - Show lighting tips

4. **Face too close/far**
   - Guide user to optimal distance
   - Show visual feedback (frame turns green when optimal)

5. **Side profile / obscured face**
   - Message: "Please face the camera directly"
   - Require frontal face with key landmarks visible

### Analysis Edge Cases

1. **Low confidence prediction**
   - Show confidence score
   - Offer to re-capture
   - Suggest better conditions

2. **Unusual lighting conditions**
   - Warn user about potential inaccuracy
   - Suggest natural lighting

3. **Makeup/filters**
   - Potentially detect and warn
   - Or: Accept as-is, user knows best

---

## App Features Roadmap

### Phase 1: MVP (M1-M2)
- Single photo capture
- On-device analysis only
- Local storage only
- Single face detection
- Basic recommendations
- Simple UI

### Phase 2: Enhanced (M3)
- History feature with local storage
- Improved UI with Stitch components
- Better error handling
- Scientific details view

### Phase 3: Advanced (Future)
- Cloud storage option (Firebase)
- Cloud analysis option
- Multiple face selection
- Progress tracking over time
- Export/share functionality
- More sophisticated recommendations

---

## Key Questions to Resolve Later

### Recommendations Database
- Where do recommendations come from?
- Hardcoded JSON? Database? API?
- How detailed should recommendations be?
- Product brands? Generic categories?

### Lighting Detection ⚠️ **CRITICAL CHALLENGE**

**Instructor Feedback**: Dr. Yiğit emphasized that **lighting is the hardest part** of skin tone detection. Lighting conditions significantly affect perceived skin tone and can lead to inaccurate classifications.

**The Problem**:
- Indoor vs outdoor lighting (different color temperatures)
- Natural light vs artificial light (fluorescent, LED, incandescent)
- Flash vs no flash
- Time of day (morning light vs evening light)
- Direct sunlight vs shade
- Same person can look different under different lighting

**Research Questions**:
- Should we detect and normalize for lighting in preprocessing?
- Should we train models on diverse lighting conditions?
- Should we guide users to optimal lighting conditions?
- Should we use color constancy algorithms?

**Potential Approaches**:
1. **User Guidance**: Guide users to neutral lighting (e.g., "Stand near a window with natural light")
2. **Lighting Detection**: Detect lighting type and warn if suboptimal
3. **Color Normalization**: Use algorithms to normalize for lighting (challenging!)
4. **Multi-lighting Training**: Train model on images with varied lighting
5. **Reference-based**: Include color reference card in frame for calibration
6. **Combination**: Guide user + detect + normalize

**For MVP**:
- Focus on user guidance (simplest approach)
- Detect obviously poor lighting and warn user
- Document lighting conditions in metadata
- Plan for more sophisticated approaches in future

**For Academic Report**:
- This will be a major discussion point
- Compare different approaches in literature
- Document our chosen approach and limitations
- Discuss future improvements

### Privacy & Data
- What data do we collect?
- What do we store?
- How long do we retain?
- Can users export their data?
- GDPR/privacy compliance considerations?

### Performance
- How fast should analysis be?
- Target: < 500ms for full pipeline?
- How to optimize for older devices?

### Monetization (Academic Consideration)
- While this is academic, should we design for potential monetization?
- Freemium model? Ads? Affiliate recommendations?
- Document as potential business model for project report

---

## References

**Design Tools**:
- Google Stitch: https://stitch.withgoogle.com
- Stitch Prompt Guide: https://discuss.ai.google.dev/t/stitch-prompt-guide/83844

**Implementation Framework**:
- shadcn_ui Flutter Package: https://pub.dev/packages/shadcn_ui
- shadcn_ui Documentation: https://flutter-shadcn-ui.mariuti.com

**Design Themes**:
- Dracula Theme: https://draculatheme.com
- Alucard Theme: (Dracula's light mode counterpart)

**Research**:
- Monk Skin Tone Scale: (see research docs)

---

## Notes

- This document should evolve as we make more decisions
- Capture "why" behind decisions for academic documentation
- These decisions inform implementation and can be referenced in final report
- Balance between academic requirements and real-world product thinking
