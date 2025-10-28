# Stitch Design System - Study Notes

**Purpose**: Document key takeaways from studying Google's Stitch and how to apply it to our skin tone detection app.

**Date**: October 27, 2025

**Resources**:
- Stitch Website: https://stitch.withgoogle.com
- Stitch Prompt Guide: https://discuss.ai.google.dev/t/stitch-prompt-guide/83844

---

## Overview

### What is Stitch?

**Key Discovery**: Stitch is NOT a traditional design system like Material Design. It's an **AI-powered UI design generation tool** that creates interfaces through natural language prompts.

**Features**:
- Uses Google DeepMind's Gemini models
- Generates high-quality mobile and web app interfaces
- Free with usage limits (350 standard/200 experimental generations per month)
- Can export to Figma or HTML/CSS code
- Available to 18+ users in 212 countries (English only)

**Purpose**: Rapid prototyping and iterative UI design through conversational prompting

### Why Stitch for This Project?

- **Speed**: Generate complete UI screens in seconds
- **Iteration**: Easy to refine designs with natural language
- **AI-powered**: Leverages latest Google DeepMind models
- **Export options**: Figma for collaboration, HTML/CSS for reference
- **Inclusive by nature**: Can explicitly prompt for diversity and accessibility

---

## Stitch Methodology: 5 Core Principles

### 1. Starting Your Project

**Approach**: Choose between high-level or detailed prompts based on complexity

#### High-Level (for brainstorming/complex apps)
- Start with general idea
- Example: *"An app for marathon runners"*
- **Best for**: Exploring concepts, complex multi-screen apps

#### Detailed (for specific results)
- Describe core functionalities upfront
- Example: *"An app for marathon runners to engage with a community, find partners, get training advice, and find races near them"*
- **Best for**: Focused apps with clear requirements

#### Set the Vibe with Adjectives 🎨

**This is CRITICAL**: Adjectives influence colors, fonts, and imagery throughout the generated design.

**Examples from Guide**:
- *"A vibrant and encouraging fitness tracking app"* → energetic colors, bold fonts
- *"A minimalist and focused app for meditation"* → muted palette, simple layouts

**For Our Skin Tone App**:
> **"A warm and friendly beauty app for skin tone analysis and personalized skincare recommendations"**

**Expected Influence**:
- Warm color palette (aligned with Dracula/Alucard themes)
- Approachable, friendly typography (NOT clinical/medical)
- Inclusive imagery showcasing diverse skin tones
- Personal care aesthetic (beauty/skincare, NOT academic)

---

### 2. Refining Your App (Iterate Screen by Screen)

**Golden Rule**: Stitch works best with **1-2 changes per prompt**

#### Be Specific: Tell WHAT + HOW

- ❌ **Vague**: *"Make it better"*
- ✅ **Specific**: *"On the homepage, add a search bar to the header"*
- ✅ **Very Specific**: *"Change the primary call-to-action button on the login screen to be larger and use the brand's primary blue color"*

#### Focus on Specific Screens/Features

Don't try to change the entire app at once. Target individual screens with detailed descriptions.

**Examples from Guide**:
- *"Product detail page for a Japandi-styled tea store. Sells herbal teas, ceramics. Neutral, minimal colors, black buttons. Soft, elegant font."*
- *"Product detail page for Japanese workwear-inspired men's athletic apparel. Dark, minimal design, dark blue primary color. Minimal clothing pictures, natural fabrics, not gaudy."*

**For Our App** (when we start designing):
- Screen 1: Camera capture screen with live preview
- Screen 2: Analysis results with Monk scale visualization
- Screen 3: Product recommendations with explanations
- Screen 4: History screen with timeline view

#### Describe Desired Imagery

Guide the style/content of images explicitly.

**Example from Guide**:
*"Music player page for 'Suburban Legends.' Album art is a macro, zoomed-in photo of ocean water. Page background/imagery should reflect this."*

**For Our App**:
- *"Show diverse faces representing all 10 Monk skin tone categories"*
- *"Use soft, natural lighting in example photos (avoid harsh shadows)"*
- *"Include variety of ages, genders, and ethnicities in UI mockups"*

---

### 3. Controlling App Theme

#### Colors

Two approaches: **specific colors** OR **mood-based**

**Specific Color Prompts**:
- *"Change primary color to forest green"*
- *"Use #FF79C6 for accent buttons"*

**Mood-Based Prompts** (Better for initial design):
- *"Update theme to a warm, inviting color palette"*
- *"Use earthy, natural tones throughout"*

**For Our App**:
We already have Dracula/Alucard themes defined in app-vision.md. When prompting Stitch:
- *"Use a warm, approachable color palette with soft purples and pinks for accents"* (Dracula-inspired)
- *"Light mode should have warm off-white backgrounds with subtle lavender accents"* (Alucard)

#### Fonts & Borders

Modify typography and element styling (buttons, containers).

**Font Prompt Examples**:
- *"Use a playful sans-serif font"*
- *"Change headings to a serif font"*

**Border/Button Prompt Examples**:
- *"Make all buttons have fully rounded corners"*
- *"Give input fields a 2px solid black border"*

**Combined Theme Example from Guide**:
*"Book discovery app: serif font for text, light green brand color for accents"*

**For Our App**:
- *"Use friendly, rounded sans-serif fonts (not clinical or technical)"*
- *"Buttons should have rounded corners for warmth"*
- *"Use soft shadows instead of hard borders"*

---

### 4. Modifying Images in Your Design

#### Be Specific When Targeting Images

Use descriptive terms from app content to identify images.

**Targeting General Images**:
- *"Change background of [all] [product] images on [landing page] to light taupe"*

**Targeting Specific Image**:
- *"On 'Team' page, image of 'Dr. Carter (Lead Dentist)': update her lab coat to black"*

#### Coordinate Images with Theme Changes

If updating theme, specify whether images should match.

**Example**:
*"Update theme to light orange. Ensure all images and illustrative icons match this new color scheme."*

---

### 5. Pro Tips for Stitch ⭐

From the official Stitch Prompt Guide:

1. **Be Clear & Concise**: Avoid ambiguity
2. **Iterate & Experiment**: Refine designs with further prompts
3. **One Major Change at a Time**: Easier to see impact and adjust
4. **Use UI/UX Keywords**:
   - Navigation bar, call-to-action button, card layout
   - Hero section, sidebar, footer
   - Tab bar, modal, drawer, bottom sheet
5. **Reference Elements Specifically**:
   - *"Primary button on sign-up form"*
   - *"Image in hero section"*
   - *"Profile picture in navigation header"*
6. **Review & Refine**: If change isn't right, rephrase or be more targeted

---

## Application to Our Skin Tone Detection App

### Initial Prompt Strategy

**Starting Prompt** (High-level + Vibe):
> *"A warm and friendly beauty app for skin tone analysis and personalized skincare recommendations. The app should feel approachable and inclusive, showcasing diversity across all skin tones. Use soft, rounded design elements and a warm color palette."*

### Screen-by-Screen Design Plan

When we use Stitch (Issue #11), we'll design in this order:

#### Screen 1: Camera Capture

**Initial Prompt**:
> *"Camera capture screen for skin tone analysis. Large camera preview in center, friendly guidance text at top ('Find good lighting and center your face'), capture button at bottom with rounded corners. Warm, minimal interface."*

**Potential Refinements**:
- *"Add subtle overlay guidelines to help user frame their face"*
- *"Change capture button to larger size with soft pink accent color"*
- *"Add lighting quality indicator (good/poor) above camera preview"*

#### Screen 2: Analysis Results

**Initial Prompt**:
> *"Results screen showing detected skin tone on Monk scale (10 categories). Display user's photo in rounded frame at top, Monk scale visualization below with their tone highlighted, friendly explanation text. Warm, approachable design."*

**Potential Refinements**:
- *"Add 'How does this work?' button at bottom for scientific transparency"*
- *"Show confidence percentage with subtle progress indicator"*
- *"Add 'Retake Photo' and 'View Recommendations' buttons below results"*

#### Screen 3: Recommendations

**Initial Prompt**:
> *"Skincare recommendations screen. Product cards with images, brief descriptions, 'Why this works for you' explanations. Scrollable list, warm and friendly aesthetic."*

**Potential Refinements**:
- *"Make product cards have rounded corners and soft shadows"*
- *"Add 'Learn More' button on each product card"*
- *"Include filter options at top (price, category, brand)"*

#### Screen 4: History

**Initial Prompt**:
> *"History timeline showing past analyses. Card-based layout with date, thumbnail, and detected tone. Option to view details. Warm, organized design."*

**Potential Refinements**:
- *"Add month dividers in timeline"*
- *"Show small Monk scale indicator on each history card"*
- *"Add swipe-to-delete gesture with trash icon"*

---

## Design Checklist for Our App

### Inclusive Design ✅
- [ ] Showcase all 10 Monk scale tones in UI examples
- [ ] Diverse representation in mockup imagery (age, gender, ethnicity)
- [ ] Accessible color contrast ratios (WCAG AA minimum)
- [ ] Large, tappable buttons for mobile (44x44pt minimum)
- [ ] Support for both light and dark modes (Alucard/Dracula)

### Warm & Friendly Aesthetic 🎨
- [ ] Rounded corners on buttons and cards
- [ ] Soft shadows (not harsh borders)
- [ ] Warm color palette (Dracula purple/pink accents)
- [ ] Friendly, conversational copy throughout
- [ ] Sans-serif fonts (approachable, not clinical)

### Privacy & Transparency 🔒
- [ ] Clear "On-device vs Cloud" toggle
- [ ] "How does this work?" button for technical details
- [ ] Storage options clearly explained (local vs cloud)
- [ ] Photo/data management controls visible
- [ ] Delete history options easily accessible

### Lighting Challenge (Critical) ⚠️
- [ ] Guidance text for optimal lighting conditions
- [ ] Visual indicators for lighting quality (good/poor)
- [ ] Warning messages for poor lighting
- [ ] Tips/instructions for improvement
- [ ] Example photos showing good vs bad lighting

---

## Key Takeaways

1. **Stitch is a tool, not a design system** - It generates designs via AI prompts, it's not a component library
2. **Vibe matters** - Adjectives in initial prompt shape the entire aesthetic
3. **Iterate slowly** - 1-2 changes per prompt works best
4. **Be specific** - Use UI/UX terminology and target specific elements
5. **Our starting vibe**: *"Warm and friendly beauty app for skin tone analysis"*
6. **Design screen by screen** - Don't try to do everything at once
7. **Use examples** - Reference existing designs or styles (e.g., "Japandi-styled", "minimalist")

---

## Prompting Vocabulary for Our App

### UI/UX Terms to Use
- **Layout**: Card layout, list view, grid, hero section, bottom sheet
- **Navigation**: Tab bar, navigation bar, drawer, bottom navigation
- **Buttons**: Primary button, secondary button, call-to-action, floating action button
- **Input**: Text field, dropdown, toggle, slider, search bar
- **Feedback**: Loading state, error message, success message, empty state
- **Images**: Avatar, thumbnail, hero image, product image, icon

### Style Terms
- **Colors**: Warm palette, muted tones, vibrant accents, soft pastels
- **Spacing**: Generous padding, tight spacing, breathing room
- **Typography**: Friendly sans-serif, bold headings, readable body text
- **Shapes**: Rounded corners, soft edges, circular elements
- **Effects**: Soft shadow, subtle gradient, gentle animation

---

## Implementation Framework: shadcn_ui for Flutter

### Overview

**shadcn_ui** is a Flutter port of the popular shadcn/ui design system, providing 40+ customizable components with a modern, clean aesthetic.

**Package**: `shadcn_ui` v0.38.5
**Documentation**: https://flutter-shadcn-ui.mariuti.com
**License**: MIT (open source)

### Why shadcn_ui?

1. **Perfect complement to Stitch**: Design in Stitch → Implement with shadcn
2. **Highly customizable**: Each component designed for extreme flexibility
3. **Modern aesthetic**: Rounded corners, soft shadows, clean design (matches our vision)
4. **Hybrid Material support**: Can use shadcn alongside Material components (critical for camera)
5. **40+ components**: Exactly what we need for our app

### Component Mapping for Our Screens

#### Camera Capture Screen
**Material Components** (camera functionality):
- `Camera` widget from camera package
- `CameraPreview` for live preview

**shadcn Components** (UI overlays):
- `ShadButton` - Capture button with rounded corners
- `ShadCard` - Guidance text container at top
- `ShadBadge` - Lighting quality indicator (Good/Poor)
- `ShadToast` - Error messages (no face, multiple faces)

#### Results Screen
**shadcn Components**:
- `ShadCard` - Photo display in rounded frame
- `ShadProgress` - Confidence percentage indicator
- `ShadBadge` - Monk scale category display
- `ShadButton` - "Retake Photo", "View Recommendations", "How does this work?"
- `ShadSheet` - Bottom sheet for scientific details
- `ShadSeparator` - Visual dividers

#### Recommendations Screen
**shadcn Components**:
- `ShadCard` - Product cards with images and descriptions
- `ShadButton` - "Learn More" on each card
- `ShadSelect` - Filter options (price, category, brand)
- `ShadDialog` - Detailed product view modal
- `ShadTabs` - Tab navigation (Skincare vs Makeup)

#### History Screen
**shadcn Components**:
- `ShadCard` - History entry cards
- `ShadCalendar` - Date-based filtering
- `ShadTable` - Alternative list view
- `ShadContextMenu` - Right-click/long-press options (view, delete)
- `ShadSeparator` - Month dividers

### Theming shadcn with Dracula/Alucard

**Approach**: Create custom shadcn themes matching our Dracula/Alucard colors

**Dark Theme** (Dracula):
```dart
ShadThemeData.dark(
  colorScheme: ShadColorScheme(
    background: Color(0xFF282A36),  // Dracula background
    foreground: Color(0xFFF8F8F2),  // Dracula foreground
    primary: Color(0xFFBD93F9),     // Purple accent
    secondary: Color(0xFFFF79C6),   // Pink accent
    // ... map all Dracula colors
  ),
)
```

**Light Theme** (Alucard):
```dart
ShadThemeData.light(
  colorScheme: ShadColorScheme(
    background: Color(0xFFF8F8F2),  // Alucard background
    foreground: Color(0xFF282A36),  // Dark text
    primary: Color(0xFFBD93F9),     // Purple accent (maintained)
    secondary: Color(0xFFFF79C6),   // Pink accent (maintained)
    // ... map all Alucard colors
  ),
)
```

**Custom Radius/Borders**:
- Use rounded corners throughout (`borderRadius: BorderRadius.circular(12)`)
- Soft shadows instead of hard borders
- Match Stitch-generated designs

---

## Updated Workflow: Stitch + shadcn_ui

### Phase 1: Design with Stitch (Issue #11)

1. **Log into Stitch** (Google account required)
2. **Generate initial designs** using our starting prompt:
   > *"A warm and friendly beauty app for skin tone analysis and personalized skincare recommendations. Use soft, rounded design elements and a warm color palette with purple and pink accents."*
3. **Iterate screen by screen** (1-2 changes per prompt)
4. **Export to Figma** for reference and collaboration
5. **Take screenshots** of final designs

### Phase 2: Explore shadcn_ui (Issue #11)

1. **Install shadcn_ui** in Flutter project
2. **Explore components** using shadcn documentation
3. **Create component inventory**:
   - List all shadcn components we'll use
   - Map to specific screens
   - Note customization needs
4. **Create custom Dracula/Alucard themes**
5. **Build small demos** of key components with our themes

### Phase 3: Component Mapping (Issue #11)

For each Stitch-generated screen:
1. **Identify shadcn components** that match the design
2. **Document component properties** needed (colors, sizes, borders)
3. **Note customizations** required
4. **Create implementation plan** for each screen

### Phase 4: Documentation (Issue #11)

1. **Update this file** with:
   - Screenshots of Stitch designs
   - Component mapping details
   - Theming code snippets
   - Implementation notes
2. **Create component library doc** (if needed)

---

## Next Steps (Issue #11: Design UI/UX Mockups)

**Updated scope for Issue #11**:

### Part A: Stitch Design Generation
1. **Log into Stitch** (requires Google account)
2. **Start with initial prompt**: "A warm and friendly beauty app for skin tone analysis..."
3. **Generate 4 core screens**:
   - Camera capture
   - Results display
   - Recommendations
   - History
4. **Iterate on each screen** with specific refinements (1-2 changes per prompt)
5. **Export to Figma** for developer handoff and collaboration
6. **Take screenshots** of final designs

### Part B: shadcn_ui Exploration
1. **Add shadcn_ui to pubspec.yaml**
2. **Explore documentation** and available components
3. **Create component inventory** for our app
4. **Build Dracula/Alucard theme** for shadcn
5. **Create small component demos** to test theming

### Part C: Design-to-Implementation Mapping
1. **Map Stitch designs to shadcn components**
2. **Document component choices** for each screen
3. **Note customizations** needed (colors, borders, spacing)
4. **Create implementation plan** for next issues

### Part D: Documentation
1. **Update stitch-notes.md** with findings
2. **Add screenshots** of Stitch designs
3. **Document component mapping**
4. **Create theming guide** for shadcn

---

## References

**Design Tools**:
- **Stitch Website**: https://stitch.withgoogle.com/
- **Stitch Prompt Guide**: https://discuss.ai.google.dev/t/stitch-prompt-guide/83844

**Implementation Framework**:
- **shadcn_ui Package**: https://pub.dev/packages/shadcn_ui
- **shadcn_ui Documentation**: https://flutter-shadcn-ui.mariuti.com

**Internal Docs**:
- **App Vision Document**: docs/design/app-vision.md (theme colors, product decisions)

**Themes & Design**:
- **Dracula Theme**: https://draculatheme.com/
- **Monk Skin Tone Scale**: 10 categories (more inclusive than Fitzpatrick)

---

## Notes

- **Issue #10 Complete**: Stitch study finished, ready to move to mockup design
- **Stitch is experimental**: Results may vary, iterate until satisfied
- **Monthly limits**: 350 standard + 200 experimental generations (should be sufficient for MVP)
- **Export early**: Save successful designs to Figma before making major changes
- **Cross-reference**: Use app-vision.md for product decisions, this doc for design methodology
