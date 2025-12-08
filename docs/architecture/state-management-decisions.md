# State Management Architecture Decisions

**Last Updated**: November 9, 2025
**Status**: Active - Current approach defined

## Current Approach: Lean State Management

### Decision Summary

For this academic project (SEDS536 - Image Understanding), we've chosen to **keep state management minimal** and focus complexity budget on the ML/image understanding aspects.

**Chosen Stack:**
- ✅ **go_router** - Type-safe navigation with deep linking support
- ✅ **Local StatefulWidget/setState** - Screen-level state management
- ✅ **shared_preferences** (future) - Persist history and user settings
- ❌ **No global state management library** - Avoiding Provider/Riverpod/BLoC initially

### Rationale

#### Why Not Event-Driven Architecture (BLoC)?

**Event-driven patterns considered:**
1. **BLoC Pattern** (`flutter_bloc`)
   - Pure event-driven: User actions → Events → State changes
   - Similar to Redux/Event Sourcing
   - Great for complex state management and audit trails

2. **Riverpod**
   - Modern reactive state management
   - Less boilerplate than BLoC
   - Compile-safe providers

3. **Provider**
   - Lightweight ChangeNotifier pattern
   - Minimal setup

**Why we're NOT using them (for now):**

1. **Scope is Simple**: App has straightforward linear flows
   - Home → Camera → Results → History
   - No complex cross-screen state dependencies
   - Data flows linearly, not reactively across components

2. **Overkill for Complexity**:
   - BLoC adds significant boilerplate (Events, States, BLoCs)
   - Would need multiple BLoCs: CameraBloc, AnalysisBloc, HistoryBloc, etc.
   - More files to maintain for marginal benefit

3. **Academic Project Focus**:
   - Course is **SEDS536 Image Understanding**
   - Value lies in: ML model accuracy, fairness testing, Monk scale implementation
   - Not evaluated on state management architecture choices

4. **Time Constraints**:
   - M1 Milestone: November 24, 2025 (2 weeks)
   - Simpler architecture = faster iteration
   - Can refactor later if needed

5. **Single-User, On-Device App**:
   - No multi-user sync (no need for Postgres listen/notify)
   - No backend real-time events
   - No collaborative features
   - On-device ML processing

#### What We Actually Need

**State Management Requirements:**

| State | Scope | Solution |
|-------|-------|----------|
| Captured image | Camera → Results | Pass via go_router navigation params |
| Analysis result (skin tone) | Results screen | Pass via go_router navigation params |
| History list | History screen only | Load from local storage on-demand |
| User profile | Profile screen only | Load from local storage on-demand |
| Camera controller | Camera screen only | Local StatefulWidget state |
| UI state (loading, errors) | Per-screen | Local setState |

**Key Insight**: No truly global shared state that needs reactive updates across screens.

## Implementation Details

### Navigation with go_router

**Passing data between screens:**

```dart
// Example: Navigate to results screen with data
context.go('/results', extra: {
  'image': capturedImage,
  'skinTone': analysisResult,
  'monkScale': monkScaleValue,
});

// Receive data in results screen
final args = GoRouterState.of(context).extra as Map<String, dynamic>;
final image = args['image'];
final skinTone = args['skinTone'];
```

**Route definitions:**

```dart
final router = GoRouter(
  routes: [
    GoRoute(
      path: '/',
      builder: (context, state) => HomeScreen(),
    ),
    GoRoute(
      path: '/camera',
      builder: (context, state) => CameraScreen(),
    ),
    GoRoute(
      path: '/results',
      builder: (context, state) => ResultsScreen(),
    ),
    GoRoute(
      path: '/history',
      builder: (context, state) => HistoryScreen(),
    ),
    GoRoute(
      path: '/profile',
      builder: (context, state) => ProfileScreen(),
    ),
  ],
);
```

### Local State Management

**Example: Camera screen state**

```dart
class CameraScreen extends StatefulWidget {
  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  XFile? _capturedImage;
  bool _isProcessing = false;

  // All state local to this screen
  // No need for global state management
}
```

### Persistent Storage (Future)

When needed, use `shared_preferences` or `sqflite`:

```dart
// Save analysis to history
final prefs = await SharedPreferences.getInstance();
await prefs.setString('last_analysis', jsonEncode(analysisData));

// Load history
final historyJson = prefs.getString('analysis_history');
final history = jsonDecode(historyJson);
```

## Future Considerations

### When to Add State Management Library

Consider adding Provider/Riverpod/BLoC if:

1. **Prop-drilling becomes painful**
   - Passing data through 3+ levels of widgets
   - Same data needed in multiple unrelated screens

2. **Need for reactive updates**
   - Multiple screens need to react to same state changes
   - Real-time updates across app

3. **App complexity grows significantly**
   - Adding collaborative features
   - Backend integration with real-time sync
   - Complex business logic spanning multiple screens

4. **Adding backend/cloud features**
   - User authentication and sync
   - Cloud storage of analyses
   - Sharing results with others

### Migration Path

If state management becomes necessary later:

**Phase 1: Add Provider for specific features**
- Start with just History state
- Keep navigation and screen state as-is
- Minimal refactoring

**Phase 2: Consider Riverpod if complexity grows**
- More type-safe than Provider
- Better testing support
- Easier to refactor incrementally

**Phase 3: BLoC only if truly complex**
- Multiple interdependent state flows
- Need for event sourcing/audit trails
- Very complex business logic

## Alternative Approaches Evaluated

### Event-Driven with Postgres Listen/Notify

**Why not applicable:**
- Postgres listen/notify is for multi-client real-time sync
- Requires backend server with Postgres database
- Overkill for single-user mobile app
- Would add unnecessary infrastructure complexity

**When it would make sense:**
- Multi-user skincare recommendation app
- Dermatologist reviewing patient analyses in real-time
- Team collaboration on skin analysis research

### Event Bus (Pub/Sub within app)

**Why not using:**
- Packages like `event_bus` create loose coupling via events
- Good for decoupling components, but adds indirection
- Our components are already decoupled via navigation
- No cross-cutting concerns that need pub/sub

**When it would make sense:**
- Analytics events tracking
- Logging system
- Plugin architecture

## Lessons for Future Projects

### Key Takeaways

1. **Match architecture to project goals**
   - Academic project ≠ Production app
   - Focus complexity where it matters (ML, not state management)

2. **Start simple, refactor when needed**
   - Premature optimization is real
   - Easy to add complexity, hard to remove it

3. **Consider evaluation criteria**
   - For SEDS536: ML accuracy, fairness, privacy matter
   - State management architecture: nice-to-have, not critical

4. **Use appropriate tools for scale**
   - 5 screens + linear flow = simple state management
   - 50 screens + complex interactions = consider BLoC/Riverpod

## References

### State Management Options

- **BLoC**: https://bloclibrary.dev/
- **Riverpod**: https://riverpod.dev/
- **Provider**: https://pub.dev/packages/provider
- **go_router**: https://pub.dev/packages/go_router

### Related Documentation

- `docs/architecture/navigation-structure.md` - (To be created in Issue #14)
- `docs/design/app-vision.md` - Overall app design philosophy
- `.claude/project-management.md` - Project milestones and timeline

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-09 | Use go_router only, no state management library | Keep architecture lean for academic project scope |
| 2025-11-09 | Pass data via navigation params | Sufficient for linear screen flows |
| 2025-11-09 | Use local setState for screen state | No cross-screen reactive state needed |
| TBD | Add shared_preferences for persistence | When implementing History feature |

---

**Note**: This is a living document. Update as architectural needs evolve during development.
