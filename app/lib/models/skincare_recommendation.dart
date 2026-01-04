/// Model for skincare recommendations based on skin tone
class SkinCareRecommendation {
  final String category;
  final String title;
  final String description;
  final String? icon;

  const SkinCareRecommendation({
    required this.category,
    required this.title,
    required this.description,
    this.icon,
  });
}

/// Collection of recommendations for a skin tone class
class SkinToneRecommendations {
  final String skinToneClass;
  final String monkScaleRange;
  final List<SkinCareRecommendation> recommendations;

  const SkinToneRecommendations({
    required this.skinToneClass,
    required this.monkScaleRange,
    required this.recommendations,
  });
}
