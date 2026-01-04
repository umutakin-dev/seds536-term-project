import 'package:flutter/material.dart';
import '../models/skincare_recommendation.dart';
import '../data/recommendations_data.dart';

/// Bottom sheet widget displaying skincare recommendations
class RecommendationsSheet extends StatelessWidget {
  final String skinToneClass;

  const RecommendationsSheet({
    super.key,
    required this.skinToneClass,
  });

  /// Show the recommendations bottom sheet
  static void show(BuildContext context, String skinToneClass) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => RecommendationsSheet(skinToneClass: skinToneClass),
    );
  }

  @override
  Widget build(BuildContext context) {
    final recommendations = RecommendationsData.getRecommendations(skinToneClass);

    return DraggableScrollableSheet(
      initialChildSize: 0.75,
      minChildSize: 0.5,
      maxChildSize: 0.95,
      builder: (context, scrollController) {
        return Container(
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
          ),
          child: Column(
            children: [
              // Handle bar
              Container(
                margin: const EdgeInsets.only(top: 12),
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[300],
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              // Header
              Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: const Color(0xFFBD93F9).withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(
                        Icons.spa_outlined,
                        color: Color(0xFFBD93F9),
                        size: 24,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Skincare Recommendations',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          Text(
                            'For $skinToneClass skin (Monk Scale ${recommendations?.monkScaleRange ?? ""})',
                            style: TextStyle(
                              fontSize: 13,
                              color: Colors.grey[600],
                            ),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.close),
                      onPressed: () => Navigator.pop(context),
                    ),
                  ],
                ),
              ),
              const Divider(height: 1),
              // Recommendations list
              Expanded(
                child: recommendations == null
                    ? const Center(
                        child: Text('No recommendations available'),
                      )
                    : ListView.separated(
                        controller: scrollController,
                        padding: EdgeInsets.only(
                          top: 16,
                          left: 16,
                          right: 16,
                          bottom: 16 + MediaQuery.of(context).padding.bottom,
                        ),
                        itemCount: recommendations.recommendations.length,
                        separatorBuilder: (_, __) => const SizedBox(height: 12),
                        itemBuilder: (context, index) {
                          final rec = recommendations.recommendations[index];
                          return _RecommendationCard(recommendation: rec);
                        },
                      ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _RecommendationCard extends StatelessWidget {
  final SkinCareRecommendation recommendation;

  const _RecommendationCard({required this.recommendation});

  IconData _getCategoryIcon(String category) {
    switch (category.toLowerCase()) {
      case 'sun protection':
        return Icons.wb_sunny_outlined;
      case 'key ingredients':
        return Icons.science_outlined;
      case 'concerns to watch':
      case 'concerns':
        return Icons.visibility_outlined;
      case 'vitamin c':
        return Icons.local_florist_outlined;
      case 'hydration':
      case 'moisturization':
        return Icons.water_drop_outlined;
      case 'exfoliation':
        return Icons.auto_fix_high_outlined;
      case 'anti-aging':
        return Icons.access_time_outlined;
      default:
        return Icons.tips_and_updates_outlined;
    }
  }

  Color _getCategoryColor(String category) {
    switch (category.toLowerCase()) {
      case 'sun protection':
        return const Color(0xFFFFB86C); // Orange
      case 'key ingredients':
        return const Color(0xFFBD93F9); // Purple
      case 'concerns to watch':
      case 'concerns':
        return const Color(0xFFFF79C6); // Pink
      case 'vitamin c':
        return const Color(0xFFFFB86C); // Orange
      case 'hydration':
      case 'moisturization':
        return const Color(0xFF8BE9FD); // Cyan
      case 'exfoliation':
        return const Color(0xFF50FA7B); // Green
      case 'anti-aging':
        return const Color(0xFFFF5555); // Red
      default:
        return const Color(0xFF6272A4); // Grey-blue
    }
  }

  @override
  Widget build(BuildContext context) {
    final color = _getCategoryColor(recommendation.category);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: color.withValues(alpha: 0.2),
          width: 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                _getCategoryIcon(recommendation.category),
                color: color,
                size: 20,
              ),
              const SizedBox(width: 8),
              Text(
                recommendation.category.toUpperCase(),
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: color,
                  letterSpacing: 0.5,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            recommendation.title,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: Color(0xFF44475A),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            recommendation.description,
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey[700],
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }
}
