import 'dart:io';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:image/image.dart' as img;
import '../models/analysis_history.dart';
import '../services/history_service.dart';
import '../widgets/recommendations_sheet.dart';

class HistoryDetailScreen extends StatefulWidget {
  final int historyId;

  const HistoryDetailScreen({
    super.key,
    required this.historyId,
  });

  @override
  State<HistoryDetailScreen> createState() => _HistoryDetailScreenState();
}

class _HistoryDetailScreenState extends State<HistoryDetailScreen> {
  final HistoryService _historyService = HistoryService();
  AnalysisHistory? _entry;
  bool _isLoading = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadEntry();
  }

  Future<void> _loadEntry() async {
    try {
      final entry = await _historyService.getHistoryById(widget.historyId);
      if (mounted) {
        setState(() {
          _entry = entry;
          _isLoading = false;
          if (entry == null) {
            _errorMessage = 'Analysis not found';
          }
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _errorMessage = 'Error loading analysis: $e';
        });
      }
    }
  }

  Future<void> _deleteEntry() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Analysis'),
        content: const Text('Are you sure you want to delete this analysis?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(
              foregroundColor: Colors.red,
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirmed == true && mounted) {
      await _historyService.deleteHistory(widget.historyId);
      if (mounted) {
        context.go('/history');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF44475A), // Dracula background
              Color(0xFF6272A4), // Dracula comment
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // Header
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back, color: Colors.white),
                      onPressed: () => context.go('/history'),
                    ),
                    const Expanded(
                      child: Text(
                        'Analysis Details',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.delete_outline, color: Colors.white),
                      onPressed: _entry != null ? _deleteEntry : null,
                    ),
                  ],
                ),
              ),
              // Content
              Expanded(
                child: _isLoading
                    ? const Center(
                        child: CircularProgressIndicator(
                          color: Color(0xFFBD93F9),
                        ),
                      )
                    : _errorMessage != null
                        ? _buildErrorState()
                        : _buildContent(),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildErrorState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(
              Icons.error_outline,
              size: 64,
              color: Colors.red,
            ),
            const SizedBox(height: 16),
            Text(
              _errorMessage!,
              style: const TextStyle(
                fontSize: 16,
                color: Colors.white,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: () => context.go('/history'),
              child: const Text('Back to History'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildContent() {
    final entry = _entry!;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Date header
          Text(
            _formatFullDate(entry.timestamp),
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.7),
              fontSize: 14,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 12),
          // Face Image
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(24),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.2),
                    blurRadius: 20,
                    offset: const Offset(0, 10),
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(24),
                child: _FaceImageWithBackground(imagePath: entry.imagePath),
              ),
            ),
          ),
          const SizedBox(height: 12),
          // Analysis Results
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.95),
              borderRadius: BorderRadius.circular(24),
            ),
            child: _buildResultsState(entry),
          ),
          const SizedBox(height: 12),
        ],
      ),
    );
  }

  Widget _buildResultsState(AnalysisHistory entry) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        // Header
        Row(
          children: [
            const Icon(
              Icons.check_circle,
              color: Color(0xFF50FA7B), // Dracula green
              size: 28,
            ),
            const SizedBox(width: 8),
            const Text(
              'Skin Tone Analysis',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const Spacer(),
            Text(
              '${entry.inferenceTimeMs}ms',
              style: const TextStyle(
                fontSize: 11,
                color: Colors.grey,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        // Main results - horizontal layout
        Row(
          children: [
            Expanded(
              child: _CompactMetric(
                label: 'Skin Tone',
                value: entry.skinToneClass,
                icon: Icons.palette_outlined,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _CompactMetric(
                label: 'Monk Scale',
                value: entry.monkScaleRange,
                icon: Icons.colorize_outlined,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _CompactMetric(
                label: 'Confidence',
                value: '${(entry.confidence * 100).toStringAsFixed(0)}%',
                icon: Icons.insights_outlined,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        // Probability bars
        const Text(
          'All Probabilities',
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: Colors.grey,
          ),
        ),
        const SizedBox(height: 8),
        ...entry.allProbabilities.entries.map((e) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: _ProbabilityBar(
              label: e.key,
              probability: e.value,
              isSelected: e.key == entry.skinToneClass,
            ),
          );
        }),
        const SizedBox(height: 8),
        // Recommendations button
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: () {
              RecommendationsSheet.show(context, entry.skinToneClass);
            },
            icon: const Icon(Icons.spa_outlined, size: 18),
            label: const Text('View Skincare Tips'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFBD93F9),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 12),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),
        ),
      ],
    );
  }

  String _formatFullDate(DateTime date) {
    final months = [
      'January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December'
    ];
    final hour = date.hour > 12 ? date.hour - 12 : (date.hour == 0 ? 12 : date.hour);
    final ampm = date.hour >= 12 ? 'PM' : 'AM';
    final minute = date.minute.toString().padLeft(2, '0');
    return '${months[date.month - 1]} ${date.day}, ${date.year} at $hour:$minute $ampm';
  }
}

class _CompactMetric extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;

  const _CompactMetric({
    required this.label,
    required this.value,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
      decoration: BoxDecoration(
        color: const Color(0xFFBD93F9).withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            color: const Color(0xFFBD93F9),
            size: 20,
          ),
          const SizedBox(height: 6),
          Text(
            label,
            style: const TextStyle(
              fontSize: 10,
              color: Colors.grey,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}

class _ProbabilityBar extends StatelessWidget {
  final String label;
  final double probability;
  final bool isSelected;

  const _ProbabilityBar({
    required this.label,
    required this.probability,
    required this.isSelected,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        SizedBox(
          width: 55,
          child: Text(
            label,
            style: TextStyle(
              fontSize: 11,
              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              color: isSelected ? const Color(0xFFBD93F9) : Colors.grey,
            ),
          ),
        ),
        const SizedBox(width: 6),
        Expanded(
          child: Stack(
            children: [
              Container(
                height: 16,
                decoration: BoxDecoration(
                  color: Colors.grey.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              FractionallySizedBox(
                widthFactor: probability,
                child: Container(
                  height: 16,
                  decoration: BoxDecoration(
                    color: isSelected
                        ? const Color(0xFFBD93F9)
                        : const Color(0xFF6272A4),
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 6),
        SizedBox(
          width: 45,
          child: Text(
            '${(probability * 100).toStringAsFixed(1)}%',
            style: TextStyle(
              fontSize: 11,
              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            ),
            textAlign: TextAlign.right,
          ),
        ),
      ],
    );
  }
}

class _FaceImageWithBackground extends StatefulWidget {
  final String imagePath;

  const _FaceImageWithBackground({required this.imagePath});

  @override
  State<_FaceImageWithBackground> createState() =>
      _FaceImageWithBackgroundState();
}

class _FaceImageWithBackgroundState extends State<_FaceImageWithBackground> {
  Color? _dominantColor;

  @override
  void initState() {
    super.initState();
    _calculateDominantColor();
  }

  Future<void> _calculateDominantColor() async {
    try {
      final file = File(widget.imagePath);
      if (!await file.exists()) {
        if (mounted) {
          setState(() {
            _dominantColor = const Color(0xFF44475A);
          });
        }
        return;
      }

      final bytes = await file.readAsBytes();
      final image = img.decodeImage(bytes);

      if (image == null) return;

      int totalR = 0, totalG = 0, totalB = 0;
      int sampleCount = 0;

      for (int y = 0; y < image.height; y += 10) {
        for (int x = 0; x < image.width; x += 10) {
          final pixel = image.getPixel(x, y);
          totalR += pixel.r.toInt();
          totalG += pixel.g.toInt();
          totalB += pixel.b.toInt();
          sampleCount++;
        }
      }

      if (sampleCount > 0 && mounted) {
        setState(() {
          _dominantColor = Color.fromRGBO(
            totalR ~/ sampleCount,
            totalG ~/ sampleCount,
            totalB ~/ sampleCount,
            1.0,
          );
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _dominantColor = const Color(0xFF44475A);
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final file = File(widget.imagePath);
    return Container(
      color: _dominantColor ?? const Color(0xFF44475A),
      child: Center(
        child: file.existsSync()
            ? Image.file(
                file,
                fit: BoxFit.contain,
                errorBuilder: (context, error, stackTrace) => _buildPlaceholder(),
              )
            : _buildPlaceholder(),
      ),
    );
  }

  Widget _buildPlaceholder() {
    return Icon(
      Icons.image_not_supported,
      size: 64,
      color: Colors.white.withValues(alpha: 0.5),
    );
  }
}
