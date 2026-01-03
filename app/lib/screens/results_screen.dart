import 'dart:io';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:image/image.dart' as img;
import '../services/skin_tone_classifier_service.dart';

class ResultsScreen extends StatefulWidget {
  final String imagePath;

  const ResultsScreen({
    super.key,
    required this.imagePath,
  });

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  final SkinToneClassifierService _classifier = SkinToneClassifierService();
  ClassificationResult? _result;
  bool _isLoading = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _runClassification();
  }

  Future<void> _runClassification() async {
    try {
      // Load model
      await _classifier.loadModel();

      // Run classification
      final result = await _classifier.classifyImage(widget.imagePath);

      if (mounted) {
        setState(() {
          _result = result;
          _isLoading = false;
          if (result == null) {
            _errorMessage = 'Classification failed. Please try again.';
          }
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _errorMessage = 'Error: $e';
        });
      }
    }
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
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
                      onPressed: () => context.go('/'),
                    ),
                    const Expanded(
                      child: Text(
                        'Analysis Results',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                    const SizedBox(width: 48), // Balance the back button
                  ],
                ),
              ),
              // Content
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      // Extracted Face Image
                      Expanded(
                        flex: 4,
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
                            child:
                                _FaceImageWithBackground(imagePath: widget.imagePath),
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      // Analysis Results
                      Expanded(
                        flex: 6,
                        child: Container(
                          padding: const EdgeInsets.all(20),
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.95),
                            borderRadius: BorderRadius.circular(24),
                          ),
                          child: _isLoading
                              ? _buildLoadingState()
                              : _errorMessage != null
                                  ? _buildErrorState()
                                  : SingleChildScrollView(
                                      child: _buildResultsState(),
                                    ),
                        ),
                      ),
                      const SizedBox(height: 16),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLoadingState() {
    return const Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        CircularProgressIndicator(
          color: Color(0xFFBD93F9),
        ),
        SizedBox(height: 24),
        Text(
          'Analyzing skin tone...',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w500,
          ),
        ),
        SizedBox(height: 8),
        Text(
          'This may take a moment',
          style: TextStyle(
            fontSize: 14,
            color: Colors.grey,
          ),
        ),
      ],
    );
  }

  Widget _buildErrorState() {
    return Column(
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
            color: Colors.red,
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 24),
        ElevatedButton(
          onPressed: () {
            setState(() {
              _isLoading = true;
              _errorMessage = null;
            });
            _runClassification();
          },
          child: const Text('Retry'),
        ),
      ],
    );
  }

  Widget _buildResultsState() {
    final result = _result!;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        Row(
          children: [
            const Icon(
              Icons.check_circle,
              color: Color(0xFF50FA7B), // Dracula green
              size: 32,
            ),
            const SizedBox(width: 12),
            const Text(
              'Skin Tone Analysis',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const Spacer(),
            Text(
              '${result.inferenceTimeMs}ms',
              style: const TextStyle(
                fontSize: 12,
                color: Colors.grey,
              ),
            ),
          ],
        ),
        const SizedBox(height: 24),
        // Main result
        _AnalysisItem(
          label: 'Skin Tone',
          value: result.className,
          icon: Icons.palette_outlined,
        ),
        const SizedBox(height: 16),
        _AnalysisItem(
          label: 'Monk Scale',
          value: result.monkScaleRange,
          icon: Icons.colorize_outlined,
        ),
        const SizedBox(height: 16),
        _AnalysisItem(
          label: 'Confidence',
          value: '${(result.confidence * 100).toStringAsFixed(1)}%',
          icon: Icons.insights_outlined,
        ),
        const SizedBox(height: 24),
        // Probability bars
        const Text(
          'All Probabilities',
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: Colors.grey,
          ),
        ),
        const SizedBox(height: 12),
        ...result.allProbabilities.entries.map((entry) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: _ProbabilityBar(
              label: entry.key,
              probability: entry.value,
              isSelected: entry.key == result.className,
            ),
          );
        }),
        const SizedBox(height: 16),
        // Info box
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFF50FA7B).withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Row(
            children: [
              const Icon(
                Icons.lightbulb_outline,
                color: Color(0xFF50FA7B),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Monk Scale ${result.monkScaleRange}: ${_getScaleDescription(result.className)}',
                  style: const TextStyle(
                    fontSize: 14,
                    color: Color(0xFF44475A),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  String _getScaleDescription(String className) {
    switch (className) {
      case 'Light':
        return 'Fair to light skin tones';
      case 'Medium':
        return 'Medium to olive skin tones';
      case 'Dark':
        return 'Deep to rich skin tones';
      default:
        return '';
    }
  }
}

class _AnalysisItem extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;

  const _AnalysisItem({
    required this.label,
    required this.value,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: const Color(0xFFBD93F9).withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(
            icon,
            color: const Color(0xFFBD93F9),
            size: 24,
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: const TextStyle(
                  fontSize: 14,
                  color: Colors.grey,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                value,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ],
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
          width: 60,
          child: Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              color: isSelected ? const Color(0xFFBD93F9) : Colors.grey,
            ),
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Stack(
            children: [
              Container(
                height: 20,
                decoration: BoxDecoration(
                  color: Colors.grey.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              FractionallySizedBox(
                widthFactor: probability,
                child: Container(
                  height: 20,
                  decoration: BoxDecoration(
                    color: isSelected
                        ? const Color(0xFFBD93F9)
                        : const Color(0xFF6272A4),
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 8),
        SizedBox(
          width: 50,
          child: Text(
            '${(probability * 100).toStringAsFixed(1)}%',
            style: TextStyle(
              fontSize: 12,
              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            ),
            textAlign: TextAlign.right,
          ),
        ),
      ],
    );
  }
}

/// Widget that displays face image centered with dominant color as background
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
      final bytes = await file.readAsBytes();
      final image = img.decodeImage(bytes);

      if (image == null) return;

      // Sample pixels from the image to find median color
      int totalR = 0, totalG = 0, totalB = 0;
      int sampleCount = 0;

      // Sample every 10th pixel for performance
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
      // Fallback to dark color if calculation fails
      if (mounted) {
        setState(() {
          _dominantColor = const Color(0xFF44475A);
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: _dominantColor ?? const Color(0xFF44475A),
      child: Center(
        child: Image.file(
          File(widget.imagePath),
          fit: BoxFit.contain,
        ),
      ),
    );
  }
}
