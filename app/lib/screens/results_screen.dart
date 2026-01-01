import 'dart:io';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:image/image.dart' as img;

class ResultsScreen extends StatelessWidget {
  final String imagePath;

  const ResultsScreen({
    super.key,
    required this.imagePath,
  });

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
                      // Extracted Face Image - full width, centered face with dominant color background
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
                            child: _FaceImageWithBackground(imagePath: imagePath),
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      // Analysis Placeholder - takes 60% of available space
                      Expanded(
                        flex: 6,
                        child: Container(
                          padding: const EdgeInsets.all(24),
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.95),
                            borderRadius: BorderRadius.circular(24),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Row(
                                children: [
                                  Icon(
                                    Icons.science_outlined,
                                    color: Color(0xFFBD93F9),
                                    size: 32,
                                  ),
                                  SizedBox(width: 12),
                                  Text(
                                    'Skin Tone Analysis',
                                    style: TextStyle(
                                      fontSize: 20,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 24),
                              _AnalysisItem(
                                label: 'Monk Scale',
                                value: 'Coming Soon',
                                icon: Icons.palette_outlined,
                              ),
                              const SizedBox(height: 16),
                              _AnalysisItem(
                                label: 'Skin Tone',
                                value: 'Analysis Pending',
                                icon: Icons.colorize_outlined,
                              ),
                              const SizedBox(height: 16),
                              _AnalysisItem(
                                label: 'Undertone',
                                value: 'Analysis Pending',
                                icon: Icons.layers_outlined,
                              ),
                              const Spacer(),
                              Container(
                                padding: const EdgeInsets.all(16),
                                decoration: BoxDecoration(
                                  color: const Color(0xFFBD93F9).withValues(alpha: 0.1),
                                  borderRadius: BorderRadius.circular(16),
                                ),
                                child: const Row(
                                  children: [
                                    Icon(
                                      Icons.info_outline,
                                      color: Color(0xFFBD93F9),
                                    ),
                                    SizedBox(width: 12),
                                    Expanded(
                                      child: Text(
                                        'ML analysis will be implemented in future issues',
                                        style: TextStyle(
                                          fontSize: 14,
                                          color: Color(0xFFBD93F9),
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
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

/// Widget that displays face image centered with dominant color as background
class _FaceImageWithBackground extends StatefulWidget {
  final String imagePath;

  const _FaceImageWithBackground({required this.imagePath});

  @override
  State<_FaceImageWithBackground> createState() => _FaceImageWithBackgroundState();
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
