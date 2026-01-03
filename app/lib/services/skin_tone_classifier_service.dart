import 'dart:io';
import 'dart:math';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Result of skin tone classification
class ClassificationResult {
  final int classIndex;
  final String className;
  final String monkScaleRange;
  final double confidence;
  final Map<String, double> allProbabilities;
  final int inferenceTimeMs;

  ClassificationResult({
    required this.classIndex,
    required this.className,
    required this.monkScaleRange,
    required this.confidence,
    required this.allProbabilities,
    required this.inferenceTimeMs,
  });

  @override
  String toString() {
    return 'ClassificationResult($className: ${(confidence * 100).toStringAsFixed(1)}%)';
  }
}

/// Service for skin tone classification using TFLite model
class SkinToneClassifierService {
  static const String _modelPath = 'assets/models/skin_tone_classifier.tflite';

  // Class names and Monk scale ranges
  static const List<String> _classNames = ['Light', 'Medium', 'Dark'];
  static const List<String> _monkScales = ['1-3', '4-7', '8-10'];

  // ImageNet normalization constants
  static const List<double> _mean = [0.485, 0.456, 0.406];
  static const List<double> _std = [0.229, 0.224, 0.225];

  // Model input size
  static const int _inputSize = 224;

  Interpreter? _interpreter;
  bool _isInitialized = false;

  /// Whether the model is loaded and ready
  bool get isInitialized => _isInitialized;

  /// Load the TFLite model from assets
  Future<void> loadModel() async {
    if (_isInitialized) return;

    try {
      _interpreter = await Interpreter.fromAsset(_modelPath);
      _isInitialized = true;
      print('SkinToneClassifier: Model loaded successfully');
      print('  Input shape: ${_interpreter!.getInputTensor(0).shape}');
      print('  Output shape: ${_interpreter!.getOutputTensor(0).shape}');
    } catch (e) {
      print('SkinToneClassifier: Failed to load model: $e');
      rethrow;
    }
  }

  /// Classify skin tone from an image file path
  ///
  /// Returns null if classification fails
  Future<ClassificationResult?> classifyImage(String imagePath) async {
    if (!_isInitialized || _interpreter == null) {
      print('SkinToneClassifier: Model not initialized');
      return null;
    }

    final stopwatch = Stopwatch()..start();

    try {
      // Load and preprocess image
      final input = await _preprocessImage(imagePath);
      if (input == null) {
        print('SkinToneClassifier: Failed to preprocess image');
        return null;
      }

      // Prepare output tensor [1, 3]
      final output = List.filled(1, List.filled(3, 0.0));

      // Run inference
      _interpreter!.run(input, output);

      stopwatch.stop();
      final inferenceTime = stopwatch.elapsedMilliseconds;

      // Get logits and apply softmax
      final logits = output[0];
      final probabilities = _softmax(logits);

      // Find predicted class
      int maxIndex = 0;
      double maxProb = probabilities[0];
      for (int i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIndex = i;
        }
      }

      // Build probability map
      final allProbs = <String, double>{};
      for (int i = 0; i < _classNames.length; i++) {
        allProbs[_classNames[i]] = probabilities[i];
      }

      return ClassificationResult(
        classIndex: maxIndex,
        className: _classNames[maxIndex],
        monkScaleRange: _monkScales[maxIndex],
        confidence: maxProb,
        allProbabilities: allProbs,
        inferenceTimeMs: inferenceTime,
      );
    } catch (e) {
      print('SkinToneClassifier: Classification failed: $e');
      return null;
    }
  }

  /// Preprocess image for model input
  ///
  /// - Load image from file
  /// - Resize to 224x224
  /// - Normalize with ImageNet mean/std
  /// - Convert to NCHW format [1, 3, 224, 224]
  Future<List<List<List<List<double>>>>?> _preprocessImage(
      String imagePath) async {
    try {
      final file = File(imagePath);
      final bytes = await file.readAsBytes();
      var image = img.decodeImage(bytes);

      if (image == null) {
        return null;
      }

      // Resize to 224x224 (center crop style like training)
      // First resize so smallest dimension is 256, then center crop
      final shortSide = min(image.width, image.height);
      final scale = 256 / shortSide;
      final newWidth = (image.width * scale).round();
      final newHeight = (image.height * scale).round();

      image = img.copyResize(image, width: newWidth, height: newHeight);

      // Center crop to 224x224
      final cropX = (image.width - _inputSize) ~/ 2;
      final cropY = (image.height - _inputSize) ~/ 2;
      image = img.copyCrop(
        image,
        x: cropX,
        y: cropY,
        width: _inputSize,
        height: _inputSize,
      );

      // Create input tensor in NCHW format [1, 3, 224, 224]
      final input = List.generate(
        1,
        (_) => List.generate(
          3,
          (_) => List.generate(
            _inputSize,
            (_) => List.filled(_inputSize, 0.0),
          ),
        ),
      );

      // Fill tensor with normalized pixel values
      for (int y = 0; y < _inputSize; y++) {
        for (int x = 0; x < _inputSize; x++) {
          final pixel = image.getPixel(x, y);

          // Get RGB values (0-255) and normalize
          final r = pixel.r.toDouble() / 255.0;
          final g = pixel.g.toDouble() / 255.0;
          final b = pixel.b.toDouble() / 255.0;

          // Apply ImageNet normalization: (value - mean) / std
          input[0][0][y][x] = (r - _mean[0]) / _std[0]; // R channel
          input[0][1][y][x] = (g - _mean[1]) / _std[1]; // G channel
          input[0][2][y][x] = (b - _mean[2]) / _std[2]; // B channel
        }
      }

      return input;
    } catch (e) {
      print('SkinToneClassifier: Image preprocessing failed: $e');
      return null;
    }
  }

  /// Apply softmax to convert logits to probabilities
  List<double> _softmax(List<double> logits) {
    // Find max for numerical stability
    final maxLogit = logits.reduce(max);

    // Compute exp(logit - max)
    final expValues = logits.map((l) => exp(l - maxLogit)).toList();

    // Sum of all exp values
    final sumExp = expValues.reduce((a, b) => a + b);

    // Normalize to get probabilities
    return expValues.map((e) => e / sumExp).toList();
  }

  /// Clean up resources
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isInitialized = false;
  }
}
