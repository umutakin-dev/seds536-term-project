import 'dart:convert';

/// Model class for storing analysis history entries
class AnalysisHistory {
  final int? id;
  final String imagePath;
  final String skinToneClass;
  final int classIndex;
  final String monkScaleRange;
  final double confidence;
  final Map<String, double> allProbabilities;
  final int inferenceTimeMs;
  final DateTime timestamp;

  AnalysisHistory({
    this.id,
    required this.imagePath,
    required this.skinToneClass,
    required this.classIndex,
    required this.monkScaleRange,
    required this.confidence,
    required this.allProbabilities,
    required this.inferenceTimeMs,
    required this.timestamp,
  });

  /// Create from database row
  factory AnalysisHistory.fromMap(Map<String, dynamic> map) {
    return AnalysisHistory(
      id: map['id'] as int?,
      imagePath: map['image_path'] as String,
      skinToneClass: map['skin_tone_class'] as String,
      classIndex: map['class_index'] as int,
      monkScaleRange: map['monk_scale_range'] as String,
      confidence: map['confidence'] as double,
      allProbabilities: Map<String, double>.from(
        jsonDecode(map['all_probabilities'] as String) as Map,
      ),
      inferenceTimeMs: map['inference_time_ms'] as int,
      timestamp: DateTime.fromMillisecondsSinceEpoch(map['timestamp'] as int),
    );
  }

  /// Convert to database row
  Map<String, dynamic> toMap() {
    return {
      if (id != null) 'id': id,
      'image_path': imagePath,
      'skin_tone_class': skinToneClass,
      'class_index': classIndex,
      'monk_scale_range': monkScaleRange,
      'confidence': confidence,
      'all_probabilities': jsonEncode(allProbabilities),
      'inference_time_ms': inferenceTimeMs,
      'timestamp': timestamp.millisecondsSinceEpoch,
    };
  }

  /// Create a copy with updated fields
  AnalysisHistory copyWith({
    int? id,
    String? imagePath,
    String? skinToneClass,
    int? classIndex,
    String? monkScaleRange,
    double? confidence,
    Map<String, double>? allProbabilities,
    int? inferenceTimeMs,
    DateTime? timestamp,
  }) {
    return AnalysisHistory(
      id: id ?? this.id,
      imagePath: imagePath ?? this.imagePath,
      skinToneClass: skinToneClass ?? this.skinToneClass,
      classIndex: classIndex ?? this.classIndex,
      monkScaleRange: monkScaleRange ?? this.monkScaleRange,
      confidence: confidence ?? this.confidence,
      allProbabilities: allProbabilities ?? this.allProbabilities,
      inferenceTimeMs: inferenceTimeMs ?? this.inferenceTimeMs,
      timestamp: timestamp ?? this.timestamp,
    );
  }

  @override
  String toString() {
    return 'AnalysisHistory(id: $id, skinToneClass: $skinToneClass, confidence: ${(confidence * 100).toStringAsFixed(1)}%)';
  }
}
