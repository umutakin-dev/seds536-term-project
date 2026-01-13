import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import '../models/analysis_history.dart';
import '../services/skin_tone_classifier_service.dart';

/// Service for managing analysis history storage
class HistoryService {
  static const String _databaseName = 'analysis_history.db';
  static const String _tableName = 'analysis_history';
  static const int _databaseVersion = 1;

  static Database? _database;

  /// Get database instance (singleton pattern)
  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  /// Initialize the database
  Future<Database> _initDatabase() async {
    final documentsDirectory = await getApplicationDocumentsDirectory();
    final path = join(documentsDirectory.path, _databaseName);

    return await openDatabase(
      path,
      version: _databaseVersion,
      onCreate: _onCreate,
    );
  }

  /// Create database tables
  Future<void> _onCreate(Database db, int version) async {
    await db.execute('''
      CREATE TABLE $_tableName (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT NOT NULL,
        skin_tone_class TEXT NOT NULL,
        class_index INTEGER NOT NULL,
        monk_scale_range TEXT NOT NULL,
        confidence REAL NOT NULL,
        all_probabilities TEXT NOT NULL,
        inference_time_ms INTEGER NOT NULL,
        timestamp INTEGER NOT NULL
      )
    ''');

    debugPrint('HistoryService: Database created');
  }

  /// Save a new analysis to history
  ///
  /// This will copy the image to a permanent location in app documents
  Future<AnalysisHistory> saveAnalysis({
    required String tempImagePath,
    required ClassificationResult result,
  }) async {
    final db = await database;

    // Copy image to permanent location
    final permanentImagePath = await _saveImagePermanently(tempImagePath);

    final entry = AnalysisHistory(
      imagePath: permanentImagePath,
      skinToneClass: result.className,
      classIndex: result.classIndex,
      monkScaleRange: result.monkScaleRange,
      confidence: result.confidence,
      allProbabilities: result.allProbabilities,
      inferenceTimeMs: result.inferenceTimeMs,
      timestamp: DateTime.now(),
    );

    final id = await db.insert(_tableName, entry.toMap());
    debugPrint('HistoryService: Saved analysis with id=$id');

    return entry.copyWith(id: id);
  }

  /// Copy image from temp location to permanent app documents directory
  Future<String> _saveImagePermanently(String tempPath) async {
    final documentsDirectory = await getApplicationDocumentsDirectory();
    final historyImagesDir = Directory(join(documentsDirectory.path, 'history_images'));

    // Create directory if it doesn't exist
    if (!await historyImagesDir.exists()) {
      await historyImagesDir.create(recursive: true);
    }

    // Generate unique filename
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final extension = tempPath.split('.').last;
    final newFilename = 'analysis_$timestamp.$extension';
    final newPath = join(historyImagesDir.path, newFilename);

    // Copy file
    final tempFile = File(tempPath);
    await tempFile.copy(newPath);

    debugPrint('HistoryService: Image saved to $newPath');
    return newPath;
  }

  /// Get all history entries, ordered by newest first
  Future<List<AnalysisHistory>> getAllHistory() async {
    final db = await database;
    final maps = await db.query(
      _tableName,
      orderBy: 'timestamp DESC',
    );

    return maps.map((map) => AnalysisHistory.fromMap(map)).toList();
  }

  /// Get a single history entry by ID
  Future<AnalysisHistory?> getHistoryById(int id) async {
    final db = await database;
    final maps = await db.query(
      _tableName,
      where: 'id = ?',
      whereArgs: [id],
      limit: 1,
    );

    if (maps.isEmpty) return null;
    return AnalysisHistory.fromMap(maps.first);
  }

  /// Delete a history entry by ID
  ///
  /// Also deletes the associated image file
  Future<bool> deleteHistory(int id) async {
    final db = await database;

    // Get the entry to find image path
    final entry = await getHistoryById(id);
    if (entry == null) return false;

    // Delete from database
    final rowsDeleted = await db.delete(
      _tableName,
      where: 'id = ?',
      whereArgs: [id],
    );

    if (rowsDeleted > 0) {
      // Delete associated image file
      try {
        final imageFile = File(entry.imagePath);
        if (await imageFile.exists()) {
          await imageFile.delete();
          debugPrint('HistoryService: Deleted image ${entry.imagePath}');
        }
      } catch (e) {
        debugPrint('HistoryService: Failed to delete image: $e');
      }
    }

    debugPrint('HistoryService: Deleted history with id=$id');
    return rowsDeleted > 0;
  }

  /// Delete all history entries
  Future<void> deleteAllHistory() async {
    final db = await database;

    // Get all entries to delete images
    final entries = await getAllHistory();

    // Delete from database
    await db.delete(_tableName);

    // Delete all image files
    for (final entry in entries) {
      try {
        final imageFile = File(entry.imagePath);
        if (await imageFile.exists()) {
          await imageFile.delete();
        }
      } catch (e) {
        debugPrint('HistoryService: Failed to delete image: $e');
      }
    }

    debugPrint('HistoryService: Deleted all history');
  }

  /// Get the count of history entries
  Future<int> getHistoryCount() async {
    final db = await database;
    final result = await db.rawQuery('SELECT COUNT(*) as count FROM $_tableName');
    return Sqflite.firstIntValue(result) ?? 0;
  }

  /// Close the database
  Future<void> close() async {
    final db = await database;
    await db.close();
    _database = null;
  }
}
