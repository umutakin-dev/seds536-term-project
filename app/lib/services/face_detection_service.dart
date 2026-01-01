import 'dart:io';
import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

/// Result of face detection operation
class FaceDetectionResult {
  final List<Face> faces;
  final int imageWidth;
  final int imageHeight;
  final String? errorMessage;

  FaceDetectionResult({
    required this.faces,
    required this.imageWidth,
    required this.imageHeight,
    this.errorMessage,
  });

  bool get hasFaces => faces.isNotEmpty;
  bool get hasError => errorMessage != null;
  bool get hasMultipleFaces => faces.length > 1;
  Face? get primaryFace => faces.isNotEmpty ? faces.first : null;
}

/// Service for detecting faces in images using Google ML Kit
class FaceDetectionService {
  FaceDetector? _faceDetector;
  bool _isProcessing = false;

  /// Whether the service is currently processing a frame
  bool get isProcessing => _isProcessing;

  FaceDetector get _detector {
    _faceDetector ??= FaceDetector(
      options: FaceDetectorOptions(
        enableContours: false,
        enableLandmarks: false,
        enableClassification: false,
        enableTracking: true,
        minFaceSize: 0.15,
        performanceMode: FaceDetectorMode.fast,
      ),
    );
    return _faceDetector!;
  }

  /// Detect faces from a live camera frame
  Future<FaceDetectionResult?> detectFacesFromCamera(
    CameraImage image,
    CameraDescription camera,
  ) async {
    if (_isProcessing) return null;
    _isProcessing = true;

    try {
      final inputImage = _convertCameraImage(image, camera);
      if (inputImage == null) {
        _isProcessing = false;
        return null;
      }

      final faces = await _detector.processImage(inputImage);

      // Sort faces by size (largest first)
      faces.sort((a, b) {
        final areaA = a.boundingBox.width * a.boundingBox.height;
        final areaB = b.boundingBox.width * b.boundingBox.height;
        return areaB.compareTo(areaA);
      });

      _isProcessing = false;
      return FaceDetectionResult(
        faces: faces,
        imageWidth: image.width,
        imageHeight: image.height,
      );
    } catch (e) {
      _isProcessing = false;
      return FaceDetectionResult(
        faces: [],
        imageWidth: image.width,
        imageHeight: image.height,
        errorMessage: 'Live detection failed: $e',
      );
    }
  }

  /// Convert CameraImage to InputImage for ML Kit processing
  InputImage? _convertCameraImage(CameraImage image, CameraDescription camera) {
    // Get rotation
    final sensorOrientation = camera.sensorOrientation;
    InputImageRotation? rotation;

    if (Platform.isAndroid) {
      // For front camera, rotation needs adjustment
      if (camera.lensDirection == CameraLensDirection.front) {
        rotation = InputImageRotationValue.fromRawValue(
          (sensorOrientation + 360) % 360,
        );
      } else {
        rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
      }
    } else if (Platform.isIOS) {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    }

    if (rotation == null) return null;

    // On Android, use nv21 format conversion for YUV_420_888
    if (Platform.isAndroid) {
      final nv21Bytes = _convertYUV420ToNV21(image);
      return InputImage.fromBytes(
        bytes: nv21Bytes,
        metadata: InputImageMetadata(
          size: ui.Size(image.width.toDouble(), image.height.toDouble()),
          rotation: rotation,
          format: InputImageFormat.nv21,
          bytesPerRow: image.width,
        ),
      );
    }

    // iOS handling
    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null) return null;

    final planes = image.planes;
    final bytes = Uint8List.fromList(
      planes.fold<List<int>>(
        <int>[],
        (List<int> previousValue, plane) => previousValue..addAll(plane.bytes),
      ),
    );

    return InputImage.fromBytes(
      bytes: bytes,
      metadata: InputImageMetadata(
        size: ui.Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: planes.first.bytesPerRow,
      ),
    );
  }

  /// Convert YUV_420_888 to NV21 format for Android
  Uint8List _convertYUV420ToNV21(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final nv21 = Uint8List(width * height * 3 ~/ 2);

    // Copy Y plane
    int yIndex = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        nv21[yIndex++] = yPlane.bytes[y * yPlane.bytesPerRow + x];
      }
    }

    // Interleave V and U planes (NV21 format is VUVU...)
    int uvIndex = width * height;
    final uvWidth = width ~/ 2;
    final uvHeight = height ~/ 2;

    for (int y = 0; y < uvHeight; y++) {
      for (int x = 0; x < uvWidth; x++) {
        final vIndex = y * vPlane.bytesPerRow + x * vPlane.bytesPerPixel!;
        final uIndex = y * uPlane.bytesPerRow + x * uPlane.bytesPerPixel!;
        nv21[uvIndex++] = vPlane.bytes[vIndex];
        nv21[uvIndex++] = uPlane.bytes[uIndex];
      }
    }

    return nv21;
  }

  /// Detect faces in an image file
  Future<FaceDetectionResult> detectFaces(String imagePath) async {
    try {
      final inputImage = InputImage.fromFilePath(imagePath);

      // Get image dimensions
      final file = File(imagePath);
      final bytes = await file.readAsBytes();
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      final decodedImage = frame.image;

      final faces = await _detector.processImage(inputImage);

      // Sort faces by size (largest first) - primary face is likely the largest
      faces.sort((a, b) {
        final areaA = a.boundingBox.width * a.boundingBox.height;
        final areaB = b.boundingBox.width * b.boundingBox.height;
        return areaB.compareTo(areaA);
      });

      return FaceDetectionResult(
        faces: faces,
        imageWidth: decodedImage.width,
        imageHeight: decodedImage.height,
      );
    } catch (e) {
      return FaceDetectionResult(
        faces: [],
        imageWidth: 0,
        imageHeight: 0,
        errorMessage: 'Face detection failed: $e',
      );
    }
  }

  /// Clean up resources
  void dispose() {
    _faceDetector?.close();
    _faceDetector = null;
  }
}
