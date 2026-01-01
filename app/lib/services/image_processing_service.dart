import 'dart:io';
import 'dart:ui' as ui;
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

/// Service for image processing operations like face extraction
class ImageProcessingService {
  /// Extract face region from an image file
  ///
  /// [imagePath] - Path to the source image
  /// [faceRect] - Bounding box of the detected face
  /// [imageSize] - Size of the image as reported by ML Kit
  /// [isFrontCamera] - Whether the image was taken with front camera (needs mirroring)
  /// [padding] - Extra padding around the face (as percentage, e.g., 0.2 = 20%)
  ///
  /// Returns the path to the cropped face image
  Future<String?> extractFace({
    required String imagePath,
    required ui.Rect faceRect,
    required ui.Size imageSize,
    bool isFrontCamera = true,
    double padding = 0.3,
  }) async {
    try {
      // Read the source image
      final file = File(imagePath);
      final bytes = await file.readAsBytes();
      final sourceImage = img.decodeImage(bytes);

      if (sourceImage == null) {
        return null;
      }

      // Calculate scale factor between ML Kit coords and actual image
      final scaleX = sourceImage.width / imageSize.width;
      final scaleY = sourceImage.height / imageSize.height;

      // Scale the face rect to actual image coordinates
      double left = faceRect.left * scaleX;
      double top = faceRect.top * scaleY;
      double width = faceRect.width * scaleX;
      double height = faceRect.height * scaleY;

      // Mirror X coordinates for front camera (image is not mirrored but preview was)
      if (isFrontCamera) {
        left = sourceImage.width - left - width;
      }

      // Add padding
      final paddingX = width * padding;
      final paddingY = height * padding;

      left = (left - paddingX).clamp(0, sourceImage.width.toDouble());
      top = (top - paddingY).clamp(0, sourceImage.height.toDouble());
      width = (width + paddingX * 2).clamp(0, sourceImage.width - left);
      height = (height + paddingY * 2).clamp(0, sourceImage.height - top);

      // Crop the face region
      final croppedImage = img.copyCrop(
        sourceImage,
        x: left.toInt(),
        y: top.toInt(),
        width: width.toInt(),
        height: height.toInt(),
      );

      // Save the cropped image
      final tempDir = await getTemporaryDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final outputPath = '${tempDir.path}/face_$timestamp.jpg';

      final outputFile = File(outputPath);
      await outputFile.writeAsBytes(img.encodeJpg(croppedImage, quality: 95));

      return outputPath;
    } catch (e) {
      print('Error extracting face: $e');
      return null;
    }
  }
}
