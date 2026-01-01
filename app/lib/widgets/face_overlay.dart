import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

/// Overlay widget that draws bounding boxes around detected faces
class FaceOverlay extends StatelessWidget {
  final List<Face> faces;
  final Size imageSize;
  final Size widgetSize;
  final bool isFrontCamera;

  const FaceOverlay({
    super.key,
    required this.faces,
    required this.imageSize,
    required this.widgetSize,
    this.isFrontCamera = true,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: widgetSize,
      painter: FaceOverlayPainter(
        faces: faces,
        imageSize: imageSize,
        widgetSize: widgetSize,
        isFrontCamera: isFrontCamera,
      ),
    );
  }
}

class FaceOverlayPainter extends CustomPainter {
  final List<Face> faces;
  final Size imageSize;
  final Size widgetSize;
  final bool isFrontCamera;

  FaceOverlayPainter({
    required this.faces,
    required this.imageSize,
    required this.widgetSize,
    required this.isFrontCamera,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.green;

    for (final face in faces) {
      final rect = _scaleRect(face.boundingBox);
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(8)),
        paint,
      );
    }
  }

  Rect _scaleRect(Rect rect) {
    // ML Kit returns coords in original image space (480x720)
    // We need to flip Y because ML Kit's Y origin is opposite to display
    final flippedTop = imageSize.height - rect.bottom;
    final flippedBottom = imageSize.height - rect.top;

    // Calculate scale for BoxFit.cover
    final scaleX = widgetSize.width / imageSize.width;
    final scaleY = widgetSize.height / imageSize.height;
    final scale = math.max(scaleX, scaleY);

    // Calculate offset due to cover fit centering
    final scaledWidth = imageSize.width * scale;
    final scaledHeight = imageSize.height * scale;
    final offsetX = (scaledWidth - widgetSize.width) / 2;
    final offsetY = (scaledHeight - widgetSize.height) / 2;

    // Apply scale and offset with flipped Y
    double left = rect.left * scale - offsetX;
    double top = flippedTop * scale - offsetY;
    double right = rect.right * scale - offsetX;
    double bottom = flippedBottom * scale - offsetY;

    // Mirror for front camera (preview is mirrored)
    if (isFrontCamera) {
      final tempLeft = left;
      left = widgetSize.width - right;
      right = widgetSize.width - tempLeft;
    }

    return Rect.fromLTRB(left, top, right, bottom);
  }

  @override
  bool shouldRepaint(FaceOverlayPainter oldDelegate) {
    return oldDelegate.faces != faces;
  }
}
