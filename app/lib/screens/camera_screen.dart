import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:go_router/go_router.dart';
import '../services/face_detection_service.dart';
import '../services/image_processing_service.dart';
import '../widgets/face_overlay.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isLoading = true;
  String _statusMessage = 'Initializing...';
  PermissionStatus? _cameraPermissionStatus;
  int _currentCameraIndex = 0;
  XFile? _capturedImage;

  // Face detection
  final FaceDetectionService _faceDetectionService = FaceDetectionService();
  final ImageProcessingService _imageProcessingService = ImageProcessingService();
  List<Face> _detectedFaces = [];
  bool _isDetecting = false;
  ui.Rect? _capturedFaceRect;
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      // Try to get available cameras directly - this will trigger permission request on iOS
      print('Attempting to get available cameras...');
      _cameras = await availableCameras();
      print('Found ${_cameras!.length} cameras');

      // Check permission status after attempting camera access
      _cameraPermissionStatus = await Permission.camera.status;
      print('Camera permission status after access attempt: $_cameraPermissionStatus');

      if (_cameras == null || _cameras!.isEmpty) {
        setState(() {
          _statusMessage = 'No cameras found on device';
          _isLoading = false;
        });
        return;
      }

      // Initialize camera at current index (default to front camera if available)
      if (_currentCameraIndex == 0 && _cameras!.length > 1) {
        // Try to find front camera first
        final frontCameraIndex = _cameras!.indexWhere(
          (camera) => camera.lensDirection == CameraLensDirection.front,
        );
        if (frontCameraIndex != -1) {
          _currentCameraIndex = frontCameraIndex;
        }
      }

      final camera = _cameras![_currentCameraIndex];
      _controller = CameraController(
        camera,
        ResolutionPreset.medium,
      );

      await _controller!.initialize();

      // Start face detection stream
      _startFaceDetection();

      setState(() {
        _statusMessage = 'Camera initialized successfully!\n${_cameras!.length} camera(s) found\nUsing: ${camera.lensDirection.name} camera';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _statusMessage = 'Error initializing camera: $e';
        _isLoading = false;
      });
    }
  }

  void _startFaceDetection() {
    if (_controller == null || !_controller!.value.isInitialized) return;

    _controller!.startImageStream((image) async {
      if (_isDetecting) return;
      _isDetecting = true;

      final camera = _cameras![_currentCameraIndex];
      final result = await _faceDetectionService.detectFacesFromCamera(image, camera);

      if (result != null && mounted) {
        setState(() {
          _detectedFaces = result.faces;
        });
      }

      _isDetecting = false;
    });
  }

  Future<void> _stopFaceDetection() async {
    if (_controller != null && _controller!.value.isStreamingImages) {
      await _controller!.stopImageStream();
    }
  }

  Future<void> _requestCameraPermission() async {
    print('Requesting camera permission...');
    final status = await Permission.camera.request();
    print('Permission request result: $status');

    setState(() {
      _cameraPermissionStatus = status;
    });

    if (status.isGranted) {
      _initializeCamera();
    } else if (status.isPermanentlyDenied) {
      setState(() {
        _statusMessage = 'Camera permission permanently denied. Please enable in Settings.';
      });
    } else {
      setState(() {
        _statusMessage = 'Camera permission denied';
      });
    }
  }

  Future<void> _switchCamera() async {
    if (_cameras == null || _cameras!.length < 2) {
      return;
    }

    setState(() {
      _isLoading = true;
      _statusMessage = 'Switching camera...';
      _detectedFaces = [];
    });

    // Stop face detection and dispose current controller
    await _stopFaceDetection();
    await _controller?.dispose();

    // Switch to next camera
    _currentCameraIndex = (_currentCameraIndex + 1) % _cameras!.length;

    try {
      final camera = _cameras![_currentCameraIndex];
      _controller = CameraController(
        camera,
        ResolutionPreset.medium,
      );

      await _controller!.initialize();

      // Restart face detection with new camera
      _startFaceDetection();

      setState(() {
        _statusMessage = 'Switched to ${camera.lensDirection.name} camera';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _statusMessage = 'Error switching camera: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _takePicture() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      return;
    }

    if (_detectedFaces.isEmpty) {
      return;
    }

    try {
      // Store the face rect before stopping detection
      _capturedFaceRect = _detectedFaces.first.boundingBox;

      // Stop the image stream before taking picture
      await _stopFaceDetection();

      // Take the picture and get the file
      final image = await _controller!.takePicture();

      setState(() {
        _capturedImage = image;
      });
    } catch (e) {
      // Show error to user
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error capturing image: $e')),
        );
      }
    }
  }

  void _retakePicture() {
    setState(() {
      _capturedImage = null;
      _capturedFaceRect = null;
      _detectedFaces = [];
    });
    // Restart face detection
    _startFaceDetection();
  }

  Future<void> _confirmPicture() async {
    if (_capturedImage == null || _capturedFaceRect == null) return;
    if (_isProcessing) return;

    setState(() {
      _isProcessing = true;
    });

    try {
      // Get image size from camera preview
      final imageSize = ui.Size(
        _controller!.value.previewSize!.height,
        _controller!.value.previewSize!.width,
      );

      // Extract face from the captured image
      final facePath = await _imageProcessingService.extractFace(
        imagePath: _capturedImage!.path,
        faceRect: _capturedFaceRect!,
        imageSize: imageSize,
        padding: 0.3, // 30% padding around face
      );

      if (facePath != null && mounted) {
        // Navigate to results screen with the cropped face image
        context.go('/results', extra: {
          'imagePath': facePath,
        });
      } else if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Failed to extract face. Please try again.')),
        );
        _retakePicture();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error processing image: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _stopFaceDetection();
    _controller?.dispose();
    _faceDetectionService.dispose();
    super.dispose();
  }

  Widget _buildFaceStatusIndicator() {
    String message;
    Color color;
    IconData icon;

    if (_detectedFaces.isEmpty) {
      message = 'Position your face in the frame';
      color = Colors.orange;
      icon = Icons.face_retouching_off;
    } else if (_detectedFaces.length == 1) {
      message = 'Face detected';
      color = Colors.green;
      icon = Icons.face;
    } else {
      message = 'Multiple faces detected - position one person';
      color = Colors.orange;
      icon = Icons.people;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: color, size: 20),
          const SizedBox(width: 8),
          Text(
            message,
            style: TextStyle(color: color, fontWeight: FontWeight.w500),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // If we have a captured image, show preview screen
    if (_capturedImage != null) {
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
                          'Preview',
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
                // Image area - same padding as camera screen
                Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Transform.flip(
                    flipX: true, // Mirror horizontally to match camera preview
                    child: Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(16),
                        image: DecorationImage(
                          image: FileImage(File(_capturedImage!.path)),
                          fit: BoxFit.cover,
                        ),
                      ),
                    ),
                  ),
                ),
              ),
                // Button area - same height as camera capture button area
                Container(
                  padding: const EdgeInsets.symmetric(vertical: 16.0, horizontal: 16.0),
                  child: Row(
                    children: [
                      Expanded(
                        child: SizedBox(
                          height: 56,
                          child: ElevatedButton.icon(
                            onPressed: _retakePicture,
                            icon: const Icon(Icons.refresh),
                            label: const Text('Retake'),
                          ),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: SizedBox(
                          height: 56,
                          child: ElevatedButton.icon(
                            onPressed: _isProcessing ? null : _confirmPicture,
                            icon: _isProcessing
                                ? const SizedBox(
                                    width: 20,
                                    height: 20,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      color: Colors.white,
                                    ),
                                  )
                                : const Icon(Icons.check),
                            label: Text(_isProcessing ? 'Processing...' : 'Confirm'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.green,
                              foregroundColor: Colors.white,
                            ),
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
      );
    }

    // Otherwise show camera screen
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
          child: _isLoading
              ? const Center(child: CircularProgressIndicator(color: Colors.white))
              : Column(
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
                              'Skin Tone Analyzer',
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
                    // Camera preview or status message
                    Expanded(
                    child: _controller != null && _controller!.value.isInitialized
                        ? Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(16),
                              child: LayoutBuilder(
                                builder: (context, constraints) {
                                  return Stack(
                                    fit: StackFit.expand,
                                    children: [
                                      // Use FittedBox to maintain aspect ratio while filling space
                                      FittedBox(
                                        fit: BoxFit.cover,
                                        clipBehavior: Clip.hardEdge,
                                        child: SizedBox(
                                          width: _controller!.value.previewSize!.height,
                                          height: _controller!.value.previewSize!.width,
                                          child: CameraPreview(_controller!),
                                        ),
                                      ),
                                      // Face detection overlay
                                      if (_detectedFaces.isNotEmpty)
                                        FaceOverlay(
                                          faces: _detectedFaces,
                                          imageSize: Size(
                                            _controller!.value.previewSize!.height,
                                            _controller!.value.previewSize!.width,
                                          ),
                                          widgetSize: Size(
                                            constraints.maxWidth,
                                            constraints.maxHeight,
                                          ),
                                          isFrontCamera: _cameras![_currentCameraIndex].lensDirection == CameraLensDirection.front,
                                        ),
                                      // Switch camera button overlay
                                      if (_cameras != null && _cameras!.length > 1)
                                        Positioned(
                                          top: 16,
                                          right: 16,
                                          child: FloatingActionButton(
                                            mini: true,
                                            onPressed: _switchCamera,
                                            backgroundColor: Colors.white.withValues(alpha: 0.8),
                                            child: const Icon(Icons.flip_camera_ios),
                                          ),
                                        ),
                                    ],
                                  );
                                },
                              ),
                            ),
                          )
                        : Center(
                            child: Padding(
                              padding: const EdgeInsets.all(16.0),
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Text(
                                    _statusMessage,
                                    textAlign: TextAlign.center,
                                    style: const TextStyle(fontSize: 16, color: Colors.white),
                                  ),
                                  const SizedBox(height: 20),
                                  if (_cameraPermissionStatus != null &&
                                      !_cameraPermissionStatus!.isGranted)
                                    ElevatedButton(
                                      onPressed: _requestCameraPermission,
                                      child: const Text('Grant Camera Permission'),
                                    ),
                                ],
                              ),
                            ),
                          ),
                  ),
                  // Face detection status indicator
                  if (_controller != null && _controller!.value.isInitialized)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 16.0),
                      child: _buildFaceStatusIndicator(),
                    ),
                    // Capture button - same container height as preview buttons
                    if (_controller != null && _controller!.value.isInitialized)
                      Container(
                        padding: const EdgeInsets.symmetric(vertical: 16.0, horizontal: 16.0),
                        child: SizedBox(
                          height: 56,
                          child: Center(
                            child: GestureDetector(
                              onTap: _detectedFaces.isNotEmpty ? _takePicture : null,
                              child: Container(
                                width: 56,
                                height: 56,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  border: Border.all(
                                    color: _detectedFaces.isNotEmpty ? Colors.green : Colors.white54,
                                    width: 3,
                                  ),
                                ),
                                child: Padding(
                                  padding: const EdgeInsets.all(3.0),
                                  child: Container(
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      color: _detectedFaces.isNotEmpty ? Colors.green : Colors.white54,
                                    ),
                                  ),
                                ),
                              ),
                            ),
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
