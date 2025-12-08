import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:go_router/go_router.dart';

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
    });

    // Dispose current controller
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

    try {
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
    });
  }

  void _confirmPicture() {
    if (_capturedImage == null) return;

    // Navigate to results screen with the captured image
    context.go('/results', extra: {
      'imagePath': _capturedImage!.path,
    });
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // If we have a captured image, show preview screen
    if (_capturedImage != null) {
      return Scaffold(
        appBar: AppBar(
          backgroundColor: Theme.of(context).colorScheme.inversePrimary,
          title: const Text('Preview'),
          leading: IconButton(
            icon: const Icon(Icons.arrow_back),
            onPressed: () => context.go('/'),
          ),
        ),
        body: Column(
          children: [
            Expanded(
              child: Center(
                child: Image.file(
                  File(_capturedImage!.path),
                  fit: BoxFit.contain,
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _retakePicture,
                      icon: const Icon(Icons.refresh),
                      label: const Text('Retake'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _confirmPicture,
                      icon: const Icon(Icons.check),
                      label: const Text('Confirm'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        backgroundColor: Colors.green,
                        foregroundColor: Colors.white,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      );
    }

    // Otherwise show camera screen
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Skin Tone Analyzer'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.go('/'),
        ),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                // Camera preview or status message
                Expanded(
                  child: _controller != null && _controller!.value.isInitialized
                      ? Stack(
                          fit: StackFit.expand,
                          children: [
                            CameraPreview(_controller!),
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
                                  style: const TextStyle(fontSize: 16),
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
                // Capture button
                if (_controller != null && _controller!.value.isInitialized)
                  SafeArea(
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 24.0),
                      color: Colors.black,
                      child: Center(
                        child: GestureDetector(
                          onTap: _takePicture,
                          child: Container(
                            width: 70,
                            height: 70,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(color: Colors.white, width: 4),
                            ),
                            child: Padding(
                              padding: const EdgeInsets.all(4.0),
                              child: Container(
                                decoration: const BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: Colors.white,
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
    );
  }
}
