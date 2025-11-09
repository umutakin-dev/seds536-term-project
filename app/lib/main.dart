import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Camera Test',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
      ),
      home: const CameraTestScreen(),
    );
  }
}

class CameraTestScreen extends StatefulWidget {
  const CameraTestScreen({super.key});

  @override
  State<CameraTestScreen> createState() => _CameraTestScreenState();
}

class _CameraTestScreenState extends State<CameraTestScreen> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isLoading = true;
  String _statusMessage = 'Initializing...';
  PermissionStatus? _cameraPermissionStatus;
  int _currentCameraIndex = 0;

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

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Camera Test - Issue #12'),
      ),
      body: Center(
        child: _isLoading
            ? const CircularProgressIndicator()
            : SingleChildScrollView(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                  // Camera preview or status message
                  if (_controller != null && _controller!.value.isInitialized)
                    Stack(
                      children: [
                        Container(
                          width: 300,
                          height: 400,
                          decoration: BoxDecoration(
                            border: Border.all(color: Colors.teal, width: 2),
                          ),
                          child: CameraPreview(_controller!),
                        ),
                        // Switch camera button overlay
                        if (_cameras != null && _cameras!.length > 1)
                          Positioned(
                            bottom: 10,
                            right: 10,
                            child: FloatingActionButton(
                              mini: true,
                              onPressed: _switchCamera,
                              child: const Icon(Icons.flip_camera_ios),
                            ),
                          ),
                      ],
                    )
                  else
                    Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        children: [
                          Text(
                            _statusMessage,
                            textAlign: TextAlign.center,
                            style: const TextStyle(fontSize: 16),
                          ),
                          const SizedBox(height: 20),
                          // Only show permission request button if needed
                          if (_cameraPermissionStatus != null && !_cameraPermissionStatus!.isGranted)
                            ElevatedButton(
                              onPressed: _requestCameraPermission,
                              child: const Text('Grant Camera Permission'),
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
}
