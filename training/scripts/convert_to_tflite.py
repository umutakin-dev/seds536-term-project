"""
Convert PyTorch skin tone classifier to TFLite format for mobile deployment.

Uses Google's ai-edge-torch for direct PyTorch -> TFLite conversion.

IMPORTANT: This script must be run on Linux (WSL on Windows) because:
- ai-edge-torch/ai-edge-litert only provides Linux wheels
- TensorFlow has limited Windows support for conversion tools

Usage (from WSL on Windows):
    cd /mnt/c/Users/ydran/workspace/seds/seds536-image-understanding/seds536-term-project
    uv sync --extra conversion --python 3.12
    uv run --python 3.12 python training/scripts/convert_to_tflite.py

Requirements:
    - Linux or WSL (Windows Subsystem for Linux)
    - Python 3.11-3.13 (TensorFlow doesn't support 3.14+)
    - uv package manager

Output:
    training/models/skin_tone_classifier.tflite
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from training.scripts.model import SkinToneClassifier, load_checkpoint


def convert_to_tflite(model, output_path: Path, input_size: tuple = (1, 3, 224, 224)):
    """Convert PyTorch model to TFLite using ai-edge-torch."""
    import ai_edge_torch

    print(f"\n[1/2] Converting PyTorch to TFLite...")

    model.eval()
    sample_input = (torch.randn(*input_size),)

    # Convert using ai-edge-torch
    edge_model = ai_edge_torch.convert(model, sample_input)

    # Export to TFLite
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(output_path))

    print(f"    TFLite model saved: {output_path}")
    print(f"    Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def verify_tflite(tflite_path: Path):
    """Quick verification of the TFLite model."""
    print(f"\n[2/2] Verifying TFLite model...")

    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"    Input shape: {input_details[0]['shape']}")
    print(f"    Input dtype: {input_details[0]['dtype']}")
    print(f"    Output shape: {output_details[0]['shape']}")
    print(f"    Output dtype: {output_details[0]['dtype']}")

    # Test inference
    input_shape = input_details[0]['shape']
    test_input = np.random.randn(*input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Apply softmax to see probabilities
    probs = np.exp(output) / np.exp(output).sum()
    print(f"    Test inference: OK")
    print(f"    Sample output probs: {probs[0]}")


def compare_outputs(pytorch_model, tflite_path: Path):
    """Compare PyTorch and TFLite outputs for consistency."""
    print(f"\n[Bonus] Comparing PyTorch vs TFLite outputs...")

    import numpy as np
    import tensorflow as tf

    # Create identical input
    np.random.seed(42)
    test_input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    test_input_torch = torch.from_numpy(test_input_np)

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input_torch).numpy()

    # TFLite inference
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], test_input_np)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    # Compare
    max_diff = np.abs(pytorch_output - tflite_output).max()
    print(f"    PyTorch output: {pytorch_output[0]}")
    print(f"    TFLite output:  {tflite_output[0]}")
    print(f"    Max difference: {max_diff:.6f}")

    if max_diff < 1e-4:
        print(f"    Status: EXCELLENT (< 0.0001)")
    elif max_diff < 1e-2:
        print(f"    Status: GOOD (< 0.01)")
    else:
        print(f"    Status: WARNING - outputs differ significantly")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TFLite")
    parser.add_argument("--checkpoint", default="training/checkpoints_3class/best_model.pth",
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--output", default="training/models/skin_tone_classifier.tflite",
                        help="Output TFLite path")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    print("=" * 60)
    print("Skin Tone Classifier: PyTorch -> TFLite Conversion")
    print("=" * 60)

    # Load PyTorch model
    print(f"\n[0/2] Loading PyTorch model: {checkpoint_path}")
    model = SkinToneClassifier(
        architecture="efficientnet_b0",
        num_classes=3,
        pretrained=False,
        dropout=0.0  # Disable dropout for inference
    )
    load_checkpoint(model, str(checkpoint_path), device="cpu")
    model.eval()
    print(f"    Model loaded successfully")

    # Convert to TFLite
    convert_to_tflite(model, output_path)

    # Verify
    verify_tflite(output_path)

    # Compare outputs
    compare_outputs(model, output_path)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"TFLite model: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
