"""
Environment check script for training setup.
Verifies GPU availability, dependencies, and dataset paths.
"""

import sys
from pathlib import Path


def check_python():
    """Check Python version."""
    print(f"Python version: {sys.version}")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print("  WARNING: Python 3.10+ recommended")
    else:
        print("  OK")
    return True


def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    print("\n--- PyTorch ---")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")

            # Test GPU with simple operation
            print("\nGPU test:")
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.matmul(x, x)
            print(f"  Matrix multiplication on GPU: OK")
            del x, y
            torch.cuda.empty_cache()
        else:
            print("  WARNING: CUDA not available, training will use CPU (slow)")

            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print(f"  MPS (Apple Silicon) available: Yes")

        return True
    except ImportError:
        print("  ERROR: PyTorch not installed")
        return False


def check_torchvision():
    """Check torchvision installation."""
    print("\n--- torchvision ---")
    try:
        import torchvision
        print(f"torchvision version: {torchvision.__version__}")
        return True
    except ImportError:
        print("  ERROR: torchvision not installed")
        return False


def check_dependencies():
    """Check other required dependencies."""
    print("\n--- Dependencies ---")

    dependencies = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("sklearn", "scikit-learn"),
        ("tqdm", "tqdm"),
    ]

    all_ok = True
    for import_name, package_name in dependencies:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{package_name}: {version}")
        except ImportError:
            print(f"{package_name}: NOT INSTALLED")
            all_ok = False

    return all_ok


def check_dataset():
    """Check if dataset exists."""
    print("\n--- Dataset ---")

    base_path = Path("training/data/ccv2_balanced")

    if not base_path.exists():
        print(f"  ERROR: Dataset not found at {base_path}")
        return False

    splits = ["train", "val", "test"]
    total_files = 0

    for split in splits:
        split_path = base_path / split
        if not split_path.exists():
            print(f"  ERROR: {split} split not found")
            continue

        split_count = 0
        for scale in range(1, 11):
            scale_path = split_path / f"scale_{scale}"
            if scale_path.exists():
                count = len(list(scale_path.glob("*.jpg")))
                split_count += count

        print(f"{split}: {split_count:,} images")
        total_files += split_count

    print(f"Total: {total_files:,} images")

    if total_files == 0:
        print("  ERROR: No images found in dataset")
        return False

    return True


def check_output_dirs():
    """Check/create output directories."""
    print("\n--- Output Directories ---")

    dirs = [
        Path("training/checkpoints"),
        Path("training/logs"),
    ]

    for d in dirs:
        if d.exists():
            print(f"{d}: exists")
        else:
            d.mkdir(parents=True, exist_ok=True)
            print(f"{d}: created")

    return True


def estimate_training_time():
    """Estimate training time based on hardware."""
    print("\n--- Training Time Estimate ---")

    try:
        import torch

        # Rough estimates based on batch size 32, ~100k training images
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            mem_gb = props.total_memory / 1024**3

            if mem_gb >= 8:
                print(f"With {mem_gb:.0f}GB GPU: ~30-60 min for 30 epochs")
            else:
                print(f"With {mem_gb:.0f}GB GPU: ~1-2 hours for 30 epochs")
        else:
            print("With CPU only: ~6-12 hours for 30 epochs (not recommended)")

    except Exception:
        print("Could not estimate training time")


def main():
    print("=" * 60)
    print("ENVIRONMENT CHECK FOR SKIN TONE CLASSIFIER TRAINING")
    print("=" * 60)

    checks = [
        ("Python", check_python),
        ("PyTorch", check_pytorch),
        ("torchvision", check_torchvision),
        ("Dependencies", check_dependencies),
        ("Dataset", check_dataset),
        ("Output dirs", check_output_dirs),
    ]

    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    estimate_training_time()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "OK" if passed else "FAILED"
        print(f"  {name}: {status}")

    print()
    if all_passed:
        print("All checks passed! Ready to train.")
        print("\nRun training with:")
        print("  python training/scripts/train.py")
    else:
        print("Some checks failed. Please fix the issues above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
