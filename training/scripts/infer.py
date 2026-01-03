"""
Quick inference script to test the trained skin tone classifier.
Usage: python training/scripts/infer.py <image_path> [image_path2 ...]
       python training/scripts/infer.py --random N  # Test on N random images from test set
"""

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from training.scripts.model import SkinToneClassifier, load_checkpoint


CLASS_NAMES = ["Light", "Medium", "Dark"]
CLASS_SCALES = ["(Monk 1-3)", "(Monk 4-7)", "(Monk 8-10)"]

# Same transforms as validation
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load the trained 3-class model."""
    model = SkinToneClassifier(
        architecture="efficientnet_b0",
        num_classes=3,
        pretrained=False,
        dropout=0.3
    )
    load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()
    return model


def predict(model, image_path: str, device: str = "cpu"):
    """Run inference on a single image."""
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_tensor = TRANSFORM(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0]
        confidence, predicted = probs.max(dim=0)

    return {
        "class_idx": predicted.item(),
        "class_name": CLASS_NAMES[predicted.item()],
        "scales": CLASS_SCALES[predicted.item()],
        "confidence": confidence.item(),
        "all_probs": {
            CLASS_NAMES[i]: probs[i].item()
            for i in range(len(CLASS_NAMES))
        }
    }


def get_random_test_images(n: int = 5):
    """Get random images from the test set."""
    test_dir = Path("training/data/ccv2_faces/test")
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return []

    all_images = []
    for class_dir in test_dir.iterdir():
        if class_dir.is_dir():
            all_images.extend(list(class_dir.glob("*.jpg")))
            all_images.extend(list(class_dir.glob("*.png")))

    if not all_images:
        print("No images found in test directory")
        return []

    return random.sample(all_images, min(n, len(all_images)))


def main():
    parser = argparse.ArgumentParser(description="Run skin tone inference")
    parser.add_argument("images", nargs="*", help="Image paths to classify")
    parser.add_argument("--random", type=int, help="Test on N random images from test set")
    parser.add_argument("--checkpoint", default="training/checkpoints_3class/best_model.pth",
                        help="Path to model checkpoint")
    args = parser.parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully!\n")
    print("-" * 60)

    # Get images to process
    if args.random:
        images = get_random_test_images(args.random)
    elif args.images:
        images = [Path(p) for p in args.images]
    else:
        print("Usage: python infer.py <image_path> OR python infer.py --random N")
        return

    # Run inference
    for img_path in images:
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue

        result = predict(model, str(img_path), device)

        # Get ground truth from path if available
        gt = ""
        if "Light" in str(img_path):
            gt = " [GT: Light]"
        elif "Medium" in str(img_path):
            gt = " [GT: Medium]"
        elif "Dark" in str(img_path):
            gt = " [GT: Dark]"

        print(f"\nImage: {img_path.name}")
        print(f"  Prediction: {result['class_name']} {result['scales']}")
        print(f"  Confidence: {result['confidence']:.1%}{gt}")
        print(f"  All probabilities:")
        for cls, prob in result['all_probs'].items():
            bar = "#" * int(prob * 20)
            print(f"    {cls:8s}: {prob:.1%} {bar}")

    print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
