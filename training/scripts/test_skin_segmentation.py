"""
Test script for comparing skin segmentation methods.

Compares:
1. Color-based (YCbCr)
2. Face oval only
3. Oval + color combined

Usage:
    uv run python training/scripts/test_skin_segmentation.py
"""

import cv2
import numpy as np
from pathlib import Path

from skin_segmentation import (
    segment_skin,
    segment_skin_with_oval,
    create_face_oval_mask,
)


def test_image(img_path: str, output_dir: Path):
    """Test all segmentation variants on a single image."""
    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not load: {img_path}")
        return

    name = Path(img_path).stem

    # Method 1: Color-based only
    mask_color = segment_skin(
        image, method='ycbcr', apply_morphology=True
    )

    # Method 2: Oval only (no color)
    mask_oval = create_face_oval_mask(
        image, width_ratio=0.7, height_ratio=0.85
    )

    # Method 3: Oval + color combined
    mask_oval_color = segment_skin_with_oval(
        image, width_ratio=0.7, height_ratio=0.85,
        combine_with_color=True, method='ycbcr'
    )

    # Method 4: Smaller oval + color (tighter)
    mask_tight = segment_skin_with_oval(
        image, width_ratio=0.6, height_ratio=0.75,
        combine_with_color=True, method='ycbcr'
    )

    # Calculate skin ratios
    def skin_ratio(mask):
        if mask is None:
            return 0.0
        return np.sum(mask > 0) / mask.size

    print(f"\n{name}:")
    print(f"  Color only:        {skin_ratio(mask_color):.1%}")
    print(f"  Oval only:         {skin_ratio(mask_oval):.1%}")
    print(f"  Oval + color:      {skin_ratio(mask_oval_color):.1%}")
    print(f"  Tight oval+color:  {skin_ratio(mask_tight):.1%}")

    # Create side-by-side comparison
    h, w = image.shape[:2]

    # Resize for display
    scale = 150 / max(h, w)
    new_size = (int(w * scale), int(h * scale))

    img_small = cv2.resize(image, new_size)

    def resize_mask(mask):
        if mask is None:
            return np.zeros((new_size[1], new_size[0]), dtype=np.uint8)
        return cv2.resize(mask, new_size)

    # Convert masks to BGR for concatenation
    masks = [mask_color, mask_oval, mask_oval_color, mask_tight]
    masks_bgr = [cv2.cvtColor(resize_mask(m), cv2.COLOR_GRAY2BGR) for m in masks]

    # Create comparison image
    comparison = np.concatenate([img_small] + masks_bgr, axis=1)

    # Add labels
    labels = ["Original", "Color", "Oval", "Oval+Color", "Tight"]
    for i, label in enumerate(labels):
        x = i * new_size[0] + 5
        cv2.putText(comparison, label, (x, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Save comparison
    output_path = output_dir / f"{name}_oval_comparison.jpg"
    cv2.imwrite(str(output_path), comparison)
    print(f"  Saved: {output_path}")

    # Save the best mask
    cv2.imwrite(str(output_dir / f"{name}_mask_oval_color.png"), mask_oval_color)


def main():
    output_dir = Path("training/data/segmentation_test")
    output_dir.mkdir(exist_ok=True)

    # Test images - good and bad examples
    test_images = [
        # Good example
        "training/data/ccv2_faces/train/scale_3/0007_portuguese_nonscripted_1_raw_frame00001775.jpg",
        # Bad examples
        "training/data/ccv2_faces/train/scale_3/0017_portuguese_nonscripted_2_raw_frame00002326.jpg",
        "training/data/ccv2_faces/train/scale_3/0029_portuguese_nonscripted_3_raw_frame00000651.jpg",
        # Previous test images
        "training/data/ccv2_faces/train/scale_1/0701_portuguese_nonscripted_3_raw_frame00000242.jpg",
        "training/data/ccv2_faces/train/scale_1/0513_portuguese_scripted_0_raw_frame00003062.jpg",
    ]

    print("Testing face oval segmentation...")
    print("=" * 60)

    for img_path in test_images:
        if Path(img_path).exists():
            test_image(img_path, output_dir)
        else:
            print(f"Not found: {img_path}")

    print("\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("\nComparison: Original | Color | Oval | Oval+Color | Tight")


if __name__ == "__main__":
    main()
