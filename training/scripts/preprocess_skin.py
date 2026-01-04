"""
Skin Segmentation Preprocessing Script.

This script preprocesses the face dataset by applying skin segmentation
and saving multiple versions of each image for flexible use in experiments.

Output per image:
    - original.jpg      : Copy of original image
    - original_masked.jpg : Skin regions only (background = black)
    - original_mask.png  : Binary mask (255=skin, 0=background)

Usage:
    python preprocess_skin.py --input training/data/ccv2_faces --output training/data/ccv2_faces_preprocessed

This allows:
    - ITA evaluation on masked images
    - CNN training on masked images
    - Applying additional preprocessing (CLAHE, gamma) using saved masks
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm

from skin_segmentation import segment_skin, segment_skin_with_oval, get_skin_statistics


def process_single_image(
    img_path: Path,
    input_base: Path,
    output_base: Path,
    save_original: bool = True,
    background_color: tuple = (0, 0, 0),
    largest_component_only: bool = False,
    use_face_roi: bool = False,
    roi_padding: float = 0.05,
    use_oval: bool = False,
    oval_width: float = 0.7,
    oval_height: float = 0.85,
) -> dict:
    """
    Process a single image: apply skin segmentation and save outputs.

    Args:
        img_path: Path to input image
        input_base: Base input directory (for computing relative path)
        output_base: Base output directory
        save_original: Whether to copy the original image
        background_color: BGR color for masked background
        largest_component_only: Keep only largest connected component
        use_face_roi: Restrict to center ROI
        roi_padding: Padding for face ROI
        use_oval: Use face oval mask combined with color detection
        oval_width: Oval width ratio (0.7 = 70% of image width)
        oval_height: Oval height ratio (0.85 = 85% of image height)

    Returns:
        Dict with processing stats
    """
    # Compute relative path and output paths
    rel_path = img_path.relative_to(input_base)
    output_dir = output_base / rel_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    suffix = img_path.suffix

    # Output file paths
    original_out = output_dir / f"{stem}{suffix}"
    masked_out = output_dir / f"{stem}_masked{suffix}"
    mask_out = output_dir / f"{stem}_mask.png"

    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        return {"path": str(img_path), "success": False, "error": "Could not load image"}

    # Apply skin segmentation with options
    if use_oval:
        # Use face oval + color segmentation (best for face crops)
        mask = segment_skin_with_oval(
            image,
            width_ratio=oval_width,
            height_ratio=oval_height,
            combine_with_color=True,
            method="ycbcr",
        )
    else:
        # Use traditional color-based segmentation
        mask = segment_skin(
            image,
            method="ycbcr",
            apply_morphology=True,
            min_skin_ratio=0.01,
            largest_component_only=largest_component_only,
            use_face_roi=use_face_roi,
            roi_padding=roi_padding,
        )

    if mask is None:
        # Fallback: create empty mask (no skin detected)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        skin_ratio = 0.0
    else:
        skin_ratio = np.sum(mask > 0) / mask.size

    # Create masked image (skin only, background = specified color)
    masked_image = np.full_like(image, background_color, dtype=np.uint8)
    masked_image[mask > 0] = image[mask > 0]

    # Save outputs
    if save_original:
        shutil.copy2(img_path, original_out)
    cv2.imwrite(str(masked_out), masked_image)
    cv2.imwrite(str(mask_out), mask)

    return {
        "path": str(rel_path),
        "success": True,
        "skin_ratio": skin_ratio,
    }


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    num_workers: int = 8,
    save_original: bool = True,
    background_color: tuple = (0, 0, 0),
    largest_component_only: bool = False,
    use_face_roi: bool = False,
    roi_padding: float = 0.05,
    use_oval: bool = False,
    oval_width: float = 0.7,
    oval_height: float = 0.85,
) -> dict:
    """
    Preprocess entire dataset with skin segmentation.

    Args:
        input_dir: Input dataset directory
        output_dir: Output directory for preprocessed data
        num_workers: Number of parallel workers
        save_original: Whether to copy original images
        background_color: BGR color for masked background
        largest_component_only: Keep only largest connected component
        use_face_roi: Restrict to center ROI
        roi_padding: Padding for face ROI
        use_oval: Use face oval mask combined with color detection
        oval_width: Oval width ratio (0.7 = 70% of image width)
        oval_height: Oval height ratio (0.85 = 85% of image height)

    Returns:
        Dict with preprocessing statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png"}
    all_images = []

    for ext in image_extensions:
        all_images.extend(input_path.rglob(f"*{ext}"))
        all_images.extend(input_path.rglob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    all_images = sorted(set(all_images))
    print(f"Found {len(all_images)} images in {input_dir}")

    if len(all_images) == 0:
        print("No images found!")
        return {"success": False, "error": "No images found"}

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Process images in parallel
    results = []
    failed = []
    skin_ratios = []

    # Use ProcessPoolExecutor for CPU-bound work
    process_fn = partial(
        process_single_image,
        input_base=input_path,
        output_base=output_path,
        save_original=save_original,
        background_color=background_color,
        largest_component_only=largest_component_only,
        use_face_roi=use_face_roi,
        roi_padding=roi_padding,
        use_oval=use_oval,
        oval_width=oval_width,
        oval_height=oval_height,
    )

    # Process with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_fn, img): img for img in all_images}

        for future in tqdm(as_completed(futures), total=len(all_images), desc="Preprocessing"):
            result = future.result()
            results.append(result)

            if result["success"]:
                skin_ratios.append(result["skin_ratio"])
            else:
                failed.append(result)

    # Compute statistics
    skin_ratios = np.array(skin_ratios)
    stats = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_images": len(all_images),
        "successful": len(skin_ratios),
        "failed": len(failed),
        "skin_ratio_stats": {
            "mean": float(np.mean(skin_ratios)) if len(skin_ratios) > 0 else 0,
            "std": float(np.std(skin_ratios)) if len(skin_ratios) > 0 else 0,
            "min": float(np.min(skin_ratios)) if len(skin_ratios) > 0 else 0,
            "max": float(np.max(skin_ratios)) if len(skin_ratios) > 0 else 0,
            "median": float(np.median(skin_ratios)) if len(skin_ratios) > 0 else 0,
        },
        "failed_images": [f["path"] for f in failed],
    }

    # Save stats to output directory
    stats_file = output_path / "preprocessing_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"\nProcessed: {stats['successful']}/{stats['total_images']} images")
    if stats['failed'] > 0:
        print(f"Failed: {stats['failed']} images")
    print(f"\nSkin Ratio Statistics:")
    print(f"  Mean:   {stats['skin_ratio_stats']['mean']:.1%}")
    print(f"  Std:    {stats['skin_ratio_stats']['std']:.1%}")
    print(f"  Range:  [{stats['skin_ratio_stats']['min']:.1%}, {stats['skin_ratio_stats']['max']:.1%}]")
    print(f"\nOutput files per image:")
    print(f"  - <name>.jpg        : Original image" if save_original else "  - (original not saved)")
    print(f"  - <name>_masked.jpg : Skin only (background=black)")
    print(f"  - <name>_mask.png   : Binary mask")
    print(f"\nStats saved to: {stats_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset with skin segmentation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="training/data/ccv2_faces",
        help="Input dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/data/ccv2_faces_preprocessed",
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="Don't copy original images (save disk space)",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="black",
        choices=["black", "gray", "white"],
        help="Background color for masked images",
    )
    parser.add_argument(
        "--largest-component",
        action="store_true",
        help="Keep only the largest connected component (removes scattered noise)",
    )
    parser.add_argument(
        "--face-roi",
        action="store_true",
        help="Restrict mask to center ROI (assumes face is centered)",
    )
    parser.add_argument(
        "--roi-padding",
        type=float,
        default=0.05,
        help="Padding for face ROI (0.05 = 5%% on each side)",
    )
    parser.add_argument(
        "--use-oval",
        action="store_true",
        help="Use face oval mask combined with color detection (best for face crops)",
    )
    parser.add_argument(
        "--oval-width",
        type=float,
        default=0.7,
        help="Oval width ratio (0.7 = 70%% of image width). Try 0.5-0.6 for smaller oval.",
    )
    parser.add_argument(
        "--oval-height",
        type=float,
        default=0.85,
        help="Oval height ratio (0.85 = 85%% of image height). Try 0.7-0.75 for smaller oval.",
    )

    args = parser.parse_args()

    # Parse background color
    bg_colors = {
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "white": (255, 255, 255),
    }
    background_color = bg_colors[args.background]

    # Run preprocessing
    preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output,
        num_workers=args.workers,
        save_original=not args.no_original,
        background_color=background_color,
        largest_component_only=args.largest_component,
        use_face_roi=args.face_roi,
        roi_padding=args.roi_padding,
        use_oval=args.use_oval,
        oval_width=args.oval_width,
        oval_height=args.oval_height,
    )


if __name__ == "__main__":
    main()
