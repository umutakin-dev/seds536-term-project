"""
ITA (Individual Typology Angle) Baseline for Skin Tone Classification.

This script implements the classical ITA method for skin tone classification
as a baseline comparison to the CNN-based approach.

ITA Formula:
    ITA = arctan((L* - 50) / b*) × (180/π)

Where L* and b* are from the CIE LAB color space.

ITA Ranges (literature-based):
    > 55°  : Very Light
    41°-55°: Light
    28°-41°: Intermediate
    10°-28°: Tan
    < 10°  : Dark

For our 3-class system:
    Light  (Monk 1-3): ITA > 41°
    Medium (Monk 4-7): ITA 10°-41°
    Dark   (Monk 8-10): ITA < 10°

Usage:
    # Without skin segmentation (original baseline)
    python ita_baseline.py --data-dir training/data/ccv2_faces/test

    # With skin segmentation (improved)
    python ita_baseline.py --data-dir training/data/ccv2_faces/test --use-skin-mask

References:
- Chardon et al., "Skin colour typology and suntanning pathways" (1991)
- Del Bino & Bernerd, "Variations in skin colour and the biological
  consequences of UV radiation exposure" (2013)
"""

import argparse
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from skin_segmentation import segment_skin


# ITA thresholds for 3-class mapping
# These can be tuned based on dataset characteristics
ITA_THRESHOLDS = {
    "light_min": 41.0,   # ITA > 41 -> Light
    "dark_max": 10.0,    # ITA < 10 -> Dark
    # 10 <= ITA <= 41 -> Medium
}

# Class names matching CNN model
CLASS_NAMES = ["Light", "Medium", "Dark"]

# Monk scale to 3-class mapping (same as CNN training)
MONK_TO_3CLASS = {
    1: 0, 2: 0, 3: 0,        # Light (scales 1-3)
    4: 1, 5: 1, 6: 1, 7: 1,  # Medium (scales 4-7)
    8: 2, 9: 2, 10: 2,       # Dark (scales 8-10)
}


def calculate_ita(image_bgr: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Calculate Individual Typology Angle (ITA) from an image.

    Args:
        image_bgr: Image in BGR format (OpenCV default)
        mask: Optional binary mask for skin region (255=skin, 0=background)

    Returns:
        ITA angle in degrees
    """
    # Convert BGR to LAB color space
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # OpenCV LAB ranges: L* [0, 255], a* [0, 255], b* [0, 255]
    # Need to convert to standard LAB: L* [0, 100], a* [-128, 127], b* [-128, 127]
    L = image_lab[:, :, 0].astype(np.float32) * 100.0 / 255.0
    b = image_lab[:, :, 2].astype(np.float32) - 128.0

    if mask is not None:
        # Use only skin pixels
        skin_mask = mask > 0
        L = L[skin_mask]
        b = b[skin_mask]
    else:
        # Use all pixels - flatten
        L = L.flatten()
        b = b.flatten()

    if len(L) == 0:
        return 0.0  # No valid pixels

    # Calculate mean L* and b*
    L_mean = np.mean(L)
    b_mean = np.mean(b)

    # Handle edge case where b* is zero (avoid division by zero)
    if abs(b_mean) < 1e-6:
        b_mean = 1e-6

    # Calculate ITA: arctan((L* - 50) / b*) × (180/π)
    ita = math.atan((L_mean - 50.0) / b_mean) * (180.0 / math.pi)

    return ita


def ita_to_class(ita: float, thresholds: dict = None) -> int:
    """
    Map ITA value to 3-class label.

    Args:
        ita: ITA angle in degrees
        thresholds: Dict with 'light_min' and 'dark_max' keys

    Returns:
        Class label: 0=Light, 1=Medium, 2=Dark
    """
    if thresholds is None:
        thresholds = ITA_THRESHOLDS

    if ita > thresholds["light_min"]:
        return 0  # Light
    elif ita < thresholds["dark_max"]:
        return 2  # Dark
    else:
        return 1  # Medium


def process_single_image(
    sample: tuple,
    use_preprocessed: bool,
    use_skin_mask: bool,
) -> dict:
    """
    Process a single image and compute ITA.

    Args:
        sample: (image_path, label, mask_path) tuple
        use_preprocessed: Whether to use preprocessed masks
        use_skin_mask: Whether to compute masks on-the-fly

    Returns:
        Dict with ita, label, skin_ratio, success
    """
    img_path, label, mask_path = sample

    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        return {"success": False, "path": str(img_path)}

    # Get skin mask
    mask = None
    skin_ratio = None

    if use_preprocessed and mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            skin_ratio = np.sum(mask > 0) / mask.size
    elif use_skin_mask:
        mask = segment_skin(image, method="ycbcr", apply_morphology=True)
        if mask is not None:
            skin_ratio = np.sum(mask > 0) / mask.size

    # Calculate ITA
    ita = calculate_ita(image, mask=mask)

    return {
        "success": True,
        "ita": ita,
        "label": label,
        "skin_ratio": skin_ratio,
    }


def load_dataset(data_dir: str, use_preprocessed: bool = False) -> list[tuple[Path, int, Path | None]]:
    """
    Load dataset from directory structure.

    Expected structure (original):
        data_dir/
        ├── scale_1/
        │   ├── image1.jpg
        │   └── ...
        ├── scale_2/
        └── ...

    Expected structure (preprocessed):
        data_dir/
        ├── scale_1/
        │   ├── image1_masked.jpg
        │   ├── image1_mask.png
        │   └── ...
        ├── scale_2/
        └── ...

    Args:
        data_dir: Path to dataset directory
        use_preprocessed: If True, look for _masked.jpg and _mask.png files

    Returns:
        List of (image_path, 3-class label, mask_path or None) tuples
    """
    data_path = Path(data_dir)
    samples = []

    for scale in range(1, 11):
        scale_dir = data_path / f"scale_{scale}"
        if not scale_dir.exists():
            print(f"Warning: {scale_dir} does not exist")
            continue

        label = MONK_TO_3CLASS[scale]

        if use_preprocessed:
            # Look for _masked.jpg files and corresponding _mask.png
            images = list(scale_dir.glob("*_masked.jpg"))
            for img_path in images:
                # Derive mask path from masked image path
                mask_path = img_path.parent / (img_path.stem.replace("_masked", "_mask") + ".png")
                if mask_path.exists():
                    samples.append((img_path, label, mask_path))
                else:
                    # Fallback: use masked image without separate mask
                    samples.append((img_path, label, None))
        else:
            # Original behavior: look for .jpg files
            images = list(scale_dir.glob("*.jpg"))
            for img_path in images:
                samples.append((img_path, label, None))

    return samples


def evaluate_ita(
    data_dir: str,
    thresholds: dict = None,
    output_file: str = None,
    verbose: bool = True,
    use_skin_mask: bool = False,
    use_preprocessed: bool = False,
    num_workers: int = 1,
) -> dict:
    """
    Evaluate ITA baseline on dataset.

    Args:
        data_dir: Path to dataset directory
        thresholds: Optional custom ITA thresholds
        output_file: Optional path to save results JSON
        verbose: Print progress and results
        use_skin_mask: If True, apply skin segmentation on-the-fly before ITA calculation
        use_preprocessed: If True, use preprocessed _masked.jpg and _mask.png files
        num_workers: Number of parallel workers (default 1 = sequential)

    Returns:
        Dictionary with evaluation metrics
    """
    if thresholds is None:
        thresholds = ITA_THRESHOLDS

    # Load dataset
    samples = load_dataset(data_dir, use_preprocessed=use_preprocessed)
    if verbose:
        print(f"Loaded {len(samples)} images from {data_dir}")
        if use_preprocessed:
            print("Using preprocessed masks (_mask.png files)")
        elif use_skin_mask:
            print("Using skin segmentation (YCbCr thresholding + morphological cleanup)")

    # Count class distribution
    class_counts = Counter(label for _, label, _ in samples)
    if verbose:
        print(f"Class distribution: {dict(sorted(class_counts.items()))}")
        for cls_id, count in sorted(class_counts.items()):
            print(f"  {CLASS_NAMES[cls_id]}: {count}")

    # Evaluate
    predictions = []
    ground_truth = []
    ita_values = []
    skin_ratios = []  # Track skin detection success
    failed_segmentations = 0

    # Set progress description
    if use_preprocessed:
        desc = "Computing ITA (preprocessed masks)"
    elif use_skin_mask:
        desc = "Computing ITA (on-the-fly skin mask)"
    else:
        desc = "Computing ITA"

    if num_workers > 1:
        # Parallel processing
        if verbose:
            print(f"Using {num_workers} workers")

        process_fn = partial(
            process_single_image,
            use_preprocessed=use_preprocessed,
            use_skin_mask=use_skin_mask,
        )

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_fn, sample): sample for sample in samples}

            for future in tqdm(as_completed(futures), total=len(samples), desc=desc, disable=not verbose):
                result = future.result()

                if not result["success"]:
                    failed_segmentations += 1
                    continue

                ita_values.append(result["ita"])
                ground_truth.append(result["label"])
                predictions.append(ita_to_class(result["ita"], thresholds))

                if result["skin_ratio"] is not None:
                    skin_ratios.append(result["skin_ratio"])
    else:
        # Sequential processing (original behavior)
        for img_path, label, mask_path in tqdm(samples, desc=desc, disable=not verbose):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load {img_path}")
                continue

            # Get skin mask
            mask = None
            if use_preprocessed and mask_path is not None:
                # Load pre-saved mask
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    skin_ratio = np.sum(mask > 0) / mask.size
                    skin_ratios.append(skin_ratio)
                else:
                    failed_segmentations += 1
            elif use_skin_mask:
                # Compute mask on-the-fly
                mask = segment_skin(image, method="ycbcr", apply_morphology=True)
                if mask is None:
                    # Skin segmentation failed, fall back to no mask
                    failed_segmentations += 1
                    mask = None
                else:
                    skin_ratio = np.sum(mask > 0) / mask.size
                    skin_ratios.append(skin_ratio)

            # Calculate ITA
            ita = calculate_ita(image, mask=mask)
            ita_values.append(ita)

            # Predict class
            pred = ita_to_class(ita, thresholds)
            predictions.append(pred)
            ground_truth.append(label)

    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    ita_values = np.array(ita_values)

    # Calculate metrics
    num_classes = 3

    # Overall accuracy
    correct = (predictions == ground_truth).sum()
    total = len(ground_truth)
    accuracy = correct / total

    # Per-class metrics
    per_class = {}
    for cls_id in range(num_classes):
        cls_mask = ground_truth == cls_id
        cls_total = cls_mask.sum()

        if cls_total > 0:
            cls_correct = ((predictions == cls_id) & cls_mask).sum()
            cls_accuracy = cls_correct / cls_total

            # Precision: TP / (TP + FP)
            pred_cls_mask = predictions == cls_id
            tp = ((predictions == cls_id) & (ground_truth == cls_id)).sum()
            fp = ((predictions == cls_id) & (ground_truth != cls_id)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            # Recall: TP / (TP + FN) = cls_accuracy
            recall = cls_accuracy

            # F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            cls_accuracy = 0
            precision = 0
            recall = 0
            f1 = 0

        per_class[CLASS_NAMES[cls_id]] = {
            "accuracy": float(cls_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cls_total),
        }

    # Confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(ground_truth, predictions):
        confusion[true, pred] += 1

    # Macro F1
    macro_f1 = np.mean([per_class[name]["f1"] for name in CLASS_NAMES])

    # Weighted F1
    total_support = sum(per_class[name]["support"] for name in CLASS_NAMES)
    weighted_f1 = sum(
        per_class[name]["f1"] * per_class[name]["support"] / total_support
        for name in CLASS_NAMES
    )

    # ITA statistics per class
    ita_stats = {}
    for cls_id in range(num_classes):
        cls_mask = ground_truth == cls_id
        if cls_mask.sum() > 0:
            cls_ita = ita_values[cls_mask]
            ita_stats[CLASS_NAMES[cls_id]] = {
                "mean": float(np.mean(cls_ita)),
                "std": float(np.std(cls_ita)),
                "min": float(np.min(cls_ita)),
                "max": float(np.max(cls_ita)),
                "median": float(np.median(cls_ita)),
            }

    # Compile results
    if use_preprocessed:
        method_name = "ITA + Preprocessed Skin Masks"
    elif use_skin_mask:
        method_name = "ITA + Skin Segmentation (on-the-fly)"
    else:
        method_name = "ITA Baseline"

    results = {
        "method": method_name,
        "use_skin_mask": use_skin_mask,
        "use_preprocessed": use_preprocessed,
        "thresholds": thresholds,
        "dataset": str(data_dir),
        "num_samples": total,
        "overall": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "ita_statistics": ita_stats,
    }

    # Add skin segmentation stats if used
    if use_skin_mask and skin_ratios:
        results["skin_segmentation"] = {
            "failed_segmentations": failed_segmentations,
            "mean_skin_ratio": float(np.mean(skin_ratios)),
            "std_skin_ratio": float(np.std(skin_ratios)),
            "min_skin_ratio": float(np.min(skin_ratios)),
            "max_skin_ratio": float(np.max(skin_ratios)),
        }

    # Print results
    if verbose:
        print("\n" + "=" * 60)
        print(f"{method_name.upper()} RESULTS")
        print("=" * 60)
        if use_preprocessed:
            print("\nSkin Masks: PREPROCESSED (loaded from _mask.png files)")
            if skin_ratios:
                print(f"  Mean skin ratio: {np.mean(skin_ratios):.1%}")
                print(f"  Failed mask loads: {failed_segmentations}")
        elif use_skin_mask:
            print("\nSkin Segmentation: ON-THE-FLY (YCbCr + morphological cleanup)")
            if skin_ratios:
                print(f"  Mean skin ratio: {np.mean(skin_ratios):.1%}")
                print(f"  Failed segmentations: {failed_segmentations}")
        print(f"\nThresholds: Light > {thresholds['light_min']}°, "
              f"Dark < {thresholds['dark_max']}°")
        print(f"\nOverall Accuracy: {accuracy:.1%}")
        print(f"Macro F1: {macro_f1:.3f}")
        print(f"Weighted F1: {weighted_f1:.3f}")

        print("\nPer-Class Performance:")
        print("-" * 60)
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 60)
        for name in CLASS_NAMES:
            m = per_class[name]
            print(f"{name:<10} {m['precision']:.3f}        {m['recall']:.3f}        "
                  f"{m['f1']:.3f}        {m['support']:<10}")

        print("\nConfusion Matrix:")
        print("-" * 40)
        print(f"{'Pred →':<10}", end="")
        for name in CLASS_NAMES:
            print(f"{name:<10}", end="")
        print()
        print("-" * 40)
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name:<10}", end="")
            for j in range(num_classes):
                print(f"{confusion[i, j]:<10}", end="")
            print()

        print("\nITA Statistics by Class:")
        print("-" * 60)
        for name in CLASS_NAMES:
            if name in ita_stats:
                s = ita_stats[name]
                print(f"{name}: mean={s['mean']:.1f}°, std={s['std']:.1f}°, "
                      f"range=[{s['min']:.1f}°, {s['max']:.1f}°]")

    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {output_file}")

    return results


def tune_thresholds(
    data_dir: str,
    light_range: tuple = (-30, 30, 5),
    dark_range: tuple = (-50, 0, 5),
    verbose: bool = True,
    use_preprocessed: bool = False,
    use_skin_mask: bool = False,
    num_workers: int = 1,
) -> dict:
    """
    Grid search for optimal ITA thresholds.

    Args:
        data_dir: Path to validation dataset
        light_range: (start, stop, step) for light threshold
        dark_range: (start, stop, step) for dark threshold
        verbose: Print progress
        use_preprocessed: Use preprocessed masks
        use_skin_mask: Use on-the-fly skin segmentation
        num_workers: Number of parallel workers

    Returns:
        Best thresholds and accuracy
    """
    samples = load_dataset(data_dir, use_preprocessed=use_preprocessed)
    print(f"Tuning thresholds on {len(samples)} samples...")

    # Pre-compute ITA values
    ita_values = []
    labels = []

    if num_workers > 1:
        process_fn = partial(
            process_single_image,
            use_preprocessed=use_preprocessed,
            use_skin_mask=use_skin_mask,
        )

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_fn, sample): sample for sample in samples}

            for future in tqdm(as_completed(futures), total=len(samples), desc="Computing ITA"):
                result = future.result()
                if result["success"]:
                    ita_values.append(result["ita"])
                    labels.append(result["label"])
    else:
        for img_path, label, mask_path in tqdm(samples, desc="Computing ITA"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            mask = None
            if use_preprocessed and mask_path is not None:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            elif use_skin_mask:
                mask = segment_skin(image, method="ycbcr", apply_morphology=True)

            ita = calculate_ita(image, mask=mask)
            ita_values.append(ita)
            labels.append(label)

    ita_values = np.array(ita_values)
    labels = np.array(labels)

    # Grid search
    best_accuracy = 0
    best_thresholds = None

    light_vals = np.arange(*light_range)
    dark_vals = np.arange(*dark_range)

    for light_min in light_vals:
        for dark_max in dark_vals:
            if dark_max >= light_min:
                continue  # Invalid threshold combination

            # Classify
            preds = np.where(
                ita_values > light_min, 0,
                np.where(ita_values < dark_max, 2, 1)
            )

            accuracy = (preds == labels).mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thresholds = {
                    "light_min": float(light_min),
                    "dark_max": float(dark_max),
                }

    if verbose:
        print(f"\nBest thresholds: Light > {best_thresholds['light_min']}°, "
              f"Dark < {best_thresholds['dark_max']}°")
        print(f"Best accuracy: {best_accuracy:.1%}")

    return {
        "best_thresholds": best_thresholds,
        "best_accuracy": float(best_accuracy),
    }


def main():
    parser = argparse.ArgumentParser(
        description="ITA Baseline for Skin Tone Classification"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="training/data/ccv2_faces/test",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/results_ita_baseline.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune thresholds on validation set first",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="training/data/ccv2_faces/val",
        help="Validation set for threshold tuning",
    )
    parser.add_argument(
        "--light-threshold",
        type=float,
        default=41.0,
        help="ITA threshold for Light class (> threshold)",
    )
    parser.add_argument(
        "--dark-threshold",
        type=float,
        default=10.0,
        help="ITA threshold for Dark class (< threshold)",
    )
    parser.add_argument(
        "--use-skin-mask",
        action="store_true",
        help="Apply skin segmentation on-the-fly (YCbCr thresholding) before ITA calculation",
    )
    parser.add_argument(
        "--use-preprocessed",
        action="store_true",
        help="Use preprocessed _masked.jpg and _mask.png files from preprocessing script",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default 1 = sequential)",
    )
    parser.add_argument(
        "--light-range",
        type=str,
        default="-30,30,5",
        help="Light threshold range for tuning: start,stop,step (default: -30,30,5)",
    )
    parser.add_argument(
        "--dark-range",
        type=str,
        default="-50,0,5",
        help="Dark threshold range for tuning: start,stop,step (default: -50,0,5)",
    )

    args = parser.parse_args()

    # Set thresholds
    thresholds = {
        "light_min": args.light_threshold,
        "dark_max": args.dark_threshold,
    }

    # Optional: tune thresholds on validation set
    if args.tune:
        # Parse threshold ranges
        light_range = tuple(int(x) for x in args.light_range.split(","))
        dark_range = tuple(int(x) for x in args.dark_range.split(","))

        print("Tuning thresholds on validation set...")
        print(f"  Light range: {light_range[0]} to {light_range[1]}, step {light_range[2]}")
        print(f"  Dark range: {dark_range[0]} to {dark_range[1]}, step {dark_range[2]}")
        tune_results = tune_thresholds(
            args.val_dir,
            light_range=light_range,
            dark_range=dark_range,
            use_preprocessed=args.use_preprocessed,
            use_skin_mask=args.use_skin_mask,
            num_workers=args.workers,
        )
        thresholds = tune_results["best_thresholds"]
        print()

    # Determine output file name
    output_file = args.output
    if output_file == "training/results_ita_baseline.json":
        if args.use_preprocessed:
            output_file = "training/results_ita_preprocessed.json"
        elif args.use_skin_mask:
            output_file = "training/results_ita_skin_segmented.json"

    # Evaluate on test set
    if args.use_preprocessed:
        method_desc = "ITA + preprocessed skin masks"
    elif args.use_skin_mask:
        method_desc = "ITA + skin segmentation (on-the-fly)"
    else:
        method_desc = "ITA baseline"
    print(f"Evaluating {method_desc} on: {args.data_dir}")
    results = evaluate_ita(
        args.data_dir,
        thresholds=thresholds,
        output_file=output_file,
        use_skin_mask=args.use_skin_mask,
        use_preprocessed=args.use_preprocessed,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
