"""
Face extraction preprocessing script.
Detects faces in images and saves cropped face regions.
Supports parallel processing for speed.
"""

import argparse
import time
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass
import json

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class FaceExtractionResult:
    """Result of face extraction for a single image."""
    input_path: str
    output_path: str | None
    success: bool
    face_found: bool
    error: str | None = None


def load_detector():
    """Load OpenCV's Haar cascade face detector."""
    model_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(model_file)


def detect_and_crop_face(
    image: np.ndarray,
    detector: cv2.CascadeClassifier,
    padding: float = 0.2,
    min_face_size: int = 50
) -> np.ndarray | None:
    """
    Detect face and return cropped region with padding.

    Args:
        image: Input image (BGR)
        detector: Haar cascade detector
        padding: Padding around face as fraction of face size
        min_face_size: Minimum face size to detect

    Returns:
        Cropped face image or None if no face found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(min_face_size, min_face_size)
    )

    if len(faces) == 0:
        return None

    # Get largest face
    areas = [w * h for (x, y, w, h) in faces]
    idx = np.argmax(areas)
    x, y, w, h = faces[idx]

    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    # Calculate padded coordinates with bounds checking
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    # Crop face region
    face_crop = image[y1:y2, x1:x2]

    return face_crop


def process_single_image(args: tuple) -> FaceExtractionResult:
    """
    Process a single image: detect face, crop, and save.

    Args:
        args: Tuple of (input_path, output_path, padding)

    Returns:
        FaceExtractionResult
    """
    input_path, output_path, padding = args

    try:
        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            return FaceExtractionResult(
                input_path=str(input_path),
                output_path=None,
                success=False,
                face_found=False,
                error="Failed to load image"
            )

        # Detect and crop face
        detector = load_detector()
        face_crop = detect_and_crop_face(image, detector, padding=padding)

        if face_crop is None:
            # No face found - copy original image
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            return FaceExtractionResult(
                input_path=str(input_path),
                output_path=str(output_path),
                success=True,
                face_found=False
            )

        # Save cropped face
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), face_crop)

        return FaceExtractionResult(
            input_path=str(input_path),
            output_path=str(output_path),
            success=True,
            face_found=True
        )

    except Exception as e:
        return FaceExtractionResult(
            input_path=str(input_path),
            output_path=None,
            success=False,
            face_found=False,
            error=str(e)
        )


def get_all_images(input_dir: Path) -> list[tuple[Path, str, str]]:
    """
    Get all images with their relative paths and scale labels.

    Returns:
        List of (image_path, split, scale) tuples
    """
    images = []
    for split in ['train', 'val', 'test']:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue
        for scale_dir in split_dir.iterdir():
            if not scale_dir.is_dir():
                continue
            for img_path in scale_dir.glob('*.jpg'):
                images.append((img_path, split, scale_dir.name))
    return images


def preprocess_faces(
    input_dir: Path,
    output_dir: Path,
    num_workers: int = 12,
    padding: float = 0.2,
    limit: int = None
):
    """
    Run face extraction on entire dataset.

    Args:
        input_dir: Input dataset directory (ccv2_balanced)
        output_dir: Output directory for cropped faces
        num_workers: Number of parallel workers
        padding: Padding around detected face
        limit: Limit number of images to process (for testing)
    """
    print("=" * 60)
    print("FACE EXTRACTION PREPROCESSING")
    print("=" * 60)

    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {num_workers}")
    print(f"Face padding: {padding}")
    if limit:
        print(f"Limit: {limit} images (TEST MODE)")

    # Get all images
    print("\nScanning for images...")
    all_images = get_all_images(input_dir)
    print(f"Found {len(all_images):,} images")

    # Apply limit if specified
    if limit and limit < len(all_images):
        import random
        random.seed(42)
        all_images = random.sample(all_images, limit)
        print(f"Limited to {len(all_images)} images for testing")

    # Prepare arguments for parallel processing
    process_args = []
    for img_path, split, scale in all_images:
        output_path = output_dir / split / scale / img_path.name
        process_args.append((img_path, output_path, padding))

    # Process in parallel
    print(f"\nProcessing with {num_workers} workers...")
    start_time = time.time()

    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, process_args),
            total=len(process_args),
            desc="Extracting faces"
        ))

    elapsed = time.time() - start_time

    # Compute statistics
    total = len(results)
    successful = sum(1 for r in results if r.success)
    faces_found = sum(1 for r in results if r.face_found)
    failed = sum(1 for r in results if not r.success)

    # Statistics by split
    split_stats = {}
    for (_, split, scale), result in zip(all_images, results):
        if split not in split_stats:
            split_stats[split] = {'total': 0, 'faces': 0, 'no_face': 0, 'failed': 0}
        split_stats[split]['total'] += 1
        if result.face_found:
            split_stats[split]['faces'] += 1
        elif result.success:
            split_stats[split]['no_face'] += 1
        else:
            split_stats[split]['failed'] += 1

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Images per second: {total/elapsed:.1f}")
    print(f"\nOverall statistics:")
    print(f"  Total images: {total:,}")
    print(f"  Successful: {successful:,} ({100*successful/total:.1f}%)")
    print(f"  Faces found: {faces_found:,} ({100*faces_found/total:.1f}%)")
    print(f"  No face (kept original): {successful - faces_found:,} ({100*(successful-faces_found)/total:.1f}%)")
    print(f"  Failed: {failed:,} ({100*failed/total:.1f}%)")

    print(f"\nPer-split statistics:")
    for split, stats in split_stats.items():
        print(f"  {split}: {stats['faces']}/{stats['total']} faces ({100*stats['faces']/stats['total']:.1f}%)")

    # Save statistics to JSON
    stats_file = output_dir / "preprocessing_stats.json"
    stats = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'total_images': total,
        'successful': successful,
        'faces_found': faces_found,
        'no_face_kept_original': successful - faces_found,
        'failed': failed,
        'elapsed_seconds': elapsed,
        'num_workers': num_workers,
        'padding': padding,
        'split_stats': split_stats
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to: {stats_file}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Extract faces from dataset images")
    parser.add_argument(
        "--input",
        type=str,
        default="training/data/ccv2_balanced",
        help="Input dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/data/ccv2_faces",
        help="Output directory for cropped faces"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.2,
        help="Padding around face as fraction of face size"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)"
    )

    args = parser.parse_args()

    preprocess_faces(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        num_workers=args.workers,
        padding=args.padding,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
