"""
MediaPipe-based Skin Mask Preprocessing.

Uses MediaPipe Face Landmarker (478 landmarks) to create precise skin masks
from existing face crops. Excludes eyes, lips, and eyebrows for accurate
skin-only regions.

Usage:
    python preprocess_mediapipe.py --input training/data/ccv2_faces --output training/data/ccv2_faces_mediapipe --workers 8

Output per image:
    - {name}_masked.jpg : Skin-only image (background = black)
    - {name}_mask.png   : Binary mask (255=skin, 0=background)
"""

# Suppress TensorFlow/MediaPipe warnings (must be before imports)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import argparse
import json
import os
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# MediaPipe model URL and path
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def ensure_model_downloaded():
    """Download the face landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MediaPipe face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Downloaded to: {MODEL_PATH}")
    return MODEL_PATH


# MediaPipe face mesh landmark indices for skin regions
# See: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

LEFT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
    386, 385, 384, 398
]

RIGHT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
    159, 160, 161, 246
]

LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
    37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82,
    81, 42, 183, 78
]

# Left eyebrow
LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]

# Right eyebrow
RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]


def create_skin_mask_from_landmarks(h: int, w: int, landmarks: np.ndarray) -> np.ndarray:
    """
    Create skin mask using MediaPipe landmarks.
    Includes face oval, excludes eyes, lips, and eyebrows.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw face oval (filled)
    face_points = np.array([landmarks[i] for i in FACE_OVAL if i < len(landmarks)])
    if len(face_points) > 0:
        cv2.fillPoly(mask, [face_points], 255)

    # Exclude eyes
    for eye_indices in [LEFT_EYE, RIGHT_EYE]:
        eye_points = np.array([landmarks[i] for i in eye_indices if i < len(landmarks)])
        if len(eye_points) > 0:
            cv2.fillPoly(mask, [eye_points], 0)

    # Exclude eyebrows
    for brow_indices in [LEFT_EYEBROW, RIGHT_EYEBROW]:
        brow_points = np.array([landmarks[i] for i in brow_indices if i < len(landmarks)])
        if len(brow_points) > 0:
            # Eyebrows are lines, so we need to create a polygon by adding offset
            if len(brow_points) >= 2:
                # Create a thick line mask
                cv2.polylines(mask, [brow_points], False, 0, thickness=8)

    # Exclude lips
    lip_points = np.array([landmarks[i] for i in LIPS if i < len(landmarks)])
    if len(lip_points) > 0:
        cv2.fillPoly(mask, [lip_points], 0)

    return mask


def process_single_image_worker(args: tuple) -> dict:
    """
    Worker function for parallel processing.
    Imports MediaPipe inside to avoid pickling issues.
    """
    img_path, input_base, output_base, model_path = args

    # Suppress ALL warnings by redirecting stderr during import
    import os
    import sys
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_minloglevel'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Redirect stderr to suppress MediaPipe warnings
    stderr_backup = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except:
        pass

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
    except ImportError:
        sys.stderr = stderr_backup
        return {"path": str(img_path), "success": False, "error": "mediapipe not installed"}
    finally:
        # Restore stderr after imports
        sys.stderr = stderr_backup

    # Compute paths
    rel_path = img_path.relative_to(input_base)
    output_dir = output_base / rel_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    masked_out = output_dir / f"{stem}_masked.jpg"
    mask_out = output_dir / f"{stem}_mask.png"

    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        return {"path": str(rel_path), "success": False, "error": "Could not load image"}

    h, w = image.shape[:2]

    # Create face landmarker (suppress warnings during creation)
    try:
        # Suppress stderr during model loading
        stderr_backup = sys.stderr
        sys.stderr = open(os.devnull, 'w')

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        landmarker = vision.FaceLandmarker.create_from_options(options)
        sys.stderr = stderr_backup

        with landmarker:
            # Convert to RGB and create MediaPipe Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Detect
            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                return {"path": str(rel_path), "success": False, "error": "No face detected"}

            # Extract landmarks
            face_landmarks = result.face_landmarks[0]
            points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks])

            # Create mask
            mask = create_skin_mask_from_landmarks(h, w, points)

    except Exception as e:
        return {"path": str(rel_path), "success": False, "error": str(e)}

    # Calculate skin ratio
    skin_ratio = np.sum(mask > 0) / mask.size

    # Create masked image
    masked_image = np.zeros_like(image)
    masked_image[mask > 0] = image[mask > 0]

    # Save outputs
    cv2.imwrite(str(masked_out), masked_image)
    cv2.imwrite(str(mask_out), mask)

    return {
        "path": str(rel_path),
        "success": True,
        "skin_ratio": skin_ratio,
        "num_landmarks": len(points),
    }


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    num_workers: int = 4,
    resume: bool = True,
) -> dict:
    """
    Preprocess entire dataset with MediaPipe face landmarks.
    """
    # Ensure model is downloaded first (in main process)
    model_path = ensure_model_downloaded()

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png"}
    all_images = []
    for ext in image_extensions:
        all_images.extend(input_path.rglob(f"*{ext}"))
        all_images.extend(input_path.rglob(f"*{ext.upper()}"))

    all_images = sorted(set(all_images))
    total_images = len(all_images)
    print(f"Found {total_images} images in {input_dir}")

    # Filter out already processed images if resuming
    if resume:
        print("Checking for already processed images...")
        images_to_process = []
        for img in tqdm(all_images, desc="Scanning", leave=False):
            rel_path = img.relative_to(input_path)
            output_dir_for_img = output_path / rel_path.parent
            mask_file = output_dir_for_img / f"{img.stem}_mask.png"
            if not mask_file.exists():
                images_to_process.append(img)

        already_done = total_images - len(images_to_process)
        if already_done > 0:
            print(f"Already processed: {already_done} images (skipping)")
            print(f"Remaining: {len(images_to_process)} images")
        all_images = images_to_process

    if len(all_images) == 0:
        if resume and already_done > 0:
            print("All images already processed!")
            return {"success": True, "already_complete": True}
        print("No images found!")
        return {"success": False, "error": "No images found"}

    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for workers
    work_args = [(img, input_path, output_path, model_path) for img in all_images]

    # Process with parallel workers
    results = []
    failed = []
    skin_ratios = []

    # Use ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image_worker, args): args[0] for args in work_args}

        for future in tqdm(as_completed(futures), total=len(all_images), desc="MediaPipe preprocessing"):
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
        "detection_rate": len(skin_ratios) / len(all_images) if all_images else 0,
        "skin_ratio_stats": {
            "mean": float(np.mean(skin_ratios)) if len(skin_ratios) > 0 else 0,
            "std": float(np.std(skin_ratios)) if len(skin_ratios) > 0 else 0,
            "min": float(np.min(skin_ratios)) if len(skin_ratios) > 0 else 0,
            "max": float(np.max(skin_ratios)) if len(skin_ratios) > 0 else 0,
            "median": float(np.median(skin_ratios)) if len(skin_ratios) > 0 else 0,
        },
        "failed_images": [f["path"] for f in failed[:100]],  # Limit to first 100
    }

    # Save stats
    stats_file = output_path / "preprocessing_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("MEDIAPIPE PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"\nProcessed: {stats['successful']}/{stats['total_images']} images")
    print(f"Detection rate: {stats['detection_rate']:.1%}")
    if stats['failed'] > 0:
        print(f"Failed: {stats['failed']} images (no face detected)")
    print(f"\nSkin Ratio Statistics:")
    print(f"  Mean:   {stats['skin_ratio_stats']['mean']:.1%}")
    print(f"  Std:    {stats['skin_ratio_stats']['std']:.1%}")
    print(f"  Range:  [{stats['skin_ratio_stats']['min']:.1%}, {stats['skin_ratio_stats']['max']:.1%}]")
    print(f"\nOutput files per image:")
    print(f"  - <name>_masked.jpg : Skin only (background=black)")
    print(f"  - <name>_mask.png   : Binary mask")
    print(f"\nStats saved to: {stats_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess face images with MediaPipe for skin mask extraction"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="training/data/ccv2_faces",
        help="Input dataset directory (existing face crops)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/data/ccv2_faces_mediapipe",
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, MediaPipe is CPU-heavy)",
    )

    args = parser.parse_args()

    preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
