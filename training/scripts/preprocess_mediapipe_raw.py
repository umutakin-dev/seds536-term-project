"""
MediaPipe Face Detection on Raw Frames (Optimized).

Runs MediaPipe directly on raw video frames (not Haar crops) to:
1. Detect faces with full image context
2. Crop face regions
3. Create skin masks using 478 landmarks

Usage:
    uv run python training/scripts/preprocess_mediapipe_raw.py --workers 12
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import argparse
import json
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# MediaPipe model
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Face mesh landmark indices
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]

# Global worker state (initialized once per worker)
_worker_landmarker = None
_worker_config = None


def ensure_model_downloaded():
    """Download the face landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MediaPipe face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Downloaded to: {MODEL_PATH}")
    return MODEL_PATH


def init_worker(model_path, config):
    """Initialize worker with MediaPipe model (called once per worker)."""
    global _worker_landmarker, _worker_config

    import os
    import sys
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_minloglevel'] = '2'

    # Suppress stderr during import
    stderr_backup = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        _worker_landmarker = vision.FaceLandmarker.create_from_options(options)
        _worker_config = config
    finally:
        sys.stderr = stderr_backup


def create_skin_mask(h: int, w: int, landmarks: np.ndarray) -> np.ndarray:
    """Create skin mask from landmarks, excluding eyes/lips/eyebrows."""
    mask = np.zeros((h, w), dtype=np.uint8)

    face_points = np.array([landmarks[i] for i in FACE_OVAL if i < len(landmarks)])
    if len(face_points) > 0:
        cv2.fillPoly(mask, [face_points], 255)

    for eye_indices in [LEFT_EYE, RIGHT_EYE]:
        eye_points = np.array([landmarks[i] for i in eye_indices if i < len(landmarks)])
        if len(eye_points) > 0:
            cv2.fillPoly(mask, [eye_points], 0)

    for brow_indices in [LEFT_EYEBROW, RIGHT_EYEBROW]:
        brow_points = np.array([landmarks[i] for i in brow_indices if i < len(landmarks)])
        if len(brow_points) >= 2:
            cv2.polylines(mask, [brow_points], False, 0, thickness=8)

    lip_points = np.array([landmarks[i] for i in LIPS if i < len(landmarks)])
    if len(lip_points) > 0:
        cv2.fillPoly(mask, [lip_points], 0)

    return mask


def process_single_image(args: tuple) -> dict:
    """Process a single raw frame using the pre-initialized landmarker."""
    global _worker_landmarker, _worker_config

    img_path, input_base, output_base = args
    crop_size = _worker_config['crop_size']
    padding = _worker_config['padding']

    import mediapipe as mp

    # Output paths
    rel_path = img_path.relative_to(input_base)
    output_dir = output_base / rel_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    face_out = output_dir / f"{stem}.jpg"
    masked_out = output_dir / f"{stem}_masked.jpg"
    mask_out = output_dir / f"{stem}_mask.png"

    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        return {"path": str(rel_path), "success": False, "error": "Could not load"}

    img_h, img_w = image.shape[:2]

    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = _worker_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return {"path": str(rel_path), "success": False, "error": "No face"}

        # Get landmarks
        face_landmarks = result.face_landmarks[0]
        points = np.array([(int(lm.x * img_w), int(lm.y * img_h)) for lm in face_landmarks])

        # Bounding box from landmarks
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        # Add padding
        face_w, face_h = x_max - x_min, y_max - y_min
        pad_x, pad_y = int(face_w * padding), int(face_h * padding)

        x_min = max(0, x_min - pad_x)
        x_max = min(img_w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(img_h, y_max + pad_y)

        # Crop and resize
        face_crop = image[y_min:y_max, x_min:x_max]
        face_crop = cv2.resize(face_crop, (crop_size, crop_size))

        # Adjust landmarks
        crop_w, crop_h = x_max - x_min, y_max - y_min
        scale_x, scale_y = crop_size / crop_w, crop_size / crop_h
        adjusted_points = np.array([
            (int((p[0] - x_min) * scale_x), int((p[1] - y_min) * scale_y))
            for p in points
        ])

        # Create mask
        mask = create_skin_mask(crop_size, crop_size, adjusted_points)
        skin_ratio = np.sum(mask > 0) / mask.size

        # Masked image
        masked_image = np.zeros_like(face_crop)
        masked_image[mask > 0] = face_crop[mask > 0]

        # Save
        cv2.imwrite(str(face_out), face_crop)
        cv2.imwrite(str(masked_out), masked_image)
        cv2.imwrite(str(mask_out), mask)

        return {"path": str(rel_path), "success": True, "skin_ratio": skin_ratio}

    except Exception as e:
        return {"path": str(rel_path), "success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="MediaPipe preprocessing on raw frames")
    parser.add_argument("--input", type=str, default="training/data/ccv2_balanced",
                        help="Input directory with raw frames")
    parser.add_argument("--output", type=str, default="training/data/ccv2_mediapipe_raw",
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--crop-size", type=int, default=224,
                        help="Output face crop size (default: 224)")
    parser.add_argument("--padding", type=float, default=0.2,
                        help="Padding around face (default: 0.2)")

    args = parser.parse_args()

    model_path = ensure_model_downloaded()
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Find images
    all_images = []
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        all_images.extend(input_path.rglob(f"*{ext}"))
    all_images = sorted(set(all_images))

    print(f"Found {len(all_images)} images")
    print(f"Output: {args.output}")
    print(f"Workers: {args.workers}")
    print(f"Crop size: {args.crop_size}x{args.crop_size}")

    output_path.mkdir(parents=True, exist_ok=True)

    config = {'crop_size': args.crop_size, 'padding': args.padding}
    work_args = [(img, input_path, output_path) for img in all_images]

    results = []
    failed = []
    skin_ratios = []

    # Use Pool with initializer (model loaded once per worker)
    with Pool(processes=args.workers, initializer=init_worker, initargs=(model_path, config)) as pool:
        for result in tqdm(pool.imap_unordered(process_single_image, work_args),
                          total=len(all_images), desc="Processing"):
            results.append(result)
            if result["success"]:
                skin_ratios.append(result["skin_ratio"])
            else:
                failed.append(result)

    # Stats
    skin_ratios = np.array(skin_ratios) if skin_ratios else np.array([0])
    stats = {
        "input_dir": str(args.input),
        "output_dir": str(args.output),
        "total_images": len(all_images),
        "successful": len(skin_ratios),
        "failed": len(failed),
        "detection_rate": len(skin_ratios) / len(all_images) if all_images else 0,
        "crop_size": args.crop_size,
        "skin_ratio_stats": {
            "mean": float(np.mean(skin_ratios)),
            "std": float(np.std(skin_ratios)),
        },
        "failed_images": [f["path"] for f in failed[:100]],
    }

    stats_file = output_path / "preprocessing_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Processed: {stats['successful']}/{stats['total_images']} ({stats['detection_rate']:.1%})")
    print(f"Failed: {stats['failed']}")
    print(f"Skin ratio: {stats['skin_ratio_stats']['mean']:.1%}")
    print(f"Stats: {stats_file}")


if __name__ == "__main__":
    main()
