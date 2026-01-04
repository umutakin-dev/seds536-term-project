"""
Compare Face Detection Methods: Haar Cascade vs MediaPipe.

This script compares detection quality and skin region extraction
to determine if better face detection can improve ITA accuracy.

Usage:
    python compare_face_detectors.py <image_path>
    python compare_face_detectors.py --batch training/data/ccv2_faces/test --max-images 20

Output:
    - Side-by-side comparison image
    - Skin mask comparison
    - Statistics on detected regions
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# MediaPipe imports (Tasks API - v0.10+)
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not installed. Run: uv sync")

# Path to MediaPipe model files (will be downloaded on first use)
import urllib.request
import os

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

def ensure_model_downloaded():
    """Download the face landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MediaPipe face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Downloaded to: {MODEL_PATH}")
    return MODEL_PATH


def detect_face_haar(image_bgr: np.ndarray) -> dict:
    """Detect face using OpenCV Haar Cascade."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        return {"success": False, "method": "haar"}

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return {"success": True, "method": "haar", "bbox": (x, y, w, h), "landmarks": None}


def detect_face_mediapipe(image_bgr: np.ndarray) -> dict:
    """Detect face using MediaPipe Face Landmarker (Tasks API) with 478 landmarks."""
    if not MEDIAPIPE_AVAILABLE:
        return {"success": False, "method": "mediapipe", "error": "not installed"}

    # Ensure model is downloaded
    model_path = ensure_model_downloaded()

    # Create face landmarker
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

    h, w = image_bgr.shape[:2]

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        # Convert to RGB and create MediaPipe Image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            return {"success": False, "method": "mediapipe"}

        # Extract landmarks (first face)
        face_landmarks = result.face_landmarks[0]
        points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks])

        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        return {
            "success": True,
            "method": "mediapipe",
            "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
            "landmarks": points,
            "num_landmarks": len(points),
        }


# MediaPipe face mesh landmark indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
        14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
        312, 13, 82, 81, 42, 183, 78]

# Forehead region (upper face)
FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
            378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109, 151, 337, 299, 333, 298, 301, 368, 264, 447,
            366, 401, 435, 288, 367, 364, 394, 395, 369, 396, 175, 171, 140, 170, 169,
            135, 138, 215, 177, 137, 227, 34, 139, 71, 68, 104, 69, 108, 151]

# Left cheek
LEFT_CHEEK = [266, 426, 436, 432, 434, 430, 431, 262, 428, 199, 208, 32, 211, 210,
              214, 192, 213, 147, 123, 116, 117, 118, 119, 120, 121, 128, 245, 193,
              168, 417, 351, 419, 248, 281, 275, 274, 271, 272, 407, 408, 409, 410,
              411, 412, 413, 414, 415]

# Right cheek
RIGHT_CHEEK = [36, 206, 216, 212, 202, 210, 211, 32, 208, 199, 428, 262, 431, 430,
               434, 432, 436, 426, 266, 330, 329, 277, 343, 412, 399, 437, 355, 371,
               266, 425, 280, 346, 347, 348, 349, 350, 357, 465, 413, 168, 193, 245,
               128, 121, 120, 119, 118, 117, 116, 123, 147]


def create_skin_mask_mediapipe(image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Create precise skin mask using MediaPipe landmarks (excludes eyes, lips)."""
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw face oval
    face_points = np.array([landmarks[i] for i in FACE_OVAL if i < len(landmarks)])
    if len(face_points) > 0:
        cv2.fillPoly(mask, [face_points], 255)

    # Exclude eyes and lips
    for indices in [LEFT_EYE, RIGHT_EYE, LIPS]:
        points = np.array([landmarks[i] for i in indices if i < len(landmarks)])
        if len(points) > 0:
            cv2.fillPoly(mask, [points], 0)

    return mask


def create_skin_mask_haar(image_bgr: np.ndarray, bbox: tuple) -> np.ndarray:
    """Create skin mask using Haar bbox + oval + color (current method)."""
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = bbox

    # Oval mask
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (x + bw // 2, y + bh // 2)
    axes = (int(bw * 0.35), int(bh * 0.42))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # YCbCr skin filter
    ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = ((ycbcr[:, :, 2] >= 77) & (ycbcr[:, :, 2] <= 127) &
                 (ycbcr[:, :, 1] >= 133) & (ycbcr[:, :, 1] <= 173)).astype(np.uint8) * 255

    return cv2.bitwise_and(mask, skin_mask)


def calculate_ita(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Calculate ITA from masked skin region."""
    if mask is None or np.sum(mask > 0) < 100:
        return None

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    skin_pixels = lab[mask > 0]

    L = np.mean(skin_pixels[:, 0]) * 100.0 / 255.0
    b = np.mean(skin_pixels[:, 2]) - 128.0

    if abs(b) < 0.001:
        b = 0.001

    ita = np.arctan((L - 50) / b) * (180 / np.pi)
    return float(ita)


def visualize_comparison(image_bgr: np.ndarray, haar_result: dict, mp_result: dict,
                         output_path: str = None) -> np.ndarray:
    """Create side-by-side visualization."""
    h, w = image_bgr.shape[:2]

    # Create masks
    if haar_result["success"]:
        mask_haar = create_skin_mask_haar(image_bgr, haar_result["bbox"])
        ita_haar = calculate_ita(image_bgr, mask_haar)
    else:
        mask_haar = np.zeros((h, w), dtype=np.uint8)
        ita_haar = None

    if mp_result["success"] and mp_result.get("landmarks") is not None:
        mask_mp = create_skin_mask_mediapipe(image_bgr, mp_result["landmarks"])
        ita_mp = calculate_ita(image_bgr, mask_mp)
    else:
        mask_mp = np.zeros((h, w), dtype=np.uint8)
        ita_mp = None

    # Create visualization images
    img_haar = image_bgr.copy()
    img_mp = image_bgr.copy()

    # Draw Haar bbox
    if haar_result["success"]:
        x, y, bw, bh = haar_result["bbox"]
        cv2.rectangle(img_haar, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    # Draw MediaPipe landmarks
    if mp_result["success"] and mp_result.get("landmarks") is not None:
        for i, (px, py) in enumerate(mp_result["landmarks"]):
            if i % 5 == 0:
                cv2.circle(img_mp, (px, py), 1, (255, 0, 255), -1)
        # Draw face oval
        face_points = np.array([mp_result["landmarks"][i] for i in FACE_OVAL
                                if i < len(mp_result["landmarks"])])
        cv2.polylines(img_mp, [face_points], True, (0, 255, 0), 2)

    # Masked images
    masked_haar = np.zeros_like(image_bgr)
    masked_haar[mask_haar > 0] = image_bgr[mask_haar > 0]

    masked_mp = np.zeros_like(image_bgr)
    masked_mp[mask_mp > 0] = image_bgr[mask_mp > 0]

    # Add text
    def add_label(img, text, pos=(10, 30)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    add_label(img_haar, "Haar Cascade")
    add_label(img_mp, f"MediaPipe ({mp_result.get('num_landmarks', 0)} pts)")

    skin_ratio_haar = np.sum(mask_haar > 0) / mask_haar.size
    skin_ratio_mp = np.sum(mask_mp > 0) / mask_mp.size

    add_label(masked_haar, f"Skin: {skin_ratio_haar:.1%}", (10, 30))
    if ita_haar is not None:
        add_label(masked_haar, f"ITA: {ita_haar:.1f}", (10, 50))

    add_label(masked_mp, f"Skin: {skin_ratio_mp:.1%}", (10, 30))
    if ita_mp is not None:
        add_label(masked_mp, f"ITA: {ita_mp:.1f}", (10, 50))

    # Combine
    row1 = np.hstack([image_bgr, img_haar, masked_haar])
    row2 = np.hstack([np.zeros_like(image_bgr), img_mp, masked_mp])
    add_label(row2, "Original (top row)", (10, 30))

    vis = np.vstack([row1, row2])

    # Resize if too large
    max_width = 1200
    if vis.shape[1] > max_width:
        scale = max_width / vis.shape[1]
        vis = cv2.resize(vis, None, fx=scale, fy=scale)

    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"Saved: {output_path}")

    return vis, {"haar_ita": ita_haar, "mp_ita": ita_mp,
                 "haar_skin": skin_ratio_haar, "mp_skin": skin_ratio_mp}


def compare_single(image_path: str, output_dir: str = None) -> dict:
    """Compare detectors on a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return None

    print(f"\n{'='*50}")
    print(f"Image: {image_path}")
    print(f"Size: {image.shape[1]}x{image.shape[0]}")

    haar = detect_face_haar(image)
    mp = detect_face_mediapipe(image)

    print(f"\nHaar Cascade: {'OK' if haar['success'] else 'FAILED'}")
    print(f"MediaPipe:    {'OK' if mp['success'] else 'FAILED'}", end="")
    if mp["success"]:
        print(f" ({mp.get('num_landmarks', 0)} landmarks)")
    else:
        print()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(output_dir) / f"compare_{Path(image_path).stem}.jpg"
    else:
        out_path = "face_detection_comparison.jpg"

    vis, stats = visualize_comparison(image, haar, mp, str(out_path))

    print(f"\nSkin ratio - Haar: {stats['haar_skin']:.1%}, MediaPipe: {stats['mp_skin']:.1%}")
    if stats['haar_ita'] and stats['mp_ita']:
        print(f"ITA        - Haar: {stats['haar_ita']:.1f}°, MediaPipe: {stats['mp_ita']:.1f}°")

    return {"haar": haar, "mp": mp, "stats": stats}


def compare_batch(input_dir: str, output_dir: str, max_images: int = 20, random_sample: bool = True):
    """Compare detectors on multiple images."""
    all_images = list(Path(input_dir).rglob("*.jpg"))

    if random_sample and len(all_images) > max_images:
        # Try to sample across different subdirectories (scales/subjects)
        by_subdir = {}
        for img in all_images:
            subdir = img.parent.name
            if subdir not in by_subdir:
                by_subdir[subdir] = []
            by_subdir[subdir].append(img)

        # Sample evenly across subdirs
        images = []
        subdirs = list(by_subdir.keys())
        random.shuffle(subdirs)

        per_subdir = max(1, max_images // len(subdirs))
        for subdir in subdirs:
            subdir_images = by_subdir[subdir]
            random.shuffle(subdir_images)
            images.extend(subdir_images[:per_subdir])
            if len(images) >= max_images:
                break

        images = images[:max_images]
        print(f"Randomly sampled {len(images)} images across {len(set(img.parent.name for img in images))} subdirs\n")
    else:
        images = all_images[:max_images]
        print(f"Comparing {len(images)} images from {input_dir}\n")

    haar_ok, mp_ok = 0, 0
    ita_diffs = []

    for img_path in images:
        result = compare_single(str(img_path), output_dir)
        if result:
            if result["haar"]["success"]:
                haar_ok += 1
            if result["mp"]["success"]:
                mp_ok += 1
            if result["stats"]["haar_ita"] and result["stats"]["mp_ita"]:
                ita_diffs.append(abs(result["stats"]["haar_ita"] - result["stats"]["mp_ita"]))

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Haar Cascade: {haar_ok}/{len(images)} ({100*haar_ok/len(images):.0f}%)")
    print(f"MediaPipe:    {mp_ok}/{len(images)} ({100*mp_ok/len(images):.0f}%)")
    if ita_diffs:
        print(f"Mean ITA difference: {np.mean(ita_diffs):.1f}° (when both succeed)")


def main():
    parser = argparse.ArgumentParser(description="Compare Haar vs MediaPipe face detection")
    parser.add_argument("input", nargs="?", help="Image path or directory")
    parser.add_argument("--batch", action="store_true", help="Batch mode on directory")
    parser.add_argument("--output", default="training/data/face_compare_output", help="Output directory")
    parser.add_argument("--max-images", type=int, default=20, help="Max images in batch mode")

    args = parser.parse_args()

    if not MEDIAPIPE_AVAILABLE:
        print("Error: mediapipe not installed. Run: uv sync")
        sys.exit(1)

    if not args.input:
        # Default: test on a sample image
        test_dir = Path("training/data/ccv2_faces/test")
        if test_dir.exists():
            sample = next(test_dir.rglob("*.jpg"), None)
            if sample:
                args.input = str(sample)
                args.batch = False
            else:
                print("No test images found. Provide an image path.")
                sys.exit(1)
        else:
            print("No input provided. Usage: python compare_face_detectors.py <image_or_dir>")
            sys.exit(1)

    if args.batch or Path(args.input).is_dir():
        compare_batch(args.input, args.output, args.max_images)
    else:
        compare_single(args.input, args.output)


if __name__ == "__main__":
    main()
