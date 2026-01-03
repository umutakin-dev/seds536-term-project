"""
Quick benchmark to estimate face detection time on the dataset.
Tests on a sample of images and extrapolates to full dataset.
Supports parallel processing for faster execution.
"""

import time
from pathlib import Path
import random
import multiprocessing as mp
from functools import partial
import os

import cv2
import numpy as np
from tqdm import tqdm


def load_opencv_detector():
    """Load OpenCV's Haar cascade face detector."""
    model_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(model_file)
    return detector


def detect_face_haar(image):
    """Detect face using Haar cascades."""
    detector = load_opencv_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        # Return largest face
        areas = [w * h for (x, y, w, h) in faces]
        idx = np.argmax(areas)
        x, y, w, h = faces[idx]
        return (x, y, x + w, y + h)
    return None


def process_single_image(img_path: Path) -> tuple[bool, bool]:
    """
    Process a single image for face detection.
    Returns: (success, face_found)
    """
    try:
        image = cv2.imread(str(img_path))
        if image is None:
            return (False, False)
        face = detect_face_haar(image)
        return (True, face is not None)
    except Exception:
        return (False, False)


def get_sample_images(data_dir: Path, sample_size: int = 200) -> list[Path]:
    """Get a random sample of images from the dataset."""
    all_images = list(data_dir.rglob("*.jpg"))
    if len(all_images) <= sample_size:
        return all_images
    return random.sample(all_images, sample_size)


def benchmark_single_threaded(sample_images: list[Path]) -> tuple[float, int]:
    """Benchmark single-threaded face detection."""
    faces_found = 0
    start_time = time.time()

    for img_path in tqdm(sample_images, desc="Single-threaded"):
        success, found = process_single_image(img_path)
        if found:
            faces_found += 1

    elapsed = time.time() - start_time
    return elapsed, faces_found


def benchmark_parallel(sample_images: list[Path], num_workers: int) -> tuple[float, int]:
    """Benchmark parallel face detection."""
    start_time = time.time()

    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, sample_images),
            total=len(sample_images),
            desc=f"Parallel ({num_workers} workers)"
        ))

    elapsed = time.time() - start_time
    faces_found = sum(1 for success, found in results if found)
    return elapsed, faces_found


def main():
    print("=" * 60)
    print("FACE DETECTION BENCHMARK (Single vs Parallel)")
    print("=" * 60)

    # System info
    cpu_count = mp.cpu_count()
    print(f"\nCPU cores available: {cpu_count}")

    data_dir = Path("training/data/ccv2_balanced/train")
    total_images = len(list(data_dir.rglob("*.jpg")))

    print(f"Total images in training set: {total_images:,}")

    # Sample size for benchmark
    sample_size = 500  # Larger sample for more accurate estimate
    print(f"Sample size for benchmark: {sample_size}")

    # Get sample images
    print("\nGetting sample images...")
    sample_images = get_sample_images(data_dir, sample_size)
    print(f"Got {len(sample_images)} sample images")

    # Benchmark single-threaded
    print(f"\n--- Single-Threaded Benchmark ---")
    single_time, single_faces = benchmark_single_threaded(sample_images)
    single_per_image = single_time / len(sample_images)
    single_total_est = single_per_image * total_images

    print(f"\nResults (Single-threaded):")
    print(f"  Time for {len(sample_images)} images: {single_time:.2f}s")
    print(f"  Time per image: {single_per_image*1000:.1f}ms")
    print(f"  Faces found: {single_faces}/{len(sample_images)} ({100*single_faces/len(sample_images):.1f}%)")
    print(f"  Estimated total time: {single_total_est/60:.1f} minutes")

    # Benchmark parallel with different worker counts
    worker_counts = [cpu_count // 2, cpu_count, cpu_count + 4]
    worker_counts = [w for w in worker_counts if w > 1]

    best_time = single_time
    best_workers = 1

    for num_workers in worker_counts:
        print(f"\n--- Parallel Benchmark ({num_workers} workers) ---")
        parallel_time, parallel_faces = benchmark_parallel(sample_images, num_workers)
        parallel_per_image = parallel_time / len(sample_images)
        parallel_total_est = parallel_per_image * total_images
        speedup = single_time / parallel_time

        print(f"\nResults ({num_workers} workers):")
        print(f"  Time for {len(sample_images)} images: {parallel_time:.2f}s")
        print(f"  Time per image: {parallel_per_image*1000:.1f}ms")
        print(f"  Faces found: {parallel_faces}/{len(sample_images)} ({100*parallel_faces/len(sample_images):.1f}%)")
        print(f"  Estimated total time: {parallel_total_est/60:.1f} minutes")
        print(f"  Speedup vs single-threaded: {speedup:.1f}x")

        if parallel_time < best_time:
            best_time = parallel_time
            best_workers = num_workers

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_per_image = best_time / len(sample_images)
    best_total_est = best_per_image * total_images
    total_dataset = total_images * 3  # train + val + test (roughly)

    print(f"\nBest configuration: {best_workers} workers")
    print(f"Estimated time for training set ({total_images:,} images): {best_total_est/60:.1f} minutes")
    print(f"Estimated time for full dataset (~{total_dataset:,} images): {best_total_est*3/60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
