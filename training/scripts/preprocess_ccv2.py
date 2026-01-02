"""
Preprocessing script for Casual Conversations v2 dataset.
Prepares frames for Monk skin tone classification training.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import random
import shutil

import pandas as pd
from tqdm import tqdm


# Configuration
ANNOTATIONS_PATH = Path('C:/Users/ydran/Downloads/ccv2-annotations/CasualConversationsV2.json')
FRAMES_DIRS = [
    Path(f'C:/Users/ydran/Downloads/ccv2-frames-part-{i}') for i in range(1, 6)
]
OUTPUT_DIR = Path('C:/Users/ydran/workspace/seds/seds536-image-understanding/seds536-term-project/training/data/ccv2_balanced')


def load_annotations(annotations_path: Path) -> list[dict]:
    """Load the CCv2 annotations JSON file."""
    with open(annotations_path, 'r') as f:
        return json.load(f)


def build_subject_info(annotations: list[dict]) -> dict:
    """Build a mapping of subject_id -> info (monk scale, videos, etc.)"""
    subjects = {}
    for ann in annotations:
        sid = ann['subject_id']
        if sid not in subjects:
            subjects[sid] = {
                'monk_scale': int(ann['monk_skin_tone']['scale'].split()[-1]),
                'monk_confidence': ann['monk_skin_tone']['confidence'],
                'videos': []
            }
        subjects[sid]['videos'].append(ann['video_name'])
    return subjects


def find_subject_frames(subject_id: str, frames_dirs: list[Path]) -> list[Path]:
    """Find all frame files for a subject across all frame directories."""
    frames = []
    for frames_dir in frames_dirs:
        subject_dir = frames_dir / subject_id
        if subject_dir.exists():
            frames.extend(sorted(subject_dir.glob('*.jpg')))
    return frames


def create_balanced_dataset(
    subjects: dict,
    frames_dirs: list[Path],
    output_dir: Path,
    max_frames_per_subject: int = 30,
    confidence_filter: list[str] = ['high', 'medium'],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Create a balanced dataset with train/val/test splits.

    Strategy for imbalanced classes:
    - Use all subjects from minority classes (scales 1, 9, 10)
    - Sample subjects from majority classes to balance
    - Split by subject to avoid data leakage
    """
    # Filter by confidence
    filtered_subjects = {
        sid: info for sid, info in subjects.items()
        if info['monk_confidence'] in confidence_filter
    }
    print(f"Subjects after confidence filter: {len(filtered_subjects)}")

    # Group subjects by Monk scale
    scale_to_subjects = defaultdict(list)
    for sid, info in filtered_subjects.items():
        scale_to_subjects[info['monk_scale']].append(sid)

    # Print distribution
    print("\nSubjects per scale (after filter):")
    for scale in range(1, 11):
        print(f"  Scale {scale}: {len(scale_to_subjects[scale])}")

    # Determine target samples per class (use minimum class size as baseline)
    min_class_size = min(len(subs) for subs in scale_to_subjects.values())
    print(f"\nMinimum class size: {min_class_size}")

    # Create output directories
    for split in ['train', 'val', 'test']:
        for scale in range(1, 11):
            (output_dir / split / f'scale_{scale}').mkdir(parents=True, exist_ok=True)

    # Process each scale
    stats = defaultdict(lambda: defaultdict(int))

    for scale in range(1, 11):
        subject_list = scale_to_subjects[scale].copy()
        random.shuffle(subject_list)

        # Split subjects into train/val/test
        n_subjects = len(subject_list)
        n_train = max(1, int(n_subjects * train_ratio))
        n_val = max(1, int(n_subjects * val_ratio))

        train_subjects = subject_list[:n_train]
        val_subjects = subject_list[n_train:n_train + n_val]
        test_subjects = subject_list[n_train + n_val:]

        print(f"\nScale {scale}: {n_subjects} subjects -> train: {len(train_subjects)}, val: {len(val_subjects)}, test: {len(test_subjects)}")

        # Process each split
        for split, split_subjects in [('train', train_subjects), ('val', val_subjects), ('test', test_subjects)]:
            for sid in tqdm(split_subjects, desc=f'Scale {scale} {split}', leave=False):
                # Find all frames for this subject
                frames = find_subject_frames(sid, frames_dirs)

                if not frames:
                    continue

                # Sample frames (limit per subject to avoid one subject dominating)
                if len(frames) > max_frames_per_subject:
                    frames = random.sample(frames, max_frames_per_subject)

                # Copy frames
                for frame_path in frames:
                    output_path = output_dir / split / f'scale_{scale}' / frame_path.name
                    if not output_path.exists():
                        shutil.copy(frame_path, output_path)
                        stats[split][scale] += 1

    # Print final statistics
    print("\n" + "=" * 50)
    print("FINAL DATASET STATISTICS")
    print("=" * 50)

    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        total = 0
        for scale in range(1, 11):
            count = stats[split][scale]
            total += count
            print(f"  Scale {scale:2d}: {count:5d} frames")
        print(f"  {'Total':8s}: {total:5d} frames")


def main():
    random.seed(42)  # For reproducibility

    print("=" * 50)
    print("CCv2 Dataset Preprocessing")
    print("=" * 50)

    # Check frames directories exist
    for frames_dir in FRAMES_DIRS:
        if not frames_dir.exists():
            print(f"ERROR: Frames directory not found: {frames_dir}")
            return
    print(f"Found {len(FRAMES_DIRS)} frame directories")

    # Load annotations
    print("\nLoading annotations...")
    annotations = load_annotations(ANNOTATIONS_PATH)
    print(f"Loaded {len(annotations)} video annotations")

    # Build subject info
    print("\nBuilding subject info...")
    subjects = build_subject_info(annotations)
    print(f"Found {len(subjects)} unique subjects")

    # Create balanced dataset
    print("\nCreating balanced dataset...")
    create_balanced_dataset(
        subjects=subjects,
        frames_dirs=FRAMES_DIRS,
        output_dir=OUTPUT_DIR,
        max_frames_per_subject=30,
        confidence_filter=['high', 'medium'],
        train_ratio=0.7,
        val_ratio=0.15,
    )

    print("\n" + "=" * 50)
    print("DONE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == '__main__':
    main()
