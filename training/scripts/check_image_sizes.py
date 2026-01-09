"""Quick script to compare raw frames vs Haar crops."""
import cv2
from pathlib import Path

raw_dir = Path("training/data/ccv2_balanced")
haar_dir = Path("training/data/ccv2_faces")

# Find a sample image
raw_sample = list(raw_dir.glob("test/scale_1/*.jpg"))[0]
haar_sample = haar_dir / raw_sample.relative_to(raw_dir)

print(f"Raw frame: {raw_sample.name}")
raw_img = cv2.imread(str(raw_sample))
print(f"  Dimensions: {raw_img.shape[1]}x{raw_img.shape[0]} (WxH)")

print(f"\nHaar crop: {haar_sample.name}")
haar_img = cv2.imread(str(haar_sample))
print(f"  Dimensions: {haar_img.shape[1]}x{haar_img.shape[0]} (WxH)")

print(f"\nSize reduction: {raw_img.shape[0]*raw_img.shape[1]} -> {haar_img.shape[0]*haar_img.shape[1]} pixels")
print(f"  ({100*haar_img.shape[0]*haar_img.shape[1]/(raw_img.shape[0]*raw_img.shape[1]):.1f}% of original)")
