"""
Skin Segmentation using Classical Image Processing Techniques.

This module implements skin detection using color space thresholding
and morphological operations - classical computer vision techniques
from SEDS536 course.

Techniques demonstrated:
1. Color space conversion (RGB -> YCbCr)
2. Thresholding for segmentation
3. Morphological operations (erosion, dilation, opening, closing)

References:
- Chai & Ngan, "Face segmentation using skin-color map in videophone applications" (1999)
- Kovac et al., "Human Skin Color Clustering for Face Detection" (2003)
"""

import cv2
import numpy as np


# Default skin color thresholds in YCbCr space
# These values are based on empirical studies of human skin colors
# and work reasonably well across different skin tones
YCBCR_SKIN_THRESHOLDS = {
    "Y_min": 0,      # Luminance - accept any brightness
    "Y_max": 255,
    "Cb_min": 77,    # Blue-difference chroma
    "Cb_max": 127,
    "Cr_min": 133,   # Red-difference chroma
    "Cr_max": 173,
}

# Alternative thresholds (more permissive, may include more skin but also more noise)
YCBCR_SKIN_THRESHOLDS_PERMISSIVE = {
    "Y_min": 0,
    "Y_max": 255,
    "Cb_min": 70,
    "Cb_max": 135,
    "Cr_min": 125,
    "Cr_max": 180,
}


def segment_skin_ycbcr(
    image_bgr: np.ndarray,
    thresholds: dict = None,
) -> np.ndarray:
    """
    Segment skin regions using YCbCr color space thresholding.

    YCbCr separates luminance (Y) from chrominance (Cb, Cr), making
    skin detection more robust to lighting variations.

    Args:
        image_bgr: Input image in BGR format (OpenCV default)
        thresholds: Dict with Y_min, Y_max, Cb_min, Cb_max, Cr_min, Cr_max

    Returns:
        Binary mask where 255 = skin, 0 = non-skin
    """
    if thresholds is None:
        thresholds = YCBCR_SKIN_THRESHOLDS

    # Convert BGR to YCbCr
    image_ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

    # Note: OpenCV uses YCrCb order, not YCbCr
    Y = image_ycbcr[:, :, 0]
    Cr = image_ycbcr[:, :, 1]
    Cb = image_ycbcr[:, :, 2]

    # Apply thresholds
    mask = (
        (Y >= thresholds["Y_min"]) & (Y <= thresholds["Y_max"]) &
        (Cb >= thresholds["Cb_min"]) & (Cb <= thresholds["Cb_max"]) &
        (Cr >= thresholds["Cr_min"]) & (Cr <= thresholds["Cr_max"])
    )

    # Convert boolean mask to uint8 (0 or 255)
    return (mask * 255).astype(np.uint8)


def segment_skin_hsv(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Segment skin regions using HSV color space thresholding.

    Alternative to YCbCr - HSV can be useful for certain lighting conditions.

    Args:
        image_bgr: Input image in BGR format

    Returns:
        Binary mask where 255 = skin, 0 = non-skin
    """
    # Convert BGR to HSV
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Skin color in HSV (hue around 0-50, which is red-yellow range)
    # Note: OpenCV uses H: 0-180, S: 0-255, V: 0-255
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([50, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(image_hsv, lower, upper)

    return mask


def apply_morphological_cleanup(
    mask: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 2,
) -> np.ndarray:
    """
    Clean up a binary mask using morphological operations.

    Operations applied:
    1. Opening (erosion -> dilation): Removes small noise/isolated pixels
    2. Closing (dilation -> erosion): Fills small holes

    Args:
        mask: Binary mask (0 or 255)
        kernel_size: Size of the structuring element
        iterations: Number of times to apply each operation

    Returns:
        Cleaned binary mask
    """
    # Create structuring element (ellipse works well for face regions)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    # Opening: Remove small noise (small white regions become black)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # Closing: Fill small holes (small black regions become white)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return cleaned


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in the mask.

    This removes scattered noise and keeps only the main skin region (face).

    Args:
        mask: Binary mask (0 or 255)

    Returns:
        Mask with only the largest connected component
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        # No components found (only background)
        return mask

    # Find the largest component (excluding background which is label 0)
    # stats[:, cv2.CC_STAT_AREA] gives the area of each component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create mask with only the largest component
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255

    return largest_mask


def apply_face_roi(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    padding: float = 0.1,
) -> np.ndarray:
    """
    Restrict mask to face region of interest using simple heuristics.

    Uses the assumption that the face is roughly centered and takes up
    a significant portion of the image (since images are already face crops).

    Args:
        mask: Binary skin mask
        image_bgr: Original image (for size reference)
        padding: Padding around center region (0.1 = 10% on each side)

    Returns:
        Mask restricted to center ROI
    """
    h, w = mask.shape[:2]

    # Define center ROI (face crops are typically centered)
    x1 = int(w * padding)
    x2 = int(w * (1 - padding))
    y1 = int(h * padding)
    y2 = int(h * (1 - padding))

    # Create ROI mask
    roi_mask = np.zeros_like(mask)
    roi_mask[y1:y2, x1:x2] = 255

    # Apply ROI to skin mask
    return cv2.bitwise_and(mask, roi_mask)


def create_face_oval_mask(
    image_bgr: np.ndarray,
    width_ratio: float = 0.7,
    height_ratio: float = 0.85,
    center_y_offset: float = -0.05,
) -> np.ndarray:
    """
    Create an ellipse mask approximating a face oval.

    Assumes the face is roughly centered in the image (typical for face crops).
    The ellipse is sized relative to image dimensions.

    Args:
        image_bgr: Input image (for dimensions)
        width_ratio: Ellipse width as ratio of image width (0.7 = 70%)
        height_ratio: Ellipse height as ratio of image height (0.85 = 85%)
        center_y_offset: Vertical offset for ellipse center (-0.05 = 5% up)
                        Faces are often in upper portion of crops

    Returns:
        Binary mask with ellipse (255 inside, 0 outside)
    """
    h, w = image_bgr.shape[:2]

    # Ellipse center (slightly above image center for typical face crops)
    center_x = w // 2
    center_y = int(h * (0.5 + center_y_offset))

    # Ellipse axes (semi-major and semi-minor)
    axis_x = int(w * width_ratio / 2)
    axis_y = int(h * height_ratio / 2)

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw filled ellipse
    cv2.ellipse(
        mask,
        center=(center_x, center_y),
        axes=(axis_x, axis_y),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1  # Filled
    )

    return mask


def segment_skin_with_oval(
    image_bgr: np.ndarray,
    width_ratio: float = 0.7,
    height_ratio: float = 0.85,
    center_y_offset: float = -0.05,
    combine_with_color: bool = True,
    method: str = "ycbcr",
) -> np.ndarray:
    """
    Segment skin using face oval mask, optionally combined with color detection.

    Args:
        image_bgr: Input image in BGR format
        width_ratio: Ellipse width ratio
        height_ratio: Ellipse height ratio
        center_y_offset: Vertical offset for center
        combine_with_color: If True, intersect oval with color-based skin detection
                           If False, use oval only
        method: Color method if combining ("ycbcr" or "hsv")

    Returns:
        Binary mask where 255 = skin, 0 = non-skin
    """
    # Create oval mask
    oval_mask = create_face_oval_mask(
        image_bgr,
        width_ratio=width_ratio,
        height_ratio=height_ratio,
        center_y_offset=center_y_offset,
    )

    if combine_with_color:
        # Get color-based skin mask
        if method == "ycbcr":
            color_mask = segment_skin_ycbcr(image_bgr)
        else:
            color_mask = segment_skin_hsv(image_bgr)

        # Combine: only keep pixels that are both in oval AND detected as skin
        mask = cv2.bitwise_and(oval_mask, color_mask)
    else:
        # Use oval only (no color filtering)
        mask = oval_mask

    return mask


def segment_skin(
    image_bgr: np.ndarray,
    method: str = "ycbcr",
    apply_morphology: bool = True,
    kernel_size: int = 5,
    morph_iterations: int = 2,
    min_skin_ratio: float = 0.05,
    largest_component_only: bool = False,
    use_face_roi: bool = False,
    roi_padding: float = 0.1,
) -> np.ndarray:
    """
    Complete skin segmentation pipeline.

    Args:
        image_bgr: Input image in BGR format
        method: "ycbcr" or "hsv"
        apply_morphology: Whether to apply morphological cleanup
        kernel_size: Kernel size for morphological operations
        morph_iterations: Iterations for morphological operations
        min_skin_ratio: Minimum ratio of skin pixels required (0-1)
                       If below this, returns None (detection failed)
        largest_component_only: Keep only the largest connected component
        use_face_roi: Restrict to center ROI (assumes face is centered)
        roi_padding: Padding for face ROI (0.1 = 10% on each side)

    Returns:
        Binary mask where 255 = skin, 0 = non-skin
        Returns None if skin ratio is below min_skin_ratio
    """
    # Step 1: Color-based segmentation
    if method == "ycbcr":
        mask = segment_skin_ycbcr(image_bgr)
    elif method == "hsv":
        mask = segment_skin_hsv(image_bgr)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ycbcr' or 'hsv'")

    # Step 2: Morphological cleanup
    if apply_morphology:
        mask = apply_morphological_cleanup(
            mask,
            kernel_size=kernel_size,
            iterations=morph_iterations,
        )

    # Step 3: Keep largest connected component (removes scattered noise)
    if largest_component_only:
        mask = keep_largest_component(mask)

    # Step 4: Apply face ROI (removes edge/background detections)
    if use_face_roi:
        mask = apply_face_roi(mask, image_bgr, padding=roi_padding)

    # Step 5: Check if enough skin was detected
    skin_ratio = np.sum(mask > 0) / mask.size
    if skin_ratio < min_skin_ratio:
        return None

    return mask


def get_skin_statistics(
    image_bgr: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """
    Get statistics about detected skin region.

    Args:
        image_bgr: Original image
        mask: Skin mask

    Returns:
        Dictionary with skin statistics
    """
    total_pixels = mask.size
    skin_pixels = np.sum(mask > 0)
    skin_ratio = skin_pixels / total_pixels

    # Get mean color values in skin region
    if skin_pixels > 0:
        skin_region = image_bgr[mask > 0]
        mean_bgr = np.mean(skin_region, axis=0)

        # Convert to LAB for additional stats
        image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        skin_lab = image_lab[mask > 0]
        mean_lab = np.mean(skin_lab, axis=0)
        # Convert OpenCV LAB to standard LAB
        mean_L = mean_lab[0] * 100.0 / 255.0
        mean_a = mean_lab[1] - 128.0
        mean_b = mean_lab[2] - 128.0
    else:
        mean_bgr = [0, 0, 0]
        mean_L, mean_a, mean_b = 0, 0, 0

    return {
        "total_pixels": int(total_pixels),
        "skin_pixels": int(skin_pixels),
        "skin_ratio": float(skin_ratio),
        "mean_bgr": [float(x) for x in mean_bgr],
        "mean_L": float(mean_L),
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
    }


def visualize_segmentation(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    output_path: str = None,
) -> np.ndarray:
    """
    Create a visualization of the skin segmentation.

    Args:
        image_bgr: Original image
        mask: Skin mask
        output_path: Optional path to save visualization

    Returns:
        Visualization image (original | mask | masked)
    """
    # Resize for display if too large
    h, w = image_bgr.shape[:2]
    max_dim = 300
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image_bgr = cv2.resize(image_bgr, new_size)
        mask = cv2.resize(mask, new_size)

    # Convert mask to 3-channel for concatenation
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Create masked image (skin only)
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    # Concatenate: original | mask | masked
    vis = np.concatenate([image_bgr, mask_rgb, masked], axis=1)

    if output_path:
        cv2.imwrite(output_path, vis)

    return vis


if __name__ == "__main__":
    # Test the skin segmentation on a sample image
    import sys

    if len(sys.argv) < 2:
        print("Usage: python skin_segmentation.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "skin_segmentation_test.jpg"

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        sys.exit(1)

    print(f"Loaded image: {image_path} ({image.shape})")

    # Segment skin
    mask = segment_skin(image, method="ycbcr", apply_morphology=True)

    if mask is None:
        print("Warning: Not enough skin detected")
        mask = segment_skin_ycbcr(image)  # Get raw mask anyway

    # Get statistics
    stats = get_skin_statistics(image, mask)
    print(f"Skin ratio: {stats['skin_ratio']:.1%}")
    print(f"Mean LAB: L={stats['mean_L']:.1f}, a={stats['mean_a']:.1f}, b={stats['mean_b']:.1f}")

    # Visualize
    vis = visualize_segmentation(image, mask, output_path)
    print(f"Saved visualization to: {output_path}")
