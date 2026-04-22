"""
SyntheticEngine/utils/mask_utils.py
=====================================
Utilities for working with color-mapped RGB masks (same convention as
DatasetCreator.py's `color_to_label` dict).

All mask images are loaded by OpenCV → BGR channel order.
`color` tuples in this module are therefore (B, G, R).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def extract_binary_mask(
    color_mask: np.ndarray,
    color: Tuple[int, int, int],
) -> np.ndarray:
    """
    Extract a binary (0 / 255) single-channel mask for one class.

    Parameters
    ----------
    color_mask : H×W×3 uint8 BGR image (as loaded by cv2.imread)
    color      : BGR tuple for the target class, e.g. (255, 255, 255)

    Returns
    -------
    H×W uint8 array — 255 where pixel == color, 0 elsewhere.
    """
    target = np.array(color, dtype=np.uint8)
    match = np.all(color_mask == target, axis=-1)
    return match.astype(np.uint8) * 255


def get_object_bbox(
    binary_mask: np.ndarray,
    margin: float = 0.0,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Bounding box of the non-zero region in a binary mask.

    Parameters
    ----------
    binary_mask : H×W uint8 (0 / 255)
    margin      : expand each side by this fraction of object width/height

    Returns
    -------
    (x0, y0, width, height) in pixel coords, or None if mask is empty.
    """
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None

    h_img, w_img = binary_mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    pad_x = int((x1 - x0) * margin)
    pad_y = int((y1 - y0) * margin)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w_img - 1, x1 + pad_x)
    y1 = min(h_img - 1, y1 + pad_y)

    return x0, y0, x1 - x0, y1 - y0


def mask_to_polygon(
    binary_mask: np.ndarray,
) -> Optional[List[Tuple[float, float]]]:
    """
    Extract the largest contour from a binary mask as normalised (x, y) pairs.
    Suitable for YOLO instance-segmentation label format.

    Returns None if no contour is found.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    h, w = binary_mask.shape[:2]
    pts = biggest.reshape(-1, 2)
    return [(float(p[0]) / w, float(p[1]) / h) for p in pts]


def soft_mask(binary_mask: np.ndarray, sigma: int = 7) -> np.ndarray:
    """
    Blur a binary mask to create a smooth alpha channel for alpha-blending.

    Returns a float32 H×W array in [0.0, 1.0].
    """
    blurred = cv2.GaussianBlur(
        binary_mask.astype(np.float32),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    return (blurred / 255.0).clip(0.0, 1.0)
