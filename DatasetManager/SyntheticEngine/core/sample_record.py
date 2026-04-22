"""
SyntheticEngine/core/sample_record.py
======================================
Central data carrier passed through every pipeline stage.
All keypoint coordinates are absolute pixel values (not normalised).
Normalisation to [0,1] happens only at label-write time in label_io.py.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class Keypoint:
    """A single keypoint with pixel-space coordinates and a visibility flag."""

    x: float       # absolute pixel col
    y: float       # absolute pixel row
    vis: int       # 0 = not labelled, 1 = occluded, 2 = visible
    point_id: int = 0  # 1-indexed (p1 … p12)

    def is_valid(self) -> bool:
        """True when the keypoint has a meaningful position."""
        return self.vis > 0


@dataclass
class SampleRecord:
    """
    Carries one training sample (image + binary mask + keypoints) through
    the entire augmentation pipeline.

    Attributes
    ----------
    image      : H×W×3 uint8 BGR array
    mask       : H×W uint8 binary (0 or 255) — one object class
    keypoints  : list of exactly `num_keypoints` Keypoint objects, ordered p1…pN
    img_path   : original source file path (used for debugging / logging)
    img_w      : image width  (== image.shape[1])
    img_h      : image height (== image.shape[0])
    class_id   : YOLO class index for this object
    source_id  : image stem, e.g. "Img6_00000"
    """

    image: np.ndarray
    mask: np.ndarray
    keypoints: List[Keypoint]
    img_path: Path
    img_w: int
    img_h: int
    class_id: int = 0
    source_id: str = ""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def visible_keypoints(self) -> List[Keypoint]:
        """Return only keypoints with vis > 0."""
        return [kp for kp in self.keypoints if kp.is_valid()]

    def kp_pixel_coords(self) -> List[tuple]:
        """Return [(x, y), ...] — the format Albumentations expects."""
        return [(kp.x, kp.y) for kp in self.keypoints]

    def update_from_aug(
        self,
        new_image: np.ndarray,
        new_mask: np.ndarray,
        new_kp_coords: list,
    ) -> "SampleRecord":
        """
        Create a new SampleRecord after a geometric transform.

        `new_kp_coords` must have the same length as `self.keypoints`.
        Any keypoint that landed outside [0, W) × [0, H) is set to vis=0.
        """
        h, w = new_image.shape[:2]
        new_kps: List[Keypoint] = []

        for i, kp in enumerate(self.keypoints):
            if kp.vis == 0:
                new_kps.append(Keypoint(0.0, 0.0, 0, kp.point_id))
                continue
            if i < len(new_kp_coords) and new_kp_coords[i] is not None:
                nx, ny = float(new_kp_coords[i][0]), float(new_kp_coords[i][1])
                if 0.0 <= nx < w and 0.0 <= ny < h:
                    new_kps.append(Keypoint(nx, ny, kp.vis, kp.point_id))
                else:
                    new_kps.append(Keypoint(0.0, 0.0, 0, kp.point_id))
            else:
                new_kps.append(Keypoint(0.0, 0.0, 0, kp.point_id))

        return dataclasses.replace(
            self,
            image=new_image,
            mask=new_mask,
            keypoints=new_kps,
            img_w=w,
            img_h=h,
        )
