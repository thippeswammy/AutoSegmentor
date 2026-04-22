"""
SyntheticEngine/augmentation/occlusion_simulator.py
===================================================
Simulates partial occlusion by adding random black patches to the image.
Updates keypoint visibility: if a keypoint falls under a patch, its `vis`
flag is set to 1 (occluded).
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

import cv2
import numpy as np

from core.sample_record import SampleRecord, Keypoint
from utils.mask_utils import get_object_bbox

log = logging.getLogger(__name__)

class OcclusionSimulator:
    """
    Applies rectangular occlusion patches to the composited image.
    """

    def __init__(self, config: Dict[str, Any], debug: bool = False):
        self.cfg = config
        self.enabled = config.get("enabled", True)
        self.debug = debug

    def apply(self, record: SampleRecord) -> SampleRecord:
        """
        Apply random black patches and update keypoint visibility.
        """
        if not self.enabled or random.random() > self.cfg.get("prob", 0.3):
            return record

        try:
            # 1. Get object bounding box to scale patches relative to object
            bbox = get_object_bbox(record.mask)
            if bbox is None:
                return record
            x_obj, y_obj, w_obj, h_obj = bbox

            # 2. Generate patches
            num_patches = random.randint(1, self.cfg.get("max_patches", 3))
            img_h, img_w = record.image.shape[:2]
            
            if self.debug: log.debug("Applying %d occlusion patches", num_patches)

            p_size_min = self.cfg.get("patch_size_min", 0.05)
            p_size_max = self.cfg.get("patch_size_max", 0.3)

            for _ in range(num_patches):
                # Random patch size relative to object
                pw = int(w_obj * random.uniform(p_size_min, p_size_max))
                ph = int(h_obj * random.uniform(p_size_min, p_size_max))
                
                if pw < 2 or ph < 2:
                    continue
                
                # Position patch so it overlaps with the object at least partially
                cx = random.randint(x_obj, x_obj + w_obj)
                cy = random.randint(y_obj, y_obj + h_obj)
                
                x1 = max(0, cx - pw // 2)
                y1 = max(0, cy - ph // 2)
                x2 = min(img_w, x1 + pw)
                y2 = min(img_h, y1 + ph)
                
                # Apply patch to image (black)
                record.image[y1:y2, x1:x2] = 0
                
                # 3. Update keypoint visibility
                for kp in record.keypoints:
                    if kp.vis == 2: # Only change "visible" to "occluded"
                        if x1 <= kp.x < x2 and y1 <= kp.y < y2:
                            kp.vis = 1 # Mark as occluded
                            if self.debug: log.debug("Keypoint %d occluded by patch", kp.point_id)
            
            return record

        except Exception as e:
            log.error("Occlusion simulation failed: %s", e)
            return record
