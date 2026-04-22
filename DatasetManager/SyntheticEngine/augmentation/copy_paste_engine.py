"""
SyntheticEngine/augmentation/copy_paste_engine.py
==================================================
The advanced blending engine.
Extracts the object via its binary mask, optionally scales it (distance simulation),
applies lighting adaptations (histogram matching, pixel math, inversion),
and alpha-blends it onto a background.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from skimage import exposure

from core.sample_record import SampleRecord, Keypoint
from utils.mask_utils import get_object_bbox, soft_mask

log = logging.getLogger(__name__)

class CopyPasteEngine:
    """
    Orchestrates extracting an object from its source and pasting it onto a background.
    """

    def __init__(self, config: Dict[str, Any], debug: bool = False):
        self.cfg = config
        self.enabled = config.get("enabled", True)
        self.debug = debug

    def paste(self, record: SampleRecord, background: np.ndarray) -> SampleRecord:
        """
        Extract the object from `record`, apply lighting/scaling, and paste onto `background`.
        Returns a NEW SampleRecord with the composited image and updated keypoints.
        """
        if not self.enabled:
            return record

        try:
            # 1. Extract Object Bounding Box
            bbox = get_object_bbox(record.mask)
            if bbox is None:
                return record
            x0, y0, w_obj, h_obj = bbox
            
            # Crop foreground and its mask
            fg = record.image[y0:y0+h_obj, x0:x0+w_obj].copy()
            obj_mask = record.mask[y0:y0+h_obj, x0:x0+w_obj].copy()
            
            # Offset keypoints to local crop coords
            local_kps = []
            for kp in record.keypoints:
                if kp.vis > 0:
                    local_kps.append((kp.x - x0, kp.y - y0))
                else:
                    local_kps.append(None)

            # 2. Random Scaling (Distance Simulation)
            scale_range = self.cfg.get("object_scale_range", [0.4, 1.1])
            scale = random.uniform(scale_range[0], scale_range[1])
            
            w_new, h_new = int(w_obj * scale), int(h_obj * scale)
            if w_new < 5 or h_new < 5:
                if self.debug: log.debug("Object scaled too small (%.2f), skipping", scale)
                return record # Too small
                
            fg = cv2.resize(fg, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            obj_mask = cv2.resize(obj_mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
            
            # Update keypoints with scale
            scaled_kps = []
            for kp_coord in local_kps:
                if kp_coord:
                    scaled_kps.append((kp_coord[0] * scale, kp_coord[1] * scale))
                else:
                    scaled_kps.append(None)

            # 3. Object-Only Inversion (Prob A)
            inversion_cfg = self.cfg.get("inversion", {})
            obj_inverted = False
            if random.random() < inversion_cfg.get("object_only_prob", 0.05):
                intensity_range = inversion_cfg.get("intensity_range", [1.0, 1.0])
                strength = random.uniform(intensity_range[0], intensity_range[1])
                
                if strength == 1.0:
                    fg = 255 - fg
                else:
                    fg = (fg.astype(np.float32) * (1.0 - strength) + 
                         (255 - fg).astype(np.float32) * strength).astype(np.uint8)
                obj_inverted = True

            # 4. Pick Paste Position on Background
            bg_h, bg_w = background.shape[:2]
            if w_new >= bg_w or h_new >= bg_h:
                if self.debug: log.debug("Object larger than background, skipping")
                return record
                
            paste_x = random.randint(0, bg_w - w_new)
            paste_y = random.randint(0, bg_h - h_new)
            
            if self.debug:
                log.debug("Pasting %s: scale=%.2f, pos=(%d, %d)", 
                          record.source_id, scale, paste_x, paste_y)
            
            # 5. Histogram Matching (Lighting Adaptation)
            hist_matched = False
            if self.cfg.get("histogram_match", True):
                bg_crop = background[paste_y:paste_y+h_new, paste_x:paste_x+w_new]
                # Match fg to bg_crop
                try:
                    # skimage match_histograms expects RGB or handles channels
                    matched_fg = exposure.match_histograms(fg, bg_crop, channel_axis=-1)
                    fg = matched_fg.astype(np.uint8)
                    hist_matched = True
                except Exception as e:
                    log.warning("Histogram matching failed: %s", e)

            # 6. Pixel Math (Shadow / Glare)
            lighting_cfg = self.cfg.get("lighting", {})
            applied_lighting = []
            if hist_matched: applied_lighting.append("HistMatch")
            if obj_inverted: applied_lighting.append("ObjInvert")

            if random.random() < lighting_cfg.get("multiply_prob", 0.3):
                m_range = lighting_cfg.get("multiply_range", [0.5, 0.9])
                factor = random.uniform(m_range[0], m_range[1])
                fg = (fg.astype(np.float32) * factor).astype(np.uint8)
                applied_lighting.append(f"Mult(%.2f)" % factor)
            
            if random.random() < lighting_cfg.get("add_prob", 0.2):
                a_range = lighting_cfg.get("add_range", [20, 70])
                val = random.randint(a_range[0], a_range[1])
                fg = cv2.add(fg, np.array([val], dtype=np.uint8))
                applied_lighting.append(f"Add(%d)" % val)

            if self.debug and applied_lighting:
                log.debug("Lighting effects: %s", ", ".join(applied_lighting))

            # 7. Alpha Blending (Soft Edges)
            sigma_cfg = self.cfg.get("alpha_blend_sigma", 7)
            if isinstance(sigma_cfg, (list, tuple)):
                sigma = random.uniform(sigma_cfg[0], sigma_cfg[1])
            else:
                sigma = sigma_cfg
                
            alpha = soft_mask(obj_mask, sigma=sigma)
            
            bg_crop = background[paste_y:paste_y+h_new, paste_x:paste_x+w_new]
            blended = (bg_crop.astype(np.float32) * (1.0 - alpha[:, :, np.newaxis]) + 
                       fg.astype(np.float32) * alpha[:, :, np.newaxis]).astype(np.uint8)
            
            # Composite onto full background
            composite = background.copy()
            composite[paste_y:paste_y+h_new, paste_x:paste_x+w_new] = blended
            
            # Create full image mask
            new_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
            new_mask[paste_y:paste_y+h_new, paste_x:paste_x+w_new] = obj_mask

            # 8. Full-Image Inversion (Prob B)
            if random.random() < inversion_cfg.get("full_image_prob", 0.03):
                intensity_range = inversion_cfg.get("intensity_range", [1.0, 1.0])
                strength = random.uniform(intensity_range[0], intensity_range[1])
                
                if strength == 1.0:
                    composite = 255 - composite
                else:
                    composite = (composite.astype(np.float32) * (1.0 - strength) + 
                                (255 - composite).astype(np.float32) * strength).astype(np.uint8)
                
                if self.debug: log.debug("Applied full-image inversion (strength=%.2f)", strength)

            # 9. Remap Keypoints to Full Coords
            final_kps: List[Keypoint] = []
            for i, kp in enumerate(record.keypoints):
                if scaled_kps[i]:
                    nx = scaled_kps[i][0] + paste_x
                    ny = scaled_kps[i][1] + paste_y
                    # Clamp check
                    if 0 <= nx < bg_w and 0 <= ny < bg_h:
                        final_kps.append(Keypoint(nx, ny, kp.vis, kp.point_id))
                    else:
                        final_kps.append(Keypoint(0.0, 0.0, 0, kp.point_id))
                else:
                    final_kps.append(Keypoint(0.0, 0.0, 0, kp.point_id))

            # Return new record
            return SampleRecord(
                image=composite,
                mask=new_mask,
                keypoints=final_kps,
                img_path=record.img_path,
                img_w=bg_w,
                img_h=bg_h,
                class_id=record.class_id,
                source_id=record.source_id
            )

        except Exception as e:
            log.error("Copy-paste failed: %s", e)
            return record
