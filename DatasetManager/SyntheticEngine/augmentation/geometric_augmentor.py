"""
SyntheticEngine/augmentation/geometric_augmentor.py
====================================================
Uses Albumentations to apply geometric transforms (rotate, flip, scale, perspective)
to the image, mask, and keypoints simultaneously.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import albumentations as A
import numpy as np

from core.sample_record import SampleRecord

log = logging.getLogger(__name__)

class GeometricAugmentor:
    """
    Handles geometric transformations that require synchronous updates to
    images, masks, and keypoints.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Albumentations pipeline based on config.
        """
        self.enabled = config.get("enabled", True)
        if not self.enabled:
            self.transform_pipeline = None
            return

        # Define the geometric pipeline
        transforms = []
        
        # Rotation
        rotate_limit = config.get("rotate_limit", 30)
        if rotate_limit > 0:
            transforms.append(A.Rotate(limit=rotate_limit, p=0.7))
            
        # Horizontal Flip
        h_flip_prob = config.get("h_flip_prob", 0.5)
        if h_flip_prob > 0:
            transforms.append(A.HorizontalFlip(p=h_flip_prob))
            
        # Vertical Flip
        v_flip_prob = config.get("v_flip_prob", 0.0)
        if v_flip_prob > 0:
            transforms.append(A.VerticalFlip(p=v_flip_prob))
            
        # Shift, Scale, Rotate (more advanced version)
        scale_range = config.get("scale_range", [0.8, 1.2])
        shear_limit = config.get("shear_limit", 10)
        transforms.append(A.ShiftScaleRotate(
            shift_limit=0.0625, 
            scale_limit=(scale_range[0]-1, scale_range[1]-1), 
            rotate_limit=0, # Already have Rotate
            p=0.5
        ))
        
        # Perspective
        perspective_scale = config.get("perspective_scale", [0.05, 0.1])
        if perspective_scale:
            transforms.append(A.Perspective(scale=perspective_scale, p=0.4))

        self.transform_pipeline = A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(
                format='xy', 
                remove_invisible=False, # We handle visibility in SampleRecord.update_from_aug
                label_fields=[]
            ),
            additional_targets={'mask': 'mask'}
        )

    def transform(self, record: SampleRecord) -> SampleRecord:
        """
        Apply geometric transformations to the record.
        """
        if not self.enabled or self.transform_pipeline is None:
            return record

        try:
            # Albumentations expects RGB for some transforms internally or better results
            # but usually it's fine with BGR. We'll stick to BGR since cv2 is used.
            
            # Prepare inputs
            kps = record.kp_pixel_coords()
            
            # Run transform
            result = self.transform_pipeline(
                image=record.image,
                mask=record.mask,
                keypoints=kps
            )
            
            # Update record
            return record.update_from_aug(
                new_image=result['image'],
                new_mask=result['mask'],
                new_kp_coords=result['keypoints']
            )
        except Exception as e:
            log.error("Geometric transform failed: %s", e)
            return record
