"""
SyntheticEngine/augmentation/photometric_augmentor.py
=====================================================
Applies pixel-level effects (color, noise, blur, compression) to the image.
These transforms do NOT change the geometry (mask/keypoints stay the same).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import albumentations as A
import numpy as np

from core.sample_record import SampleRecord

log = logging.getLogger(__name__)

class PhotometricAugmentor:
    """
    Handles photometric transformations (color, noise, blur).
    Applied to the final composited image.
    """

    def __init__(self, config: Dict[str, Any], debug: bool = False):
        """
        Initialize the Albumentations pipeline based on config.
        """
        self.enabled = config.get("enabled", True)
        self.debug = debug
        if not self.enabled:
            self.transform_pipeline = None
            return

        transforms = []
        
        # Brightness & Contrast
        brightness = config.get("brightness_range", [-0.2, 0.2])
        contrast = config.get("contrast_range", [0.8, 1.2])
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=brightness,
            contrast_limit=(contrast[0]-1, contrast[1]-1),
            p=0.8
        ))
        
        # Hue, Saturation, Value
        hue_limit = config.get("hue_shift_limit", 20)
        sat_range = config.get("saturation_range", [0.7, 1.3])
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=hue_limit,
            sat_shift_limit=(int((sat_range[0]-1)*100), int((sat_range[1]-1)*100)),
            val_shift_limit=20,
            p=0.5
        ))
        
        # Blur
        blur_prob = config.get("blur_prob", 0.2)
        if blur_prob > 0:
            transforms.append(A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=(3, 7)),
                A.MedianBlur(blur_limit=5),
            ], p=blur_prob))
            
        # Noise
        noise_prob = config.get("noise_prob", 0.2)
        if noise_prob > 0:
            transforms.append(A.GaussNoise(std_range=(0.2, 0.5), p=noise_prob))
            
        # JPEG Compression
        jpeg_prob = config.get("jpeg_prob", 0.1)
        if jpeg_prob > 0:
            transforms.append(A.ImageCompression(quality_range=(60, 100), p=jpeg_prob))

        composer = A.ReplayCompose if self.debug else A.Compose
        self.transform_pipeline = composer(transforms)

    def transform(self, record: SampleRecord) -> SampleRecord:
        """
        Apply photometric transformations to the image in the record.
        """
        if not self.enabled or self.transform_pipeline is None:
            return record

        try:
            result = self.transform_pipeline(image=record.image)
            
            if self.debug and 'replay' in result:
                applied = []
                for t in result['replay']['transforms']:
                    # Special handling for OneOf
                    if 'transforms' in t:
                        for sub_t in t['transforms']:
                            if sub_t.get('applied', False):
                                applied.append(sub_t['__class_fullname__'])
                    elif t.get('applied', False):
                        applied.append(t['__class_fullname__'])
                if applied:
                    log.debug("Photometric transforms applied: %s", ", ".join(applied))

            record.image = result['image']
            return record
        except Exception as e:
            log.error("Photometric transform failed: %s", e)
            return record
