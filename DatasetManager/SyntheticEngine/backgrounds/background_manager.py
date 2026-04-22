"""
SyntheticEngine/backgrounds/background_manager.py
==================================================
Scans the backgrounds directory and serves random background images.
Caches paths at initialization for fast access.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

class BackgroundManager:
    """
    Manages a pool of background images.
    """

    def __init__(self, bg_dir: str):
        self.bg_dir = Path(bg_dir)
        self.bg_paths: List[Path] = []
        
        if not self.bg_dir.exists():
            log.warning("Backgrounds directory not found: %s", self.bg_dir)
            return

        # Scan for images
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            self.bg_paths.extend(list(self.bg_dir.rglob(f"*{ext}")))
            self.bg_paths.extend(list(self.bg_dir.rglob(f"*{ext.upper()}")))

        log.info("Found %d background images in %s", len(self.bg_paths), self.bg_dir)

    def get_random_bg(self, target_size: Optional[tuple] = None) -> np.ndarray:
        """
        Return a random background image.
        If target_size (h, w) is provided, the background is resized/cropped to match.
        """
        if not self.bg_paths:
            # Return a gray canvas if no backgrounds available
            h, w = target_size if target_size else (720, 1280)
            return np.full((h, w, 3), 128, dtype=np.uint8)

        bg_path = random.choice(self.bg_paths)
        bg = cv2.imread(str(bg_path))
        
        if bg is None:
            log.error("Failed to read background: %s", bg_path)
            return self.get_random_bg(target_size) # Try again

        if target_size:
            target_h, target_w = target_size
            bg_h, bg_w = bg.shape[:2]
            
            # Simple resize for now. 
            # Could be improved to random crop or aspect-ratio preserving resize.
            if bg_h != target_h or bg_w != target_w:
                bg = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                
        return bg
