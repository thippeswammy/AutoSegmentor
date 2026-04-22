"""
SyntheticEngine/pipeline/yolo_writer.py
=========================================
Handles writing images and labels to the YOLO folder structure.
Assigns samples to train/valid/test splits based on configured ratios.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict

import cv2

# Add YolovDatasetManager to path to reuse create_yolo_structure
sys.path.append(str(Path(__file__).resolve().parents[2] / "YolovDatasetManager"))
try:
    import create_yolo_structure
except ImportError:
    # Fallback if path manipulation fails
    create_yolo_structure = None

from core.sample_record import SampleRecord
from core.label_io import (
    write_yolo_pose_label, 
    write_yolo_seg_label, 
    write_yolo_box_label
)

log = logging.getLogger(__name__)

class YoloWriter:
    """
    Writes SampleRecords to a YOLO-formatted dataset on disk.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.output_dir = Path(config['output']['dataset_dir'])
        self.folder_name = config['output']['folder_name']
        
        # Create folder structure
        if create_yolo_structure:
            full_path, _ = create_yolo_structure.create_yolo_folder_structure(
                folder_name=self.folder_name,
                main_path=str(self.output_dir),
                num_classes=config.get('class_names', ['pallet'])
            )
            self.full_path = Path(full_path)
        else:
            # Manual fallback
            self.full_path = self.output_dir / self.folder_name
            for split in ('train', 'valid', 'test'):
                for sub in ('images', 'labels_pose', 'labels_mask', 'labels_box'):
                    (self.full_path / split / sub).mkdir(parents=True, exist_ok=True)

        self.train_split = config['output'].get('train_split', 0.85)
        self.val_split = config['output'].get('val_split', 0.10)
        self.test_split = config['output'].get('test_split', 0.05)
        
        self.export_cfg = config.get('export', {'pose': True, 'segmentation': False, 'box': False})

    def write(self, record: SampleRecord, index: int) -> str:
        """
        Save the sample to a random split based on ratios.
        Returns the path to the saved image.
        """
        # Determine split
        r = random.random()
        if r < self.train_split:
            split = "train"
        elif r < self.train_split + self.val_split:
            split = "valid"
        else:
            split = "test"

        base_name = f"{record.source_id}_{index:05d}"
        
        # 1. Save Image
        img_filename = f"{base_name}.jpg"
        img_path = self.full_path / split / "images" / img_filename
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), record.image)

        # 2. Save Labels
        if self.export_cfg.get('pose', True):
            label_path = self.full_path / split / "labels_pose" / f"{base_name}.txt"
            write_yolo_pose_label(record, label_path, bbox_margin=self.cfg.get('bbox_margin', 0.05))

        if self.export_cfg.get('segmentation', False):
            label_path = self.full_path / split / "labels_mask" / f"{base_name}.txt"
            write_yolo_seg_label(record, label_path)

        if self.export_cfg.get('box', False):
            label_path = self.full_path / split / "labels_box" / f"{base_name}.txt"
            write_yolo_box_label(record, label_path, bbox_margin=self.cfg.get('bbox_margin', 0.05))

        return str(img_path)
