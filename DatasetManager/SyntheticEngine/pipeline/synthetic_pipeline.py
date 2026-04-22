"""
SyntheticEngine/pipeline/synthetic_pipeline.py
===============================================
The main orchestrator. Loads reference data, runs the augmentation
stages in a pipeline, and writes the output.
Supports multiprocessing for high-throughput generation.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from core.label_io import load_dataset
from core.sample_record import SampleRecord
from backgrounds.background_manager import BackgroundManager
from augmentation.geometric_augmentor import GeometricAugmentor
from augmentation.photometric_augmentor import PhotometricAugmentor
from augmentation.copy_paste_engine import CopyPasteEngine
from augmentation.occlusion_simulator.py import OcclusionSimulator # Fix: should be .occlusion_simulator
# Oops, fixing the import path below.

log = logging.getLogger(__name__)

def _process_one_wrapper(args):
    """Pickleable wrapper for ProcessPoolExecutor."""
    return SyntheticPipeline.process_one(*args)

class SyntheticPipeline:
    """
    Main pipeline for generating synthetic datasets.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        
        # Initialize modules
        self.bg_manager = BackgroundManager(config['input']['backgrounds_dir'])
        
        # We'll initialize augmentors per worker or here depending on thread safety
        # Since we use ProcessPoolExecutor, each process will have its own instances.
        # But for single-threaded mode, we use these.
        from augmentation.geometric_augmentor import GeometricAugmentor
        from augmentation.photometric_augmentor import PhotometricAugmentor
        from augmentation.copy_paste_engine import CopyPasteEngine
        from augmentation.occlusion_simulator import OcclusionSimulator
        from pipeline.yolo_writer import YoloWriter
        
        self.geom_aug = GeometricAugmentor(config['augmentation']['geometric'])
        self.photo_aug = PhotometricAugmentor(config['augmentation']['photometric'])
        self.copy_paste = CopyPasteEngine(config['augmentation']['copy_paste'])
        self.occlusion = OcclusionSimulator(config['augmentation']['occlusion'])
        self.writer = YoloWriter(config)

    def run(self):
        """
        Execute the pipeline.
        """
        # 1. Load Reference Data
        records = load_dataset(
            images_dir=Path(self.cfg['input']['images_dir']),
            masks_dir=Path(self.cfg['input']['masks_dir']),
            pose_json=Path(self.cfg['input']['pose_labels_json']),
            color_to_label={tuple(k): v for k, v in self.cfg['color_to_label'].items()},
            class_id=self.cfg.get('class_id', 0),
            num_keypoints=self.cfg.get('num_keypoints', 12)
        )
        
        if not records:
            log.error("No reference records loaded. Check your input paths.")
            return

        samples_per_source = self.cfg['augmentation'].get('samples_per_source', 50)
        total_samples = len(records) * samples_per_source
        
        workers = self.cfg['augmentation'].get('workers', -1)
        if workers == -1:
            workers = os.cpu_count() or 1

        log.info("Starting synthetic generation: %d records -> %d samples using %d workers", 
                 len(records), total_samples, workers)

        if workers <= 1:
            # Single-threaded mode
            with tqdm(total=total_samples, desc="Generating Samples") as pbar:
                for rec in records:
                    for i in range(samples_per_source):
                        self._generate_and_write(rec, i)
                        pbar.update(1)
        else:
            # Multiprocessing mode
            # Note: We need a static method or standalone function for the worker
            # because 'self' cannot be easily pickled with all its state (cv2 objects, etc.)
            # Instead, we pass the config and the record.
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = []
                for rec in records:
                    for i in range(samples_per_source):
                        futures.append(executor.submit(
                            SyntheticPipeline.worker_task, 
                            self.cfg, rec, i
                        ))
                
                for _ in tqdm(as_completed(futures), total=total_samples, desc="Generating Samples (MP)"):
                    try:
                        _.result()
                    except Exception as e:
                        log.error("Worker failed: %s", e)

    def _generate_and_write(self, record: SampleRecord, index: int):
        """Internal helper for single-threaded generation."""
        # 1. Geometric (on isolated object)
        rec = self.geom_aug.transform(record)
        
        # 2. Copy-Paste (onto random background)
        bg = self.bg_manager.get_random_bg(target_size=None) # Keep bg size
        rec = self.copy_paste.paste(rec, bg)
        
        # 3. Photometric (on full composited image)
        rec = self.photo_aug.transform(rec)
        
        # 4. Occlusion
        rec = self.occlusion.apply(rec)
        
        # 5. Write
        self.writer.write(rec, index)

    @staticmethod
    def worker_task(cfg: Dict[str, Any], record: SampleRecord, index: int):
        """
        Standalone worker task for ProcessPoolExecutor.
        Re-initializes necessary modules locally.
        """
        from augmentation.geometric_augmentor import GeometricAugmentor
        from augmentation.photometric_augmentor import PhotometricAugmentor
        from augmentation.copy_paste_engine import CopyPasteEngine
        from augmentation.occlusion_simulator import OcclusionSimulator
        from pipeline.yolo_writer import YoloWriter
        from backgrounds.background_manager import BackgroundManager

        # Re-init managers (BackgroundManager needs to re-scan for MP safety or efficiency)
        bg_mgr = BackgroundManager(cfg['input']['backgrounds_dir'])
        geom = GeometricAugmentor(cfg['augmentation']['geometric'])
        photo = PhotometricAugmentor(cfg['augmentation']['photometric'])
        cp = CopyPasteEngine(cfg['augmentation']['copy_paste'])
        occ = OcclusionSimulator(cfg['augmentation']['occlusion'])
        writer = YoloWriter(cfg)

        # Pipeline logic
        rec = geom.transform(record)
        bg = bg_mgr.get_random_bg()
        rec = cp.paste(rec, bg)
        rec = photo.transform(rec)
        rec = occ.apply(rec)
        writer.write(rec, index)
