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
from augmentation.occlusion_simulator import OcclusionSimulator

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
        
        debug = config.get('debug', False)
        self.geom_aug = GeometricAugmentor(config['augmentation']['geometric'], debug=debug)
        self.photo_aug = PhotometricAugmentor(config['augmentation']['photometric'], debug=debug)
        self.copy_paste = CopyPasteEngine(config['augmentation']['copy_paste'], debug=debug)
        self.occlusion = OcclusionSimulator(config['augmentation']['occlusion'], debug=debug)
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
            output_path_str = str(self.writer.full_path)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = []
                for rec in records:
                    for i in range(samples_per_source):
                        futures.append(executor.submit(
                            SyntheticPipeline.worker_task, 
                            self.cfg, rec, i, output_path_str
                        ))
                
                for _ in tqdm(as_completed(futures), total=total_samples, desc="Generating Samples (MP)"):
                    try:
                        _.result()
                    except Exception as e:
                        log.error("Worker failed: %s", e)

    def _generate_and_write(self, record: SampleRecord, index: int):
        """Internal helper for single-threaded generation."""
        import random
        aug_cfg = self.cfg['augmentation']
        rec = record

        # 1. Geometric (on isolated object)
        geom_prob = aug_cfg['geometric'].get('prob', 1.0)
        if random.random() < geom_prob:
            rec = self.geom_aug.transform(rec)
        
        # 2. Copy-Paste (onto random background)
        cp_prob = aug_cfg['copy_paste'].get('prob', 1.0)
        if random.random() < cp_prob:
            bg = self.bg_manager.get_random_bg(target_size=None)
            rec = self.copy_paste.paste(rec, bg)
        
        # 3. Photometric (on full composited image)
        photo_prob = aug_cfg['photometric'].get('prob', 1.0)
        if random.random() < photo_prob:
            rec = self.photo_aug.transform(rec)
        
        # 4. Occlusion
        occ_prob = aug_cfg['occlusion'].get('prob', 1.0)
        if random.random() < occ_prob:
            rec = self.occlusion.apply(rec)
        
        # 5. Write
        self.writer.write(rec, index)

    @staticmethod
    def worker_task(cfg: Dict[str, Any], record: SampleRecord, index: int, output_path: str):
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

        # Re-init managers
        debug = cfg.get('debug', False)
        bg_mgr = BackgroundManager(cfg['input']['backgrounds_dir'])
        geom = GeometricAugmentor(cfg['augmentation']['geometric'], debug=debug)
        photo = PhotometricAugmentor(cfg['augmentation']['photometric'], debug=debug)
        cp = CopyPasteEngine(cfg['augmentation']['copy_paste'], debug=debug)
        occ = OcclusionSimulator(cfg['augmentation']['occlusion'], debug=debug)
        writer = YoloWriter(cfg, pre_created_path=Path(output_path))

        # Pipeline logic
        import random
        aug_cfg = cfg['augmentation']
        rec = record

        # 1. Geometric
        geom_prob = aug_cfg['geometric'].get('prob', 1.0)
        if random.random() < geom_prob:
            rec = geom.transform(rec)

        # 2. Copy-Paste
        cp_prob = aug_cfg['copy_paste'].get('prob', 1.0)
        if random.random() < cp_prob:
            bg = bg_mgr.get_random_bg()
            rec = cp.paste(rec, bg)

        # 3. Photometric
        photo_prob = aug_cfg['photometric'].get('prob', 1.0)
        if random.random() < photo_prob:
            rec = photo.transform(rec)

        # 4. Occlusion
        occ_prob = aug_cfg['occlusion'].get('prob', 1.0)
        if random.random() < occ_prob:
            rec = occ.apply(rec)

        writer.write(rec, index)
