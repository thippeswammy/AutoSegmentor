"""
SyntheticEngine/debug_sweep.py
=============================
A diagnostic tool that takes a single reference image and generates a grid
of samples using different augmentation intensities (Low, Mid, High).
Helps tune the [min, max] ranges in default_config.yaml.
"""
import os
import sys
import yaml
import cv2
import numpy as np
from pathlib import Path
import logging
import itertools

# Ensure local imports work
sys.path.append(str(Path(__file__).resolve().parent))

from core.label_io import load_dataset
from augmentation.geometric_augmentor import GeometricAugmentor
from augmentation.photometric_augmentor import PhotometricAugmentor
from augmentation.copy_paste_engine import CopyPasteEngine
from augmentation.occlusion_simulator import OcclusionSimulator
from backgrounds.background_manager import BackgroundManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DebugSweep")

def get_intensity_val(range_or_val, level):
    """level: 0 (min), 1 (mid), 2 (max)"""
    if not isinstance(range_or_val, (list, tuple)):
        return range_or_val
    
    low, high = range_or_val
    if level == 0: return low
    if level == 1: return (low + high) / 2
    return high

def create_debug_config(base_cfg, level, combo_stages):
    """
    Creates a config for a specific set of active stages and intensity level.
    combo_stages: list of stage names to enable
    """
    import copy
    cfg = copy.deepcopy(base_cfg)
    aug = cfg['augmentation']
    
    # 1. Reset all to disabled
    for stage in ['geometric', 'copy_paste', 'photometric', 'occlusion']:
        aug[stage]['enabled'] = False
        aug[stage]['prob'] = 1.0
        if stage == 'copy_paste':
            aug[stage]['inversion']['enabled'] = False
            aug[stage]['lighting']['enabled'] = False

    # 2. Enable based on combo_stages
    if any(s in combo_stages for s in ['inversion', 'lighting']):
        aug['copy_paste']['enabled'] = True
    
    if 'geometric' in combo_stages: aug['geometric']['enabled'] = True
    if 'photometric' in combo_stages: aug['photometric']['enabled'] = True
    if 'occlusion' in combo_stages: aug['occlusion']['enabled'] = True
    if 'inversion' in combo_stages: 
        aug['copy_paste']['enabled'] = True
        aug['copy_paste']['inversion']['enabled'] = True
    if 'lighting' in combo_stages: 
        aug['copy_paste']['enabled'] = True
        aug['copy_paste']['lighting']['enabled'] = True

    # 3. Set Intensity (if enabled)
    # Geometric
    g = aug['geometric']
    if g['enabled']:
        g['rotate_limit'] = get_intensity_val([10, 45], level)
        g['scale_range'] = [get_intensity_val([0.9, 0.7], level), get_intensity_val([1.1, 1.3], level)]

    # Copy Paste
    cp = aug['copy_paste']
    if cp['enabled']:
        cp['object_scale_range'] = [get_intensity_val(cp['object_scale_range'], level)] * 2
        cp['alpha_blend_sigma'] = get_intensity_val(cp['alpha_blend_sigma'], level)
        if cp['inversion']['enabled']:
            cp['inversion']['intensity_range'] = [get_intensity_val(cp['inversion']['intensity_range'], level)] * 2
        
    # Photometric
    p = aug['photometric']
    if p['enabled']:
        p['brightness_range'] = [get_intensity_val(p['brightness_range'], level)] * 2
        p['contrast_range'] = [get_intensity_val(p['contrast_range'], level)] * 2
    
    # Occlusion
    o = aug['occlusion']
    if o['enabled']:
        o['max_patches'] = level + 1
        o['patch_size_max'] = get_intensity_val([o['patch_size_min'], o['patch_size_max']], level)

    return cfg

def main():
    config_path = "config/default_config.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    # Setup paths
    out_dir = Path("outputs/debug_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process color mapping
    c2l_raw = base_config.get('color_to_label', {})
    c2l = {}
    for k, v in c2l_raw.items():
        if isinstance(k, str) and k.startswith("["):
            t = tuple(map(int, k.strip("[]").split(",")))
            c2l[t] = v
        else:
            c2l[k] = v

    # Load ONE record
    records = load_dataset(
        images_dir=Path(base_config['input']['images_dir']),
        masks_dir=Path(base_config['input']['masks_dir']),
        pose_json=Path(base_config['input']['pose_labels_json']),
        color_to_label=c2l,
        class_id=base_config.get('class_id', 0),
        num_keypoints=base_config.get('num_keypoints', 12)
    )
    
    if not records:
        print("No images found to sweep.")
        return
        
    record = records[0]
    bg_mgr = BackgroundManager(base_config['input']['backgrounds_dir'])
    bg = bg_mgr.get_random_bg()

    levels = ["Low", "Mid", "High"]
    stages = ["geometric", "inversion", "lighting", "photometric", "occlusion"]
    
    # Generate ALL combinations of stages (1 to 5)
    all_combos = []
    for r in range(1, len(stages) + 1):
        for combo in itertools.combinations(stages, r):
            all_combos.append(combo)

    total_imgs = len(all_combos) * len(levels)
    print(f"Generating {total_imgs} variations ({len(all_combos)} combos x {len(levels)} levels)...")

    for combo in all_combos:
        combo_name = "+".join([s[:4] for s in combo]) # Short name e.g. geom+inv
        for i, level_name in enumerate(levels):
            cfg = create_debug_config(base_config, i, combo)
            
            # Initialize modules
            geom = GeometricAugmentor(cfg['augmentation']['geometric'], debug=False)
            cp = CopyPasteEngine(cfg['augmentation']['copy_paste'], debug=False)
            photo = PhotometricAugmentor(cfg['augmentation']['photometric'], debug=False)
            occ = OcclusionSimulator(cfg['augmentation']['occlusion'], debug=False)
            
            # Run Pipeline
            rec = record
            if cfg['augmentation']['geometric']['enabled']:
                rec = geom.transform(rec)
            if cfg['augmentation']['copy_paste']['enabled']:
                rec = cp.paste(rec, bg)
            if cfg['augmentation']['photometric']['enabled']:
                rec = photo.transform(rec)
            if cfg['augmentation']['occlusion']['enabled']:
                rec = occ.apply(rec)
            
            # Save
            save_path = out_dir / f"{combo_name}_{level_name}.jpg"
            cv2.imwrite(str(save_path), rec.image)
            # log.info(f"Saved to {save_path}")

    print(f"\nExhaustive Sweep Matrix Complete! Check {out_dir} for {total_imgs} results.")

if __name__ == "__main__":
    main()
