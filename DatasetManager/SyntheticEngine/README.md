# SyntheticEngine — Advanced Synthetic Data Generation

A fully automated pipeline to transform a small set of real-world reference images (with SAM2 masks and keypoints) into a massive, geometrically robust dataset for training YOLO pose estimation models.

## Key Features
- **Geometric Sync**: Uses Albumentations to transform images, masks, and keypoints in perfect unison.
- **Copy-Paste Augmentation**: Extracts objects and blends them onto new backgrounds with alpha-softening and histogram matching.
- **Environmental Simulation**: Lighting adaptation, pixel math (shadows/glare), and color inversion.
- **Spatial Alterations**: Random scaling (distance simulation) and perspective warping.
- **Simulated Occlusion**: Random black patches that automatically update keypoint visibility.
- **High Throughput**: Multi-process generation using all available CPU cores.

## Installation
Ensure you have the requirements installed in your project venv:
```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick Start
1. Place clean background images in `backgrounds/`.
2. Review paths and parameters in `config/default_config.yaml`.
3. Run the generator:
   ```powershell
    F:\RunningProjects\AutoSegmentor\.venv\Scripts\python.exe .\debug_sweep.py
   ```

## Configuration
All parameters are controlled via `config/default_config.yaml`.
- `samples_per_source`: Controls dataset scale (e.g., 10 ref images * 100 samples = 1,000 output images).
- `workers`: Set to `-1` for max speed, or `1` for single-threaded debugging.
- `export`: Toggle `pose`, `segmentation`, or `box` label outputs.

## Debugging & Visualisation
After generation, you can visually verify the keypoints on random samples:
```powershell
..\.venv\Scripts\python.exe utils/visualise.py --dataset outputs/pallet_synthetic_v1
```
Keypoints are color-coded:
- **Red**: Fully visible (vis=2)
- **Cyan**: Partially occluded (vis=1)
- **Labels**: p1, p2, etc., based on ID.
