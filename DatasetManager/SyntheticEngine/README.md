# SyntheticEngine — Advanced Synthetic Data Generation

The **SyntheticEngine** is a high-throughput pipeline designed to multiply a small set of real-world reference images into a massive, robust dataset for YOLO pose estimation.

[< Back to Dataset Manager](../README.md)
## Key Features
- **Geometric Sync**: Transforms images, masks, and keypoints in perfect unison using Albumentations.
- **Copy-Paste Augmentation**: Extracts objects and blends them onto new backgrounds with alpha-softening and histogram matching.
- **Environmental Simulation**: Lighting adaptation, shadows, glare, and color inversion.
- **Occlusion Tracking**: Random occlusion patches automatically update keypoint visibility flags.

## Installation
The engine uses the main project dependencies. Ensure your `.venv` is active:
```bash
pip install albumentations opencv-python numpy tqdm
```

## Quick Start
1.  **Prepare Backgrounds**: Place clean background images in the `backgrounds/` directory.
2.  **Configure**: Review `config/default_config.yaml`.
3.  **Run Sweep**:
    ```bash
    python debug_sweep.py
    ```

## Configuration
Controlled via `config/default_config.yaml`:
- `samples_per_source`: Number of synthetic images to generate per reference frame.
- `workers`: Number of parallel processes (set to `-1` for all CPU cores).
- `export`: Toggle output formats (Pose, Segmentation, Box).

## Visualisation
Verify the quality of generated data using the visualisation tool:
```bash
python utils/visualise.py --dataset outputs/your_dataset_name
```
- **Red Points**: Fully visible.
- **Cyan Points**: Occluded.
