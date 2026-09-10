# Dataset Manager

Once you've verified annotations in the main app, `DatasetManager/` takes over as a separate,
downstream toolchain — it's never called automatically by the annotation pipeline itself.
There are two independent tools here, used one after the other.

## YOLO Dataset Manager

Converts verified images and masks into a YOLOv8/v11-ready dataset. This is what runs when
you press `Ctrl+E` in the app (via `ExportDialog`), or standalone via
`DatasetManager/YolovDatasetManager/DatasetCreator.py`.

- **Input**: verified images + masks from `workspace/working_dir/<video>/verified/`.
- **Mask → polygon**: color-coded segmentation masks become normalized YOLO polygon
  annotations.
- **Output formats, all at once**: detection (bbox), instance segmentation, and pose.
- **Augmentation**: blur, noise, and color jitter (via Albumentations), multiplying each
  reference frame into several training variants by default.
- **Splitting**: configurable `train`/`valid`/`test` ratios, plus a generated `data.yaml`
  ready to hand to `ultralytics`.
- **Performance**: multithreaded, with optional CUDA acceleration for image transforms.

See the
[YOLO Dataset Manager README](https://github.com/thippeswammy/AutoSegmentor/blob/master/DatasetManager/YolovDatasetManager/README.md)
for configuration details.

## Synthetic Engine

A separate, **offline** tool for going beyond what SAM2/CoTracker3 annotated: it multiplies
a small set of verified reference frames into a much larger, more varied training set. It has
its own entry point (`run.py` / `debug_sweep.py`) and isn't invoked by the main pipeline or
by the YOLO exporter — you run it as its own step, after export, when you want more data than
your source footage alone provides.

- **Copy-paste augmentation**: extracts labeled objects and composites them onto new
  backgrounds, with alpha-softening and histogram matching so they blend in.
- **Geometric + photometric transforms**: applied in sync across the image, its mask, and
  its keypoints, so nothing drifts out of alignment.
- **Environmental simulation**: lighting changes, shadows, glare.
- **Occlusion simulation**: randomly occludes parts of an object and automatically updates
  that object's keypoint visibility flags to match — so occluded synthetic data is labeled
  correctly, not just visually plausible.

See the
[Synthetic Engine README](https://github.com/thippeswammy/AutoSegmentor/blob/master/DatasetManager/SyntheticEngine/README.md)
for the full configuration reference.

## Where this fits

```text
Annotate & verify (main app)
        │
        ▼
  Ctrl+E → YOLO Dataset Manager  →  train/valid/test + data.yaml
                                          │
                                          ▼ (optional, more data)
                                   Synthetic Engine  →  augmented dataset
```

Both tools are independent of each other and of the annotation pipeline — you can run the
YOLO exporter without ever touching the Synthetic Engine, and the Synthetic Engine's
requirements are met by the same project virtual environment (no separate setup).
