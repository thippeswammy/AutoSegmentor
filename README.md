# SAM2 Video Processing Pipeline

This repository contains a comprehensive video processing pipeline built around the SAM2 segmentation model. The pipeline automates the extraction of frames from videos, allows interactive annotation, performs mask prediction, overlays masks on original images, and compiles the results into output videos.

## Features

- **Frame Extraction**: Extracts frames from input videos to images.
- **Interactive Annotation**: Annotate frames with points and class labels via a GUI.
- **Batch Processing**: Processes frames in configurable batches for efficient GPU usage.
- **Mask Prediction**: Uses the SAM2 model to predict segmentation masks for annotated objects.
- **Mask Overlay**: Overlays predicted masks on original images for visual verification.
- **Verification Step**: Allows user to verify and curate results before finalizing output.
- **Video Compilation**: Creates output videos from original images, predicted masks, and overlays.

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [numpy](https://numpy.org/)
- [GPUtil](https://github.com/anderskm/gputil)
- [tqdm](https://tqdm.github.io/)
- [pygetwindow](https://github.com/asweigart/pygetwindow)
- `sam2` library and model checkpoint (`sam2_hiera_large.pt`)
- Platform dependencies for GUI (may require X11 on Linux)

Install dependencies with:

```bash
pip install torch torchvision opencv-python numpy GPUtil tqdm pygetwindow
```

## Usage

### 1. Prepare Inputs

- Place your videos in the input directory (e.g., `./VideoInputs/Video1.mp4`, `./VideoInputs/Video2.mp4`, ...).
- Ensure the `sam2_hiera_large.pt` checkpoint and `sam2_hiera_l.yaml` config are available in the paths specified by the script (see arguments below).

### 2. Run the Pipeline

You can run the pipeline via command line:

```bash
python sam2_video_predictor_long.py \
  --video_start 1 \
  --video_end 2 \
  --prefix Img \
  --batch_size 120 \
  --fps 30 \
  --delete no \
```

**Argument Descriptions:**
- `--video_start`: Starting video number (inclusive).
- `--video_end`: Ending video number (exclusive).
- `--prefix`: Prefix for output filenames (default: 'Img').
- `--batch_size`: Number of frames to process per batch (default: 120).
- `--fps`: Frames per second for output videos (default: 30).
- `--delete`: Whether to auto-delete working directory (`yes`/`no`).
- `--working_dir_name`: Name for the working directory (default: 'working_dir').
- `--video_path_template`: Template for video file paths (default: `./VideoInputs/Video{}.mp4`).
- `--images_extract_dir`: Directory to extract images.
- `--temp_processing_dir`: Directory for intermediate batch processing.
- `--rendered_dir`: Directory for rendered masks.
- `--overlap_dir`: Directory for mask overlays.
- `--verified_img_dir`: Directory for verified original images.
- `--verified_mask_dir`: Directory for verified mask images.
- `--final_video_path`: Directory to save the final output videos.

### 3. Annotation and Verification

- During processing, the script will open a window for interactive annotation. Use mouse clicks to annotate, number keys to change class, `u` to undo, and `Enter` to finish annotating a frame.
- After processing, overlays of masks on images are generated for verification.
- The pipeline prompts you to verify overlays before copying the results for final video creation.

### 4. Output

- Verified original images, masks, and overlay images are stored in the specified verified/output folders.
- Three `.mp4` videos are created:
  - Original frames video
  - Mask prediction video
  - Overlay video

## PyInstaller Packaging

To create a standalone executable with PyInstaller:

```bash
pyinstaller --name sam2_video_predictor_long \
  --add-data "../checkpoints/sam2_hiera_large.pt;checkpoints" \
  --add-data "../sam2_configs/sam2_hiera_l.yaml;sam2_configs" \
  --add-data "../sam2_configs;sam2_configs" \
  --hidden-import torch \
  --hidden-import cv2 \
  --hidden-import numpy \
  --hidden-import GPUtil \
  --hidden-import sam2 \
  --hidden-import sam2.sam2_configs \
  --collect-all sam2 \
  --onefile sam2_video_predictor_long.py
```

Adjust paths as needed for your project structure.

## Keyboard Controls for Annotation

- **1-9**: Change current class label.
- **Left Click**: Add point for current class.
- **Right Click**: Add point for current class (different visualization).
- **u**: Undo last point.
- **r**: Reset current frame points.
- **Enter**: Finish annotating the current frame.
- **q**: Quit annotation.

## Directory Structure

```
.
├── VideoInputs/
│   └── Video1.mp4, Video2.mp4, ...
├── checkpoints/
│   └── sam2_hiera_large.pt
├── sam2_configs/
│   └── sam2_hiera_l.yaml
├── working_dir/
│   ├── images/
│   ├── temp/
│   ├── render/
│   ├── overlap/
│   └── verified/
│       ├── images/
│       └── mask/
├── outputs/
│   ├── OrgVideo1.mp4
│   ├── MaskVideo1.mp4
│   └── OverlappedVideo1.mp4
└── sam2_video_predictor_long.py
```

## Notes

- Ensure your GPU drivers and environment are set up for PyTorch and OpenCV.
- The pipeline is interactive; user input is required at several stages.
- For best results, use videos with consistent resolution and naming.

## Acknowledgements

- [SAM2](https://github.com/facebookresearch/segment-anything) by Meta AI
- PyTorch, OpenCV, and the open-source community

```
**For questions or bug reports, please open an issue.**
