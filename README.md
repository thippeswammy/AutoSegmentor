# SAM2 Video Processing Pipeline

This repository contains a comprehensive video processing pipeline built around the SAM2 (Segment Anything Model 2) for video segmentation, with additional functionality to create YOLO-compatible datasets for training. The pipeline automates frame extraction, allows interactive point-based annotation, performs mask prediction, overlays masks on original images, compiles results into output videos, and processes the resulting images and masks into a YOLO dataset format.


## Features
- **Frame Extraction**: Extracts frames from input videos into images using `FrameExtractor`.
- **Interactive Annotation**: Annotate frames with points and class labels via an OpenCV-based GUI.
- **Batch Processing**: Processes frames in configurable batches for efficient GPU usage.
- **Mask Prediction**: Uses the SAM2 model to predict segmentation masks for annotated objects.
- **Mask Overlay**: Overlays predicted masks on original images for visual verification using `ImageOverlayProcessor`.
- **Verification Step**: Prompts user verification of overlays before finalizing output.
- **Video Compilation**: Creates output videos (original, masks, and overlays) using `VideoCreator`.
- **Dataset Creation**: Converts SAM2-generated masks and images into YOLO format with augmentations using `YoloProcessor`.
- **Directory Management**: Manages temporary and output directories with optional cleanup.

## Requirements
- Python 3.8+
- PyTorch (with CUDA support for GPU acceleration)
- OpenCV (`opencv-python`)
- NumPy
- GPUtil
- tqdm
- pygetwindow
- Pillow (`PIL`)
- SAM2 library and model checkpoint (`sam2_hiera_large.pt`)
- Custom modules: `FrameExtractor`, `ImageCopier`, `ImageOverlayProcessor`, `VideoCreator`, `create_yolo_structure`
- Platform dependencies for GUI (e.g., X11 on Linux for OpenCV windows)

Install dependencies with:
```bash
pip install torch torchvision opencv-python numpy GPUtil tqdm pygetwindow pillow
```

Ensure the SAM2 model checkpoint (`sam2_hiera_large.pt`) and configuration (`sam2_hiera_l.yaml`) are placed in the `checkpoints` and `sam2_configs` directories, respectively. The `create_yolo_structure` module is required for dataset creation and must be available in the `DatasetManager/YolovDatasetManager` directory.

## Usage

### 1. Prepare Inputs
- Place input videos in the specified directory (e.g., `./VideoInputs/Video1.mp4`, `./VideoInputs/Video2.mp4`, ...).
- Ensure the SAM2 model checkpoint (`sam2_hiera_large.pt`) and configuration (`sam2_hiera_l.yaml`) are in the correct paths as specified by the script's arguments.
- Verify that custom modules (`FrameExtractor`, `ImageCopier`, `ImageOverlayProcessor`, `VideoCreator`, `create_yolo_structure`) are available in the project directory.

### 2. Running the SAM2 Video Processing Pipeline
Navigate to the project directory:
```bash
cd ./segment-anything-3
```

Run the SAM2 pipeline to generate images, masks, and videos:
```bash
python sam2_video_predictor.py ^
  --video_start 1 ^
  --video_end 2 ^
  --prefix Img ^
  --batch_size 120 ^
  --fps 30 ^
  --delete no ^
  --working_dir_name working_dir ^
  --video_path_template .\VideoInputs\Video{}.mp4 ^
  --images_extract_dir .\working_dir\images ^
  --temp_processing_dir .\working_dir\temp ^
  --rendered_dir .\working_dir\render ^
  --overlap_dir .\working_dir\overlap ^
  --verified_img_dir .\working_dir\verified\images ^
  --verified_mask_dir .\working_dir\verified\mask ^
  --final_video_path .\outputs

```

#### Argument Descriptions
- `--video_start`: Starting video number (inclusive, default: 99).
- `--video_end`: Ending video number (exclusive, default: 1).
- `--prefix`: Prefix for output filenames (default: `Img`).
- `--batch_size`: Number of frames to process per batch (default: 120).
- `--fps`: Frames per second for output videos (default: 30).
- `--delete`: Auto-delete working directory without prompt (`yes`/`no`, default: `yes`).
- `--working_dir_name`: Base directory for working files (default: `working_dir`).
- `--video_path_template`: Template for video file paths (default: `./VideoInputs/Video{}.mp4`).
- `--images_extract_dir`: Directory for extracted frames (default: `./working_dir/images`).
- `--temp_processing_dir`: Directory for temporary batch processing (default: `./working_dir/temp`).
- `--rendered_dir`: Directory for rendered masks (default: `./working_dir/render`).
- `--overlap_dir`: Directory for mask overlays (default: `./working_dir/overlap`).
- `--verified_img_dir`: Directory for verified original images (default: `./working_dir/verified/images`).
- `--verified_mask_dir`: Directory for verified mask images (default: `./working_dir/verified/mask`).
- `--final_video_path`: Directory for output videos (default: `./outputs`).

### 3. Annotation and Verification
- **Interactive Annotation**:
  - The script opens an OpenCV window (`SAM2 Annotation Tool`) for annotating frames.
  - Use mouse clicks to add points and keyboard controls to manage annotations (see below).
  - After annotating a frame, press `Enter` to save points and proceed to the next batch.
- **Verification**:
  - Overlays of masks on original images are generated in `overlap_dir` for review.
  - The pipeline prompts for verification (unless `--delete yes` is specified) before copying verified images and masks to `verified_img_dir` and `verified_mask_dir`.
  - Three output videos are created: original frames, mask predictions, and overlays.

### 4. Creating a YOLO Dataset
The `YoloProcessor` script processes the SAM2-generated images and masks (from `working_dir/images` and `working_dir/render`) into a YOLO-compatible dataset with augmentations. It creates a dataset structure with `train`, `valid`, and `test` splits, including images and YOLO-format label files.

Run the dataset creation script:
```bash
cd ..\DatasetManager\YolovDatasetManager\DatasetCreatere.py
python DatasetCreatere.py
```

The script uses a configuration dictionary (defined in `DatasetCreatere.py`) to specify parameters such as:
- `dataset_path`: Path to the SAM2 working directory (e.g., `./segment-anything-3/working_dir`).
- `SOURCE_mask_folder_name`: Directory containing SAM2-generated masks (e.g., `render`).
- `SOURCE_original_folder_name`: Directory containing original images (e.g., `images`).
- `augment_times`: Number of augmentations per image (default: 10).
- `train_split`, `val_split`, `test_split`: Proportions for dataset splits (e.g., 0.9, 0.1, 0.0).
- `class_to_id`: Mapping of class names to IDs (e.g., `{'road': 0}`).
- `color_to_label`: Mapping of mask colors to labels (e.g., `{(255, 255, 255): 0}`).
- `dataset_saving_working_dir`: Output directory for the YOLO dataset (e.g., `./DatasetManager`).
- `Keep_val_dataset_original`: Whether to keep validation images unaugmented (default: `True`).

The script applies augmentations (e.g., Gaussian blur, noise, color jitter) and converts masks to YOLO-format polygon annotations.

### 5. Output
- **SAM2 Pipeline Outputs**:
  - Verified images and masks in `verified_img_dir` and `verified_mask_dir`.
  - Videos in `final_video_path`:
    - `OrgVideo<video_number>.mp4`: Original frames.
    - `MaskVideo<video_number>.mp4`: Predicted masks.
    - `OverlappedVideo<video_number>.mp4`: Overlaid images.
- **YOLO Dataset Outputs**:
  - Dataset structure in `dataset_saving_working_dir/<folder_name>` (e.g., `./DatasetManager/road`):
    - `train/images/`, `train/labels/`: Training images and YOLO-format labels.
    - `valid/images/`, `valid/labels/`: Validation images and labels.
    - `test/images/`, `test/labels/`: Test images and labels (if `test_split` > 0).

## Keyboard and Mouse Controls for Annotation
- **1-9**: Change the current class label (1 to 9).
- **Left Click**: Add a foreground point for the current class and instance.
- **Right Click**: Add a background point for the current class and instance (visualized differently).
- **u**: Undo the last added point.
- **r**: Reset all points for the current frame.
- **Tab**: Increment the instance ID.
- **Shift + Tab**: Decrement the instance ID.
- **f**: Jump to a specific frame index for annotation.
- **Enter**: Save points and labels for the current frame and proceed.
- **q**: Quit the annotation process.

A zoom view window (`Zoom View`) shows a magnified view of the area around the mouse cursor during annotation.

## Directory Structure

```
SAM2/
├── DatasetManager/
│   └── YolovDatasetManager/
│       └── create_yolo_structure.py
│       └── DatasetCreatere.py
├── checkpoints/
│   └── sam2_hiera_large.pt
├── sam2/
│   └── __init__.py
├── sam2_configs/
│   └── sam2_hiera_l.yaml
├── segment-anything-3/
│   ├── VideoInputs/
│   │   └── Video1.mp4, Video2.mp4, ...
│   ├── working_dir/
│   │   ├── images/
│   │   ├── temp/
│   │   ├── render/
│   │   ├── overlap/
│   │   └── verified/
│   │       ├── images/
│   │       └── mask/
│   ├── outputs/
│   │   ├── OrgVideo1.mp4
│   │   ├── MaskVideo1.mp4
│   │   └── OverlappedVideo1.mp4
│   ├── sam2_video_predictor.py
│   └── UserPrompts/
│           └── points_labels_*.json
```

## Notes
- Ensure GPU drivers and PyTorch are configured for CUDA if using a GPU.
- The SAM2 pipeline requires user interaction for annotation and verification unless `--delete yes` is specified.
- Video filenames must follow the pattern specified in `--video_path_template` (e.g., `Video1.mp4`).
- Frame filenames must follow the pattern `<prefix>_<frame_index>.jpg` (e.g., `Img_00000.jpg`).
- The `YoloProcessor` script expects masks in `render` and images in `images` directories from the SAM2 pipeline.
- The SAM2 model, custom modules (`FrameExtractor`, etc.), and `create_yolo_structure` are critical dependencies; ensure they are correctly installed and accessible.

## Troubleshooting
- **Missing SAM2 Checkpoint**: Ensure `sam2_hiera_large.pt` and `sam2_hiera_l.yaml` are in the `checkpoints` and `sam2_configs` directories.
- **OpenCV Window Issues**: Verify that your system supports OpenCV's GUI (e.g., X11 on Linux or a compatible display server).
- **Invalid Frame Filenames**: Ensure extracted frames follow the expected naming pattern (`<prefix>_<frame_index>.jpg`).
- **GPU Errors**: Check GPU availability with `GPUtil.showUtilization()` and ensure PyTorch is CUDA-enabled.
- **Missing Custom Modules**: Verify that `FrameExtractor.py`, `ImageCopier.py`, `ImageOverlayProcessor.py`, `VideoCreator.py`, and `create_yolo_structure.py` are in the project directory.
- **YOLO Dataset Issues**: Ensure the `color_to_label` mapping matches the mask colors produced by `sam2_video_predictor.py`. Check that source directories (`images` and `render`) exist and contain valid files.


## Acknowledgements

- [SAM2](https://github.com/facebookresearch/segment-anything) by Meta AI
- PyTorch, OpenCV, and the open-source community

```
**For questions or bug reports, please open an issue.**
