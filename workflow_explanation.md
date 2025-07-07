# SAM2 Video Processing Pipeline Workflow Explanation

This document provides a detailed, step-by-step explanation of the workflow for the SAM2 video processing pipeline, focusing on how the main script (`sam3/sam3_video_predictor_demo.py`) orchestrates the process and interacts with other files in the `sam3` package. The pipeline automates video frame extraction, interactive annotation, mask prediction, overlay generation, verification, video creation, and YOLO dataset creation. Each file's purpose, inputs, outputs, and interactions are described to clarify the data flow and processing steps.

## Overview
The pipeline processes input videos (e.g., `sam3/inputs/VideoInputs/Video1.mp4`) to generate segmentation masks using the SAM2 model, produce verification overlays, create output videos, and optionally generate a YOLO-compatible dataset. The workflow is driven by `sam3_video_predictor_demo.py`, which loads configuration parameters and coordinates tasks through `sam3/utils/pipeline.py` and other modular components.

## Workflow Steps and File Interactions

### 1. Configuration Loading (`sam3_video_predictor_demo.py`, `sam3/inputs/config/default_config.yaml`)
- **Purpose**: `sam3_video_predictor_demo.py` is the main entry point, responsible for loading the configuration and iterating through videos for processing.
- **Process**:
  - The `load_config` function reads `sam3/inputs/config/default_config.yaml`, which defines parameters like `video_start`, `video_end`, `prefix`, `batch_size`, `fps`, `delete`, and directory paths (e.g., `images_extract_dir`, `rendered_dir`).
  - It validates required keys and normalizes the `delete` parameter (`true`/`false` to `'yes'`/`'no'`).
- **Outputs**: A configuration dictionary passed to `run_pipeline` for each video.
- **Interactions**:
  - Uses `logger_config.py` to log errors or info (e.g., missing config keys, file not found).
  - Calls `FileManager.py` (via `run_pipeline`) to manage directories.
- **Data Flow**: The config dictionary sets parameters for all subsequent steps, ensuring consistent file paths and processing settings.

### 2. Directory Management (`FileManager.py`, `sam3_video_predictor_demo.py`)
- **Purpose**: `FileManager.py` provides utilities for directory and file management, used by `sam3_video_predictor_demo.py` to prepare the working directory.
- **Process**:
  - `sam3_video_predictor_demo.py` checks if `working_dir_name` (e.g., `sam3/working_dir`) exists.
  - If `delete` is `'yes'`, it calls `FileManager.clear_directory` to remove the directory; if `'no'`, it prompts the user for confirmation.
  - After processing, it repeats the cleanup process based on `delete`.
- **Outputs**: A clean `sam3/working_dir` for intermediate files (e.g., `images`, `temp`, `render`, `overlap`, `verified`).
- **Interactions**:
  - `FileManager.py`: Uses `clear_directory` to delete directory contents and `ensure_directory` to create necessary folders.
  - `logger_config.py`: Logs directory clearing actions or user decisions.
- **Data Flow**: Ensures a fresh working directory for each video, preventing conflicts from previous runs.

### 3. Frame Extraction (`FrameExtractor.py`, `sam2_video_predictor.py`)
- **Purpose**: `FrameExtractor.py` extracts frames from input videos and saves them as images.
- **Process**:
  - `sam3_video_predictor_demo.py` calls `run_pipeline`, which initializes `SAM2VideoProcessor` (from `sam2_video_predictor.py`).
  - `SAM2VideoProcessor` creates a `FrameExtractor` instance, passing config parameters (e.g., `video_number`, `prefix`, `video_path_template`, `images_extract_dir`, `images_ending_count`).
  - `FrameExtractor.run()` opens the video (e.g., `sam3/inputs/VideoInputs/Video1.mp4`), extracts frames up to `images_ending_count` (if specified), and saves them as `<prefix><video_number>_<frame_index>.jpeg` (e.g., `Img1_00001.jpeg`) in `sam3/working_dir/images`.
- **Outputs**: Image files in `sam3/working_dir/images`.
- **Interactions**:
  - `FileManager.py`: Uses `ensure_directory` to create `sam3/working_dir/images`.
  - `logger_config.py`: Logs errors (e.g., video file not found).
- **Data Flow**: Frames are saved for subsequent annotation and mask prediction.

### 4. Batch Frame Management (`FrameHandler.py`, `sam2_video_predictor.py`)
- **Purpose**: `FrameHandler.py` manages frame paths and copies batches of frames to a temporary directory for processing.
- **Process**:
  - `SAM2VideoProcessor` (in `sam2_video_predictor.py`) initializes `FrameHandler` with `frames_directory` (`sam3/working_dir/images`) and `temp_directory` (`sam3/working_dir/temp`).
  - `FrameHandler.get_frame_files` retrieves sorted frame paths using `FileManager.get_frame_paths`.
  - `FrameHandler.move_and_copy_frames` copies a batch of frames (size defined by `batch_size`) to `sam3/working_dir/temp`, renaming them as `<frame_index>.jpg`.
- **Outputs**: Batch frames in `sam3/working_dir/temp`.
- **Interactions**:
  - `FileManager.py`: Uses `ensure_directory` and `clear_directory` for `sam3/working_dir/temp`.
  - `logger_config.py`: Logs file copying and naming issues.
- **Data Flow**: Prepares batches for annotation and mask prediction, ensuring efficient processing.

### 5. Model Initialization (`SAM2Config.py`, `SAM2Model.py`, `sam2_video_predictor.py`)
- **Purpose**: `SAM2Config.py` and `SAM2Model.py` configure and initialize the SAM2 model.
- **Process**:
  - `SAM2VideoProcessor` (in `sam2_video_predictor.py`) creates a `SAM2Config` instance with parameters like `video_number`, `batch_size`, `prefix`, `label_colors`, and paths to `sam2_hiera_large.pt` and `sam2_hiera_l.yaml`.
  - `SAM2Config` uses `FileManager.get_resource_path` to resolve paths and `ensure_directory` for output directories.
  - `SAM2Model` initializes the SAM2 predictor using `build_sam2_video_predictor`, setting the device (CPU/GPU) and memory settings.
- **Outputs**: A configured SAM2 predictor for mask generation.
- **Interactions**:
  - `FileManager.py`: For path resolution and directory creation.
  - `logger_config.py`: Logs device selection and initialization status.
- **Data Flow**: Provides the SAM2 model for mask prediction in subsequent steps.

### 6. Interactive Annotation (`AnnotationManager.py`, `UserInteraction.py`, `sam2_video_predictor.py`)
- **Purpose**: `AnnotationManager.py` and `UserInteraction.py` handle user annotations for frames, saving points and labels.
- **Process**:
  - `SAM2VideoProcessor.run()` checks for existing annotations via `AnnotationManager.check_data_sufficiency`.
  - If annotations are needed, `UserInteraction.collect_user_points` opens an OpenCV window (`SAM2 Annotation Tool`) for the first frame of each batch.
  - Users add points (left-click for foreground, right-click for background), select class labels (1-9), and manage instance IDs (`Tab`/`Shift+Tab`). `UserInteraction.click_event` handles mouse events, encoding labels as `class_id * 1000 + instance_id`.
  - `AnnotationManager` saves annotations to `sam3/inputs/UserPrompts/points_labels_<prefix><video_number>.json`.
- **Outputs**: JSON files with points, labels, and frame indices.
- **Interactions**:
  - `SAM2Config.py`: Provides `label_colors` for visualization.
  - `sam2_video_predictor.py`: Uses `SAM2VideoProcessor.user_prompt_adder` to preview masks during annotation.
  - `logger_config.py`: Logs annotation actions and errors.
- **Data Flow**: Annotations are used for mask prediction and stored for reuse.

### 7. Mask Prediction (`MaskProcessor.py`, `sam2_video_predictor.py`)
- **Purpose**: `MaskProcessor.py` generates segmentation masks using the SAM2 model.
- **Process**:
  - `SAM2VideoProcessor.run()` calls `MaskProcessor.generate_mask` for each batch.
  - `MaskProcessor` initializes the SAM2 predictor with `SAM2Model`, loads batch frames from `sam3/working_dir/temp`, and applies annotations via `prompt_encoding` or `auto_prompt_encoding` (for propagating previous masks).
  - `MaskProcessor.binary_mask_2_color_mask` converts binary masks to color masks using `label_colors` from `SAM2Config`, saving them as `<prefix><video_number>_<index>.png` in `sam3/working_dir/render`.
- **Outputs**: Color mask images in `sam3/working_dir/render`.
- **Interactions**:
  - `SAM2Model.py`: Provides the SAM2 predictor.
  - `SAM2Config.py`: Supplies configuration (e.g., `label_colors`).
  - `FileManager.py`: Ensures output directory exists.
  - `logger_config.py`: Logs mask generation progress.
- **Data Flow**: Masks are generated for overlay and dataset creation.

### 8. Overlay Generation (`ImageOverlayProcessor.py`, `pipeline.py`)
- **Purpose**: `ImageOverlayProcessor.py` creates overlaid images for verification.
- **Process**:
  - `run_pipeline` (called by `sam3_video_predictor_demo.py`) initializes `ImageOverlayProcessor` with `images_extract_dir`, `rendered_dir`, and `overlap_dir`.
  - `ImageOverlayProcessor.process_all_images` loads original images and masks, blends them (using `overlay_mask_on_image`), and saves results in `sam3/working_dir/overlap`.
- **Outputs**: Overlaid images in `sam3/working_dir/overlap`.
- **Interactions**:
  - `FileManager.py`: Ensures `sam3/working_dir/overlap` exists.
  - `logger_config.py`: Logs processing progress.
- **Data Flow**: Overlaid images are used for verification.

### 9. Verification (`ImageCopier.py`, `pipeline.py`)
- **Purpose**: `ImageCopier.py` copies verified images and masks after user review.
- **Process**:
  - `run_pipeline` prompts for verification if `delete` is `'no'`, asking users to confirm overlay accuracy.
  - If confirmed, `ImageCopier.copy_images` copies images and masks from `sam3/working_dir/images` and `sam3/working_dir/render` to `sam3/working_dir/verified/images` and `sam3/working_dir/verified/mask`, filtering based on `sam3/working_dir/overlap`.
- **Outputs**: Verified images and masks in `sam3/working_dir/verified`.
- **Interactions**:
  - `FileManager.py`: Ensures output directories exist.
  - `logger_config.py`: Logs copying and user input.
- **Data Flow**: Verified files are used for video creation.

### 10. Video Creation (`VideoCreator.py`, `pipeline.py`)
- **Purpose**: `VideoCreator.py` generates output videos from verified images and masks.
- **Process**:
  - `run_pipeline` initializes `VideoCreator` with `verified_img_dir`, `verified_mask_dir`, `overlap_dir`, and `final_video_path`.
  - `VideoCreator.run` creates three videos (`OrgVideo<video_number>.mp4`, `MaskVideo<video_number>.mp4`, `OverlappedVideo<video_number>.mp4`) in `sam3/outputs` using the specified `fps`.
- **Outputs**: Videos in `sam3/outputs`.
- **Interactions**:
  - `FileManager.py`: Ensures `sam3/outputs` exists.
  - `logger_config.py`: Logs video creation status.
- **Data Flow**: Videos are the final pipeline output.

### 11. Dataset Creation (`DatasetCreator.py`, `create_yolo_structure.py`)
- **Purpose**: `DatasetCreator.py` converts images and masks into a YOLO dataset with augmentations.
- **Process**:
  - Run separately via `python DatasetCreator.py` in `DatasetManager/YolovDatasetManager`.
  - Uses a configuration dictionary to process images/masks from `sam3/working_dir/images` and `sam3/working_dir/render`, applying augmentations (e.g., color jitter, Gaussian blur) and creating `train`, `valid`, and `test` splits in `DatasetManager/road_dataset`.
  - `create_yolo_structure.py` organizes the dataset structure and converts masks to YOLO polygon annotations.
- **Outputs**: YOLO dataset with images and `.txt` label files.
- **Interactions**:
  - Uses `color_to_label` aligned with `SAM2Config.label_colors`.
  - `logger_config.py`: Logs dataset creation (assumed, not provided).
- **Data Flow**: Generates a dataset for training object detection models.

## Notes
- The workflow is modular, with each file handling a specific task.
- `sam3_video_predictor_demo.py` ensures seamless integration via `run_pipeline`.
- Logging (`logger_config.py`) provides debugging and monitoring throughout.