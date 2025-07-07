# SAM2 Video Processing Pipeline

This repository provides a comprehensive pipeline for video segmentation using the SAM2 (Segment Anything Model 2) model by Meta AI, with functionality to create YOLO-compatible datasets for training. The pipeline automates frame extraction, supports interactive point-based annotation, performs mask prediction, overlays masks on original images, compiles results into output videos, and processes images and masks into a YOLO dataset format with augmentations. The codebase is organized under the `sam3` package, with modular components for file management, model handling, user interaction, and pipeline execution. Future enhancements will include support for additional dataset formats (e.g., COCO, Pascal VOC) for broader compatibility.

## Features
- **Frame Extraction**: Extracts frames from input videos into images using `FrameExtractor`.
- **Interactive Annotation**: Annotates frames with points and class labels via an OpenCV-based GUI, managed by `AnnotationManager` and `UserInteraction`.
- **Batch Processing**: Processes frames in configurable batches for efficient GPU usage, handled by `FrameHandler`.
- **Mask Prediction**: Uses SAM2 to predict segmentation masks for annotated objects, processed by `MaskProcessor`.
- **Mask Overlay**: Overlays predicted masks on original images for verification using `ImageOverlayProcessor`.
- **Verification Step**: Prompts user verification of overlays before finalizing output.
- **Video Compilation**: Creates output videos (original, masks, and overlays) using `VideoCreator`.
- **Dataset Creation**: Converts SAM2-generated masks and images into YOLO format with augmentations using `DatasetCreator`, supporting multiple classes (e.g., `road`, `cars`, `trucks`).
- **Directory Management**: Manages temporary and output directories with optional cleanup using `FileManager`.
- **Model Configuration**: Configures SAM2 model settings via `SAM2Config` and manages model initialization with `SAM2Model`.

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
- Custom modules: `FileManager`, `FrameExtractor`, `FrameHandler`, `MaskProcessor`, `ImageCopier`, `ImageOverlayProcessor`, `VideoCreator`, `SAM2Config`, `SAM2Model`, `sam2_video_predictor`, `AnnotationManager`, `UserInteraction`, `pipeline`, `create_yolo_structure`
- Platform dependencies for GUI (e.g., X11 on Linux or compatible display server on Windows)

Install dependencies with:
```bash
pip install torch torchvision opencv-python numpy GPUtil tqdm pygetwindow pillow禁止

Ensure the SAM2 model checkpoint (`sam2_hiera_large.pt`) and configuration (`sam2_hiera_l.yaml`) are placed in the `checkpoints` and `sam2_configs` directories, respectively. The `create_yolo_structure` module, required for dataset creation, must be in the `DatasetManager/YolovDatasetManager` directory.

## Usage

### 1. Prepare Inputs
- Place input videos in the specified directory (e.g., `sam3/inputs/VideoInputs/Video1.mp4`, `sam3/inputs/VideoInputs/Video2.mp4`, ...).
- Ensure the SAM2 model checkpoint (`sam2_hiera_large.pt`) and configuration (`sam2_hiera_l.yaml`) are in the `checkpoints` and `sam2_configs` directories.
- Verify that all custom modules are available in the `sam3/utils` directory structure.

### 2. Running the SAM2 Video Processing Pipeline

The `sam3/sam3_video_predictor_demo.py` script orchestrates the video processing workflow, configured via a YAML file.

**Prerequisites:**
- Ensure input videos are in the location specified by `video_path_template` in the configuration file (e.g., `sam3/inputs/VideoInputs/Video{}.mp4`).
- Verify that the SAM2 model checkpoint and configuration are in the `checkpoints` and `sam2_configs` directories, respectively.

**Execution:**

Navigate to the root directory of the project (where `sam3` is located):
```bash
cd path/to/your/project_root
```

Run the main pipeline script:
```bash
python sam3/sam3_video_predictor_demo.py
```
The script loads parameters from `sam3/inputs/config/default_config.yaml` by default. To use a different configuration file, modify the `config_path` variable in the `load_config` function in `sam3/sam3_video_predictor_demo.py`.

#### Configuration (`sam3/inputs/config/default_config.yaml`)

All operational parameters are defined in `sam3/inputs/config/default_config.yaml`. Users **must** modify this file to control:
- Input video range (`video_start`, `video_end`)
- File naming (`prefix`)
- Processing parameters (`batch_size`, `fps`)
- Directory paths (`working_dir_name`, `images_extract_dir`, `rendered_dir`, etc.)
- Cleanup behavior (`delete`)

#### Configuration Details

Key parameters in `sam3/inputs/config/default_config.yaml`:
- `video_start` (int): Starting video sequence number (e.g., 1 for `Video1.mp4`).
- `video_end` (int): Number of videos to process (e.g., 2 for `Video1.mp4` and `Video2.mp4`).
- `prefix` (str): Prefix for naming intermediate files (e.g., "Img").
- `batch_size` (int): Number of frames processed per batch.
- `fps` (int): Frames per second for output videos.
- `delete` (bool): Controls cleanup of `working_dir_name` (`true` for automatic deletion, `false` for user prompt).
- `working_dir_name` (str): Main directory for intermediate files (e.g., "working_dir").
- `video_path_template` (str): Path template for input videos (e.g., `sam3/inputs/VideoInputs/Video{}.mp4`).
- `images_extract_dir` (str): Directory for extracted frames (e.g., `sam3/working_dir/images`).
- `temp_processing_dir` (str): Directory for temporary files (e.g., `sam3/working_dir/temp`).
- `rendered_dir` (str): Directory for rendered masks (e.g., `sam3/working_dir/render`).
- `overlap_dir` (str): Directory for overlaid images (e.g., `sam3/working_dir/overlap`).
- `verified_img_dir` (str): Directory for verified original images (e.g., `sam3/working_dir/verified/images`).
- `verified_mask_dir` (str): Directory for verified masks (e.g., `sam3/working_dir/verified/mask`).
- `final_video_path` (str): Directory for output videos (e.g., `sam3/outputs`).
- `images_ending_count` (int): Number of digits for frame numbering (e.g., 5 for `00001.jpg`).

Modify these values in `sam3/inputs/config/default_config.yaml` to customize execution.

#### Core Workflow of `sam3_video_predictor_demo.py`

The script performs:
1. **Loads Configuration**: Reads parameters from the YAML file using `load_config`.
2. **Iterates Through Videos**: Processes videos from `video_start` to `video_end`.
3. **Manages Working Directory**: Clears `working_dir_name` based on `delete` setting using `FileManager`.
4. **Executes Processing Pipeline**: Calls `run_pipeline` (from `sam3/utils/pipeline.py`), which:
   - Extracts frames using `FrameExtractor`.
   - Manages batches with `FrameHandler`.
   - Generates masks with `MaskProcessor` and `SAM2VideoProcessor`.
   - Overlays masks with `ImageOverlayProcessor`.
   - Copies verified images/masks with `ImageCopier`.
   - Creates videos with `VideoCreator`.
5. **Post-Processing Cleanup**: Manages `working_dir_name` based on `delete` setting.

### 3. Annotation

The `sam3_video_predictor_demo.py` script orchestrates the annotation and verification processes as part of the SAM2 video processing pipeline. It loads configuration parameters from `sam3/inputs/config/default_config.yaml` and delegates tasks to `run_pipeline` (from `sam3/utils/pipeline.py`), which coordinates frame extraction, annotation, mask prediction, overlay generation, verification, and video creation.

- **Interactive Annotation**:
  - **Process**: The `SAM2VideoProcessor` (from `sam3/utils/Model/sam2_video_predictor.py`) uses `UserInteraction` and `AnnotationManager` (from `sam3/utils/UserUI/`) to provide an OpenCV-based GUI for annotating frames. Users add foreground/background points and class labels via mouse clicks and keyboard controls. Annotations are saved as JSON files in `sam3/inputs/UserPrompts/points_labels_<prefix><video_number>.json`.
  - **Details**: Users select points (left-click for foreground, right-click for background), assign class labels (1-9), and manage instance IDs (`Tab`/`Shift+Tab`). A `Zoom View` window shows a magnified area around the cursor. Annotations are used by `MaskProcessor` to generate segmentation masks via the SAM2 model.

### 4. Creating a YOLO Dataset
The `DatasetCreator.py` script processes images and masks from `sam3/working_dir/images` and `sam3/working_dir/render` into a YOLO-compatible dataset with augmentations, supporting multiple classes (e.g., `road`, `cars`, `trucks`).

Run the dataset creation script:
```bash
cd DatasetManager/YolovDatasetManager
python DatasetCreator.py
```

#### Configuration for `DatasetCreator.py`
Sample configuration in `DatasetCreator.py`:
```python
CONFIG = {
    "dataset_path": r"../sam3/working_dir",
    "SOURCE_mask_folder_name": "render",
    "SOURCE_original_folder_name": "images",
    "SOURCE_mask_type_ext": ".png",
    "SOURCE_img_type_ext": ".jpeg",
    "augment_times": 10,
    "test_split": 0.0,
    "val_split": 0.1,
    "train_split": 0.9,
    "Keep_val_dataset_original": True,
    "num_threads": os.cpu_count() - 2,
    "class_to_id": {
        "road": 0,
        "cars": 1,
        "trucks": 2
    },
    "color_to_label": {
        (255, 255, 255): 0,  # road
        (0, 0, 255): 1,      # cars
        (255, 0, 0): 2       # trucks
    },
    "class_names": ["road", "cars", "trucks"],
    "dataset_saving_working_dir": r".\DatasetManager",
    "folder_name": "road_dataset",
    "DESTINATION_img_type_ext": ".jpg",
    "DESTINATION_label_type_ext": ".txt",
    "FromDataType": "",
    "ToDataTypeFormate": ""
}
```

#### Multiple Class Support and Instance ID Management
- **Multiple Classes**: Maps mask colors to class IDs in `color_to_label`, aligned with `label_colors` in `SAM2Config`.
- **Instance ID Management**: Uses `Tab`/`Shift+Tab` to increment/decrement instance IDs per class, encoding labels as `class_id * 1000 + instance_id`.
- **YOLO Format**: Converts masks to polygon annotations (e.g., `0 0.1 0.2 ...` for `road`).

#### Augmentations
- Color jitter, Gaussian blur, average blur, Gaussian noise, salt-and-pepper noise.

#### Future Dataset Formats
- Support for COCO and Pascal VOC formats is planned.

### 5. Output
- **SAM2 Pipeline Outputs**:
  - Verified images and masks in `sam3/working_dir/verified/images` and `sam3/working_dir/verified/mask`.
  - Videos in `sam3/outputs`:
    - `OrgVideo<video_number>.mp4`: Original frames.
    - `MaskVideo<video_number>.mp4`: Predicted masks.
    - `OverlappedVideo<video_number>.mp4`: Overlaid images.
- **YOLO Dataset Outputs**:
  - Dataset in `dataset_saving_working_dir/<folder_name>` (e.g., `DatasetManager/road_dataset`):
    - `train/images/`, `train/labels/`
    - `valid/images/`, `valid/labels/`
    - `test/images/`, `test/labels/` (if `test_split` > 0).

### 6. Component Scripts
- **FileManager.py**: Utilities for directory creation (`ensure_directory`), clearing (`clear_directory`), and frame path retrieval (`get_frame_paths`).
- **FrameExtractor.py**: Extracts video frames into images with configurable limits and progress tracking.
- **FrameHandler.py**: Manages frame paths and batch copying to a temporary directory.
- **MaskProcessor.py**: Converts masks to color images, generates bounding boxes, and processes batch masks with SAM2.
- **ImageCopier.py**: Copies verified images/masks to output directories, filtering based on overlays.
- **ImageOverlayProcessor.py**: Overlays masks on images for verification, supporting multi-threaded processing.
- **VideoCreator.py**: Creates videos from image folders using multi-threading.
- **SAM2Config.py**: Configures SAM2 model parameters (e.g., paths, label colors, batch size).
- **SAM2Model.py**: Initializes the SAM2 model, manages device selection (CPU/GPU), and monitors GPU memory.
- **sam2_video_predictor.py**: Core processing class (`SAM2VideoProcessor`) for frame annotation, mask prediction, and user interaction.
- **AnnotationManager.py**: Manages annotation data (points, labels, frame indices), saving/loading to/from JSON.
- **UserInteraction.py**: Handles GUI for annotation, including mouse/keyboard controls and zoom view.
- **pipeline.py**: Orchestrates the pipeline, integrating frame extraction, mask processing, overlay generation, and video creation.
- **logger_config.py**: Configures logging for debugging and monitoring.

## Keyboard and Mouse Controls for Annotation
- **1-9**: Change class label (mapped to `class_to_id`).
- **Left Click**: Add foreground point.
- **Right Click**: Add background point.
- **u**: Undo last point.
- **r**: Reset points for current frame.
- **Tab**: Increment instance ID.
- **Shift + Tab**: Decrement instance ID.
- **f**: Jump to specific frame index.
- **Enter**: Save points and proceed.
- **q**: Quit annotation.

A `Zoom View` window shows a magnified view around the mouse cursor.

## Directory Structure
```
SAM2/
├── DatasetManager/
│   └── YolovDatasetManager/
│       ├── create_yolo_structure.py
│       └── DatasetCreator.py
├── checkpoints/
│   └── sam2_hiera_large.pt
├── sam2/
│   └── __init__.py
├── sam2_configs/
│   └── sam2_hiera_l.yaml
├── sam3/
│   ├── __init__.py
│   ├── sam3_video_predictor_demo.py
│   ├── inputs/
│   │   ├── config/
│   │   │   └── default_config.yaml
│   │   ├── UserPrompts/
│   │   │   └── points_labels_*.json
│   │   └── VideoInputs/
│   │       └── Video*.mp4
│   ├── outputs/
│   │   ├── OrgVideo*.mp4
│   │   ├── MaskVideo*.mp4
│   │   └── OverlappedVideo*.mp4
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── FileManagement/
│   │   │   ├── __init__.py
│   │   │   ├── FileManager.py
│   │   │   ├── FrameExtractor.py
│   │   │   ├── FrameHandler.py
│   │   │   ├── ImageCopier.py
│   │   │   ├── ImageOverlayProcessor.py
│   │   │   ├── MaskProcessor.py
│   │   │   └── VideoCreator.py
│   │   ├── Model/
│   │   │   ├── __init__.py
│   │   │   ├── SAM2Config.py
│   │   │   ├── SAM2Model.py
│   │   │   └── sam2_video_predictor.py
│   │   ├── UserUI/
│   │   │   ├── __init__.py
│   │   │   ├── AnnotationManager.py
│   │   │   ├── logger_config.py
│   │   │   └── UserInteraction.py
│   │   └── pipeline.py
│   ├── working_dir/
│   │   ├── images/
│   │   ├── temp/
│   │   ├── render/
│   │   ├── overlap/
│   │   └── verified/
│   │       ├── images/
│   │       └── mask/
```

## Notes
- Ensure GPU drivers and PyTorch are configured for CUDA if using a GPU.
- The pipeline requires user interaction for annotation/verification unless `delete` is `true`.
- Video filenames must follow `video_path_template` (e.g., `Video1.mp4`).
- Frame filenames must follow `<prefix>_<frame_index>.jpg` (e.g., `Img_00000.jpg`).
- The `color_to_label` in `DatasetCreator.py` must match `label_colors` in `SAM2Config`.
- All custom modules must be in the `sam3/utils` directory structure.

## Troubleshooting
- **Missing SAM2 Checkpoint**: Ensure `sam2_hiera_large.pt` and `sam2_hiera_l.yaml` are in `checkpoints` and `sam2_configs`.
- **OpenCV Window Issues**: Verify system GUI support (e.g., X11 on Linux).
- **Invalid Frame Filenames**: Ensure frames follow the expected naming pattern.
- **GPU Errors**: Check GPU availability with `GPUtil.showUtilization()` and PyTorch CUDA support.
- **Missing Modules**: Verify all custom modules are in `sam3/utils`.
- **YOLO Dataset Issues**: Ensure `color_to_label` matches mask colors from `MaskProcessor`.

## Acknowledgements
- [SAM2](https://github.com/facebookresearch/segment-anything) by Meta AI
- PyTorch, OpenCV, and the open-source community

**For questions or bug reports, please open an issue.**