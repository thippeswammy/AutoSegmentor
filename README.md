# AutoSegmentor

[![GitHub](https://img.shields.io/github/stars/thippeswammy/AutoSegmentor?style=social)](https://github.com/thippeswammy/AutoSegmentor)
[![Demo Video](https://img.shields.io/badge/Demo-Video-blue)](https://drive.google.com/file/d/1vvW7xPivgNbbkHP49AU1s5goSi0Zlr0S/view?usp=sharing)

![AutoSegmenter2](https://github.com/user-attachments/assets/9cb00e30-7c1a-4c6f-8f3d-ac2811424e00)

_AutoSegmentor is a game-changer for anyone working with video data in computer vision. This open-source project provides a comprehensive auto-labeling system that converts raw videos—including long videos and complex scenes—into structured datasets using Meta AI's cutting-edge [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/segment-anything)._

**Main purpose:**  
Build an auto-labeling pipeline that converts raw videos into structured YOLO-compatible datasets using SAM2, with real-time segmentation enabled by CUDA acceleration, multithreading, and an interactive GUI annotation system.  
AutoSegmentor supports long videos and visually-rich content graphics, making it ideal for both standard and advanced video processing tasks.  
**You can create datasets for any required object or class simply by giving visual prompts—no manual labeling required.**

> **Demo:**  
> [Watch the demo video (long video, includes GFX)](https://drive.google.com/file/d/1vvW7xPivgNbbkHP49AU1s5goSi0Zlr0S/view?usp=sharing)

---

## Features

- **Automated Frame Extraction:** Extracts frames from short or long videos using a robust, configurable pipeline.
- **Interactive Annotation:** Point-based, multi-class annotation with real-time OpenCV GUI.
- **Batch & Real-time Processing:** Efficient batch segmentation with CUDA and multithreading.
- **Mask Prediction & Overlay:** Predicts masks with SAM2 and overlays for easy verification.
- **Output Video Compilation:** Produces original, mask, and overlay videos for review.
- **YOLO Dataset Creation:** Converts masks/images into YOLO format with augmentations.
- **Long Video Support:** Handles visually-rich videos and lengthy footage efficiently.
- **Directory & File Management:** Automated temp/output dir handling and cleanup.
- **Extensible Dataset Support:** (WIP) Future support for COCO, Pascal VOC, etc.
- **Open Source:** MIT-licensed and community-friendly.

---

## Demo

- **Demo Video:** [Watch here (Google Drive, long video with GFX)](https://drive.google.com/file/d/1vvW7xPivgNbbkHP49AU1s5goSi0Zlr0S/view?usp=sharing)
- **Source Code:** [https://github.com/thippeswammy/AutoSegmentor](https://github.com/thippeswammy/AutoSegmentor)

---

## Requirements

- Python 3.8+
- PyTorch (with CUDA for GPU acceleration)
- OpenCV (`opencv-python`)
- NumPy
- GPUtil
- tqdm
- pygetwindow
- Pillow (`PIL`)
- **SAM2** library and checkpoint (`sam2_hiera_large.pt`)
- Custom modules: `FileManager`, `FrameExtractor`, `FrameHandler`, `MaskProcessor`, `ImageCopier`, `ImageOverlayProcessor`, `VideoCreator`, `SAM2Config`, `SAM2Model`, `sam2_video_predictor`, `AnnotationManager`, `UserInteraction`, `pipeline`, `create_yolo_structure`
- Platform dependencies for GUI (e.g., X11 on Linux or compatible display server on Windows)

Install dependencies:
```bash
pip install torch torchvision opencv-python numpy GPUtil tqdm pygetwindow pillow
```
_Ensure your GPU drivers & PyTorch are CUDA-ready if using GPU._

**SAM2 files:** Place `sam2_hiera_large.pt` in `checkpoints/`, and `sam2_hiera_l.yaml` in `sam2_configs/`.

---

## Usage

### 1. Prepare Inputs

- Place input videos (including long) in `sam3/inputs/VideoInputs/` (e.g., `Video1.mp4`, `Video2.mp4`, ...).
- Ensure the SAM2 checkpoint and config are in `checkpoints/` and `sam2_configs/`.
- Confirm all custom modules exist in `sam3/utils/`.

### 2. Run the Main Pipeline

- Navigate to the root directory (where `sam3` is located):

```bash
cd path/to/your/AutoSegmentor
python sam3/sam3_video_predictor_demo.py
```
- By default, this loads parameters from `sam3/inputs/config/default_config.yaml`.

#### Custom Configuration

Edit `sam3/inputs/config/default_config.yaml` to control:
- Input video range (`video_start`, `video_end`)
- Filename prefix (`prefix`)
- Processing params (`batch_size`, `fps`)
- Directory paths (e.g., `working_dir_name`, `images_extract_dir`)
- Cleanup policy (`delete`: auto/manual)

**Sample config keys:**
```yaml
video_start: 1
video_end: 2
prefix: "Img"
batch_size: 8
fps: 24
delete: false
working_dir_name: "working_dir"
video_path_template: "sam3/inputs/VideoInputs/Video{}.mp4"
...
```

---

### 3. Annotation

The annotation and verification processes are orchestrated as part of the pipeline and are highly interactive:

- Uses an OpenCV-based GUI for point-and-click annotation.
- Save annotations as JSON per video in `sam3/inputs/UserPrompts/points_labels_<prefix><video_number>.json`.
- Supported keyboard and mouse controls:
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

A zoom window shows a magnified area around the cursor for precision annotation.

---

### 4. Create YOLO Dataset

- Converts processed images/masks into YOLO V8-compatible datasets, with augmentations (color jitter, blur, noise, etc.).
- Multi-class support via color mapping.

Run:
```bash
cd DatasetManager/YolovDatasetManager
python DatasetCreator.py
```

**Example CONFIG in `DatasetCreator.py`:**
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
        (255, 255, 255): 0,   # road
        (0, 0, 255): 1,       # cars
        (255, 0, 0): 2        # trucks
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
**Note:** `color_to_label` must match the mask colors output by `MaskProcessor` & set in `SAM2Config`.

**Multiple Classes & Instance ID:**  
Maps mask colors to class IDs in `color_to_label`, and uses keyboard shortcuts for instance ID management (labels are encoded as `class_id * 1000 + instance_id`).

**YOLO Format:**  
Converts masks to polygon annotations (e.g., `0 0.1 0.2 ...` for `road`).

**Augmentations:**  
Color jitter, Gaussian blur, average blur, Gaussian noise, salt-and-pepper noise.

**Future formats:**  
COCO and Pascal VOC support are planned.

---

## Output Structure

- **AutoSegmentor Pipeline Outputs:**
  - Verified images and masks in `sam3/working_dir/verified/images` and `sam3/working_dir/verified/mask`.
  - Videos in `sam3/outputs/`:
    - `OrgVideo<video_number>.mp4`: Original frames.
    - `MaskVideo<video_number>.mp4`: Predicted masks.
    - `OverlappedVideo<video_number>.mp4`: Overlaid images.
- **YOLO Dataset Outputs:**
  - Dataset in `dataset_saving_working_dir/<folder_name>` (e.g., `DatasetManager/road_dataset`):
    - `train/images/`, `train/labels/`
    - `valid/images/`, `valid/labels/`
    - `test/images/`, `test/labels/` (if `test_split` > 0).

---

## Component Scripts

- **FileManager.py**: Utilities for directory creation, clearing, and frame path retrieval.
- **FrameExtractor.py**: Extracts video frames into images with configurable limits and progress tracking.
- **FrameHandler.py**: Manages frame paths and batch copying to a temporary directory.
- **MaskProcessor.py**: Converts masks to color images, generates bounding boxes, and processes batch masks with SAM2.
- **ImageCopier.py**: Copies verified images/masks to output directories, filtering based on overlays.
- **ImageOverlayProcessor.py**: Overlays masks on images for verification, supporting multi-threaded processing.
- **VideoCreator.py**: Creates videos from image folders using multi-threading.
- **SAM2Config.py**: Configures SAM2 model parameters (e.g., paths, label colors, batch size).
- **SAM2Model.py**: Initializes the SAM2 model, manages device selection (CPU/GPU), and monitors GPU memory.
- **sam2_video_predictor.py**: Core processing class for frame annotation, mask prediction, and user interaction.
- **AnnotationManager.py**: Manages annotation data, saving/loading to/from JSON.
- **UserInteraction.py**: Handles GUI for annotation, including mouse/keyboard controls and zoom view.
- **pipeline.py**: Orchestrates the pipeline, integrating all stages.
- **logger_config.py**: Configures logging for debugging and monitoring.

---

## Directory Structure

```
AutoSegmentor/
├── DatasetManager/
│   └── YolovDatasetManager/
│       ├── create_yolo_structure.py
│       └── DatasetCreator.py
├── checkpoints/
│   └── sam2_hiera_large.pt
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

---

## Troubleshooting

- **Missing SAM2 Checkpoint:** Ensure `sam2_hiera_large.pt` is in `checkpoints/`, and YAML config in `sam2_configs/`.
- **OpenCV Window Issues:** Verify system GUI support (e.g., X11 on Linux).
- **Invalid Frame Filenames:** Ensure frames follow the expected naming pattern.
- **GPU Errors:** Check GPU availability with `GPUtil.showUtilization()` and PyTorch CUDA support.
- **Missing Modules:** Verify all custom modules are in `sam3/utils`.
- **YOLO Dataset Issues:** Ensure `color_to_label` matches mask colors from `MaskProcessor`.

---

## Acknowledgements

- [Meta AI's SAM2](https://github.com/facebookresearch/segment-anything)
- PyTorch, OpenCV, and the open-source vision community

---

## Get Involved

AutoSegmentor is open source and welcomes contributions!  
**Star, fork, or open issues at:**  
[https://github.com/thippeswammy/AutoSegmentor](https://github.com/thippeswammy/AutoSegmentor)

**For questions or bug reports, please open an issue.**
