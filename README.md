# SAM2 Video Processing Pipeline

This repository provides a comprehensive pipeline for video segmentation using the SAM2 (Segment Anything Model 2) model, with functionality to create YOLO-compatible datasets for training. The pipeline automates frame extraction, supports interactive point-based annotation, performs mask prediction, overlays masks on original images, compiles results into output videos, and processes images and masks into a YOLO dataset format with augmentations. Future enhancements will include support for additional dataset formats (e.g., COCO, Pascal VOC) for broader compatibility.

## Features
- **Frame Extraction**: Extracts frames from input videos into images using `FrameExtractor`.
- **Interactive Annotation**: Annotates frames with points and class labels via an OpenCV-based GUI, supporting multiple classes and instance ID management.
- **Batch Processing**: Processes frames in configurable batches for efficient GPU usage.
- **Mask Prediction**: Uses SAM2 to predict segmentation masks for annotated objects.
- **Mask Overlay**: Overlays predicted masks on original images for verification using `ImageOverlayProcessor`.
- **Verification Step**: Prompts user verification of overlays before finalizing output.
- **Video Compilation**: Creates output videos (original, masks, and overlays) using `VideoCreator`.
- **Dataset Creation**: Converts SAM2-generated masks and images into YOLO format with augmentations using `DatasetCreator`, with support for multiple classes (e.g., `road`, `cars`, `trucks`).
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
- Platform dependencies for GUI (e.g., X11 on Linux or compatible display server on Windows)

Install dependencies with:
```bash
pip install torch torchvision opencv-python numpy GPUtil tqdm pygetwindow pillow
```

Ensure the SAM2 model checkpoint (`sam2_hiera_large.pt`) and configuration (`sam2_hiera_l.yaml`) are placed in the `checkpoints` and `sam2_configs` directories, respectively. The `create_yolo_structure` module, required for dataset creation, must be in the `DatasetManager/YolovDatasetManager` directory.

## Usage

### 1. Prepare Inputs
- Place input videos in the specified directory (e.g., `.\VideoInputs\Video1.mp4`, `.\VideoInputs\Video2.mp4`, ...).
- Ensure the SAM2 model checkpoint (`sam2_hiera_large.pt`) and configuration (`sam2_hiera_l.yaml`) are in the `checkpoints` and `sam2_configs` directories.
- Verify that custom modules (`FrameExtractor`, `ImageCopier`, `ImageOverlayProcessor`, `VideoCreator`, `create_yolo_structure`) are available in the project directory.

### 2. Running the SAM2 Video Processing Pipeline
Navigate to the project directory:
```bash
cd .\segment-anything-3
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
- `--video_path_template`: Template for video file paths (default: `.\VideoInputs\Video{}.mp4`).
- `--images_extract_dir`: Directory for extracted frames (default: `.\working_dir\images`).
- `--temp_processing_dir`: Directory for temporary batch processing (default: `.\working_dir\temp`).
- `--rendered_dir`: Directory for rendered masks (default: `.\working_dir\render`).
- `--overlap_dir`: Directory for mask overlays (default: `.\working_dir\overlap`).
- `--verified_img_dir`: Directory for verified original images (default: `.\working_dir\verified\images`).
- `--verified_mask_dir`: Directory for verified mask images (default: `.\working_dir\verified\mask`).
- `--final_video_path`: Directory for output videos (default: `.\outputs`).

#### Modes of Operation for `sam2_video_predictor.py`
The `sam2_video_predictor.py` script operates in two primary modes:
1. **Interactive Annotation Mode**:
   - Opens an OpenCV GUI (`SAM2 Annotation Tool`) for users to annotate frames by adding foreground (left-click) or background (right-click) points.
   - Supports multiple classes (e.g., `road`, `cars`, `trucks`) via number keys (1-9) to select class labels.
   - Manages instance IDs within each class using `Tab` (increment) and `Shift+Tab` (decrement). Instance IDs are encoded as `class_id * 1000 + instance_id` (e.g., `1001` for class 1, instance 1; `2003` for class 2, instance 3).
   - Saves points and labels to JSON files in `working_dir/UserPrompts` (e.g., `points_labels_Img<video_number>.json`) for reuse.
   - Provides a zoom view window (`Zoom View`) for precise annotation.
2. **Batch Processing Mode**:
   - Processes frames in batches (controlled by `--batch_size`) using the SAM2 model to propagate masks across frames based on user-provided points.
   - Loads saved points and labels from JSON files to resume processing if sufficient data exists.
   - Uses multi-threading (`ThreadPoolExecutor`) to parallelize mask generation and image writing.
   - Checks for sufficient user input (`check_data_sufficiency`) before processing each batch, pausing if more annotations are needed.

### 3. Annotation and Verification
- **Interactive Annotation**:
  - The script opens an OpenCV window for annotating frames.
  - Use mouse clicks and keyboard controls (see below) to add points and manage annotations.
  - Press `Enter` to save points and proceed to the next batch.
- **Verification**:
  - Overlays of masks on original images are generated in `overlap_dir` for review.
  - The pipeline prompts for verification (unless `--delete yes` is specified) before copying verified images and masks to `verified_img_dir` and `verified_mask_dir`.
  - Three output videos are created: original frames, mask predictions, and overlays.

### 4. Creating a YOLO Dataset
The `DatasetCreator.py` script processes SAM2-generated images and masks (from `working_dir/images` and `working_dir/render`) into a YOLO-compatible dataset with augmentations. It supports multiple classes (e.g., `road`, `cars`, `trucks`) by mapping mask colors to class IDs and creates a dataset structure with `train`, `valid`, and `test` splits, including images and YOLO-format label files.

Run the dataset creation script:
```bash
cd ../DatasetManager/YolovDatasetManager
python DatasetCreator.py
```

#### Configuration for `DatasetCreator.py`
The script uses a configuration dictionary defined in `DatasetCreator.py`. A sample configuration supporting multiple classes is:
```python
CONFIG = {
    "dataset_path": r"../../sam3/working_dir",
    "SOURCE_mask_folder_name": "render",
    "SOURCE_original_folder_name": "images",
    "SOURCE_mask_type_ext": ".png",
    "SOURCE_img_type_ext": ".jpeg",
    "augment_times": 10,  # Number of augmentations per image
    "test_split": 0.0,  # Percentage for test set
    "val_split": 0.1,  # Percentage for validation set
    "train_split": 0.9,  # Percentage for training set
    "Keep_val_dataset_original": True,  # Keep validation images unaugmented
    "num_threads": os.cpu_count() - 2,  # Threads for parallel processing
    "class_to_id": {
        "road": 0,
        "cars": 1,
        "trucks": 2
    },  # Class name to ID mapping
    "color_to_label": {
        (255, 255, 255): 0,  # road
        (0, 0, 255): 1,      # cars
        (255, 0, 0): 2       # trucks
    },  # Mask color to label mapping
    "class_names": ["road", "cars", "trucks"],  # List of class names
    "dataset_saving_working_dir": r".\DatasetManager",
    "folder_name": "road_dataset",  # Dataset folder name
    "DESTINATION_img_type_ext": ".jpg",
    "DESTINATION_label_type_ext": ".txt",
    "FromDataType": "",
    "ToDataTypeFormate": ""
}
```

#### Multiple Class Support and Instance ID Management
- **Multiple Classes**: The pipeline supports multiple classes (e.g., `road`, `cars`, `trucks`) by mapping mask colors to class IDs in `DatasetCreator.py` via `color_to_label`. These colors must match the `label_colors` dictionary in `sam2_video_predictor.py` (e.g., `(255, 255, 255)` for `road`, `(0, 0, 255)` for `cars`).
- **Instance ID Management**: During annotation in `sam2_video_predictor.py`, instance IDs are managed per class using `Tab` (increment) and `Shift+Tab` (decrement). The encoded label is calculated as `class_id * 1000 + instance_id` (e.g., `1001` for class 1, instance 1; `2003` for class 2, instance 3), ensuring unique IDs for each instance within a class.
- **YOLO Format**: Masks are converted to polygon annotations in YOLO format (e.g., `0 0.1 0.2 0.3 0.4 ...` for `road`, `1 0.5 0.6 ...` for `cars`). Each `.txt` file contains annotations for all objects in an image, with class IDs corresponding to the `class_to_id` mapping.

#### Augmentations
`DatasetCreator.py` applies the following augmentations to training images (validation images remain unaugmented if `Keep_val_dataset_original` is `True`):
- **Color Jitter**: Adjusts brightness and contrast with varying ranges per augmentation iteration.
- **Gaussian Blur**: Applies a Gaussian kernel (size 3 or 5) to blur images.
- **Average Blur**: Applies an average kernel (size 3 or 5) for uniform blurring.
- **Gaussian Noise**: Adds random noise with configurable mean and sigma.
- **Salt-and-Pepper Noise**: Adds random white (salt) and black (pepper) pixels.

#### YOLO Format
- Each label file (`.txt`) contains one line per object, with the class ID followed by normalized (x, y) coordinates of polygon vertices.
- Images and labels are distributed into `train`, `valid`, and `test` folders based on the specified splits.

#### Future Dataset Formats
- The pipeline currently supports YOLO format for dataset creation. Future enhancements will include support for other standard formats, such as:
  - **COCO**: JSON-based annotations with bounding boxes, segmentation masks, and categories.
  - **Pascal VOC**: XML-based annotations for bounding boxes and class labels.
  - These formats will be implemented to convert SAM2-generated masks and original video frames into datasets compatible with various object detection frameworks.

### 5. Output
- **SAM2 Pipeline Outputs**:
  - Verified images and masks in `verified_img_dir` and `verified_mask_dir`.
  - Videos in `final_video_path`:
    - `OrgVideo<video_number>.mp4`: Original frames.
    - `MaskVideo<video_number>.mp4`: Predicted masks.
    - `OverlappedVideo<video_number>.mp4`: Overlaid images.
- **YOLO Dataset Outputs**:
  - Dataset structure in `dataset_saving_working_dir/<folder_name>` (e.g., `.\DatasetManager\road_dataset`):
    - `train/images/`, `train/labels/`: Training images and YOLO-format labels.
    - `valid/images/`, `valid/labels/`: Validation images and labels.
    - `test/images/`, `test/labels/`: Test images and labels (if `test_split` > 0).

### 6. Component Scripts
- **FrameExtractor.py**:
  - Extracts frames from input videos and saves them as images (e.g., `Img<video_number>_00001.jpeg`) in the specified output directory.
  - Supports limiting the number of extracted frames (`limitedImages`) and uses `tqdm` for progress tracking.
  - Ensures the output directory exists and validates video file accessibility.
- **ImageCopier.py**:
  - Copies verified original images and masks from `images` and `render` to `verified/images` and `verified/mask` directories, respectively.
  - Filters images based on the presence of corresponding overlay images in `overlap_dir`.
  - Uses multi-threading (`ThreadPoolExecutor`) for efficient copying with progress tracking via `tqdm`.
- **ImageOverlayProcessor.py**:
  - Overlays masks on original images to create visual verification images, saved in `overlap_dir`.
  - Supports filtering images by prefix (`all_consider`) and minimum image count (`image_count`).
  - Uses multi-threading for parallel processing with `tqdm` progress bars.
- **VideoCreator.py**:
  - Creates videos from images in specified folders (e.g., `verified/images`, `verified/mask`, `overlap`).
  - Outputs three videos per video number: original, mask, and overlay videos (e.g., `OrgVideo1.mp4`, `MaskVideo1.mp4`, `OverlappedVideo1.mp4`).
  - Uses threading for parallel video creation and `tqdm` for progress tracking.

## Keyboard and Mouse Controls for Annotation
- **1-9**: Change the current class label (1 to 9, mapped to `class_to_id` in `DatasetCreator.py`).
- **Left Click**: Add a foreground point for the current class and instance.
- **Right Click**: Add a background point for the current class and instance (visualized differently).
- **u**: Undo the last added point.
- **r**: Reset all points for the current frame.
- **Tab**: Increment the instance ID for the current class.
- **Shift + Tab**: Decrement the instance ID for the current class.
- **f**: Jump to a specific frame index for annotation.
- **Enter**: Save points and labels for the current frame and proceed.
- **q**: Quit the annotation process.

A zoom view window (`Zoom View`) shows a magnified view of the area around the mouse cursor during annotation.

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
├── segment-anything-3/
│   ├── VideoInputs/
│   │   └── Video1.mp4, Video2.mp4, ...
│   ├── working_dir/
│   │   ├── images/             # Extracted frames
│   │   ├── temp/               # Temporary processing files
│   │   ├── render/             # Rendered masks
│   │   ├── overlap/            # Overlaid images
│   │   ├── verified/
│   │   │   ├── images/         # Verified original images
│   │   │   └── mask/           # Verified masks
│   │   └── UserPrompts/
│   │       └── points_labels_*.json  # Stored points and labels
│   ├── outputs/
│   │   ├── OrgVideo1.mp4       # Original video
│   │   ├── MaskVideo1.mp4      # Mask video
│   │   └── OverlappedVideo1.mp4 # Overlay video
│   ├── FrameExtractor.py
│   ├── ImageCopier.py
│   ├── ImageOverlayProcessor.py
│   ├── logger_config.py
│   ├── VideoCreator.py
│   └── sam2_video_predictor.py
```

## Notes
- Ensure GPU drivers and PyTorch are configured for CUDA if using a GPU.
- The SAM2 pipeline requires user interaction for annotation and verification unless `--delete yes` is specified.
- Video filenames must follow the pattern specified in `--video_path_template` (e.g., `Video1.mp4`).
- Frame filenames must follow the pattern `<prefix>_<frame_index>.jpg` (e.g., `Img_00000.jpg`).
- The `color_to_label` mapping in `DatasetCreator.py` must match the `label_colors` dictionary in `sam2_video_predictor.py` (e.g., `(0, 0, 255): 1` for `cars`, `(255, 0, 0): 2` for `trucks`).
- The `class_to_id` and `class_names` in `DatasetCreator.py` should align with the classes used during annotation in `sam2_video_predictor.py`.
- The SAM2 model, custom modules (`FrameExtractor`, etc.), and `create_yolo_structure` are critical dependencies; ensure they are correctly installed and accessible.
- Future implementations will add support for COCO and Pascal VOC formats to enable broader compatibility with object detection frameworks.

## Troubleshooting
- **Missing SAM2 Checkpoint**: Ensure `sam2_hiera_large.pt` and `sam2_hiera_l.yaml` are in the `checkpoints` and `sam2_configs` directories.
- **OpenCV Window Issues**: Verify that your system supports OpenCV's GUI (e.g., X11 on Linux or a compatible display server on Windows).
- **Invalid Frame Filenames**: Ensure extracted frames follow the expected naming pattern (`<prefix>_<frame_index>.jpg`).
- **GPU Errors**: Check GPU availability with `GPUtil.showUtilization()` and ensure PyTorch is CUDA-enabled.
- **Missing Custom Modules**: Verify that `FrameExtractor.py`, `ImageCopier.py`, `ImageOverlayProcessor.py`, `VideoCreator.py`, and `create_yolo_structure.py` are in the project directory.
- **YOLO Dataset Issues**: Ensure the `color_to_label` mapping matches the mask colors produced by `sam2_video_predictor.py`. Check that source directories (`images` and `render`) exist and contain valid files.
- **Instance ID Conflicts**: Ensure instance IDs are unique within each class during annotation (managed via `Tab`/`Shift+Tab`).

## Acknowledgements
- [SAM2](https://github.com/facebookresearch/segment-anything) by Meta AI
- PyTorch, OpenCV, and the open-source community

**For questions or bug reports, please open an issue.**
