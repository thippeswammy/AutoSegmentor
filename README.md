# sam3 Video Processing Pipeline

This repository provides a comprehensive pipeline for video segmentation using the sam3 (Segment Anything Model 3) model, with functionality to create YOLO-compatible datasets for training. The pipeline automates frame extraction, supports interactive point-based annotation, performs mask prediction, overlays masks on original images, compiles results into output videos, and processes images and masks into a YOLO dataset format with augmentations. Future enhancements will include support for additional dataset formats (e.g., COCO, Pascal VOC) for broader compatibility.

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

### 2. Running the sam3 Video Processing Pipeline

The `sam3/sam3_video_predictor.py` script orchestrates the video processing workflow. Unlike previous versions, it is primarily configured using a YAML file instead of command-line arguments.

**Prerequisites:**
- Ensure your input videos are in the location specified by `video_path_template` in your configuration file.
- Verify that the SAM2 model checkpoint (e.g., `sam2_hiera_large.pt`) and its configuration (e.g., `sam2_hiera_l.yaml`) are correctly placed in the `checkpoints` and `sam2_configs` directories, respectively. These are utilized by the underlying `run_pipeline` function.

**Execution:**

Navigate to the root directory of the project (where `sam3` directory is located):
```bash
cd path/to/your/project_root
```

Run the main pipeline script:
```bash
python sam3/sam3_video_predictor.py
```
The script will load its parameters from `sam3/inputs/config/default_config.yaml` by default. To use a different configuration file, you would need to modify the `config_path` variable within the `load_config` function in `sam3/sam3_video_predictor.py`.

#### Configuration (`sam3/inputs/config/default_config.yaml`)

All operational parameters for `sam3_video_predictor.py` are defined in a YAML configuration file, by default `sam3/inputs/config/default_config.yaml`. Users **must** modify this file to control aspects such as:
- Input video range (`video_start`, `video_end`)
- File naming (`prefix`)
- Processing parameters (`batch_size`, `fps`)
- Directory paths for various stages (`working_dir_name`, `images_extract_dir`, `rendered_dir`, etc.)
- Cleanup behavior (`delete`)

#### Configuration Details (`sam3/inputs/config/default_config.yaml`)

The behavior of `sam3_video_predictor.py` is controlled by parameters in the `sam3/inputs/config/default_config.yaml` file. Below are the key parameters:

-   `video_start` (int): The starting number of the video sequence to process (e.g., if you have `Video1.mp4`, `Video2.mp4`, and want to start with `Video1.mp4`, set this to 1).
-   `video_end` (int): The number of videos to process. For example, if `video_start` is 1 and `video_end` is 2, it will process `Video1.mp4` and `Video2.mp4`.
-   `prefix` (str): A prefix string used for naming intermediate files and directories (e.g., "Img").
-   `batch_size` (int): The number of frames to be processed in a single batch by the underlying `run_pipeline` function. This affects memory usage and processing speed.
-   `fps` (int): Frames per second for the output videos created by `run_pipeline`.
-   `delete` (str or bool): Controls cleanup of the `working_dir_name`.
    *   `'yes'` or `true`: Automatically deletes the working directory before processing a new video and after it's done.
    *   `'no'` or `false`: Prompts the user for confirmation before deleting the working directory.
-   `working_dir_name` (str): The name of the main directory where all intermediate files (extracted frames, masks, overlaps, etc.) for the current video will be stored (e.g., "working_dir").
-   `video_path_template` (str): A template string defining the path to input video files. It should include `{}` where the video number will be inserted (e.g., `.\VideoInputs\Video{}.mp4`). The `working_dir` placeholder is not typically used here as it points to source videos.
-   `images_extract_dir` (str): Path template for the directory where extracted frames will be saved. Typically includes `working_dir` as a placeholder (e.g., `.\working_dir\images`).
-   `temp_processing_dir` (str): Path template for temporary files during batch processing (e.g., `.\working_dir\temp`).
-   `rendered_dir` (str): Path template for storing rendered mask images (e.g., `.\working_dir\render`).
-   `overlap_dir` (str): Path template for storing images with masks overlaid (e.g., `.\working_dir\overlap`).
-   `verified_img_dir` (str): Path template for user-verified original images (e.g., `.\working_dir\verified\images`).
-   `verified_mask_dir` (str): Path template for user-verified mask images (e.g., `.\working_dir\verified\mask`).
-   `final_video_path` (str): The directory where final output videos (original, mask, overlap) will be saved (e.g., `.\outputs`). This path does not typically use the `working_dir` placeholder as it's a final output location.
-   `images_ending_count` (int): Specifies the number of digits to use for frame numbering in filenames (e.g., if 5, frames will be `00001.jpg`, `00002.jpg`).

To customize the pipeline's execution, modify these values in `sam3/inputs/config/default_config.yaml` before running `sam3_video_predictor.py`.

#### Core Workflow of `sam3_video_predictor.py`

The `sam3_video_predictor.py` script performs the following high-level steps:

1.  **Loads Configuration:** Reads parameters from the YAML file.
2.  **Iterates Through Videos:** Processes a sequence of videos based on `video_start` and `video_end` from the config.
3.  **Manages Working Directory:**
    *   Before processing each video, it checks if the specified `working_dir_name` exists.
    *   If `delete` is set to 'yes' in the config, it automatically removes the directory.
    *   If `delete` is 'no', it prompts the user for confirmation to clear the directory. The script will exit if the user chooses not to clear it.
4.  **Executes Processing Pipeline:** Calls the `run_pipeline` function (from `sam3.utils.pipeline`), passing the loaded and constructed configuration parameters. This `run_pipeline` function encapsulates the detailed video processing steps, including:
    *   Frame extraction
    *   Interactive annotation (if required by the underlying steps)
    *   Mask prediction using the SAM2 model
    *   Mask overlay generation
    *   Video compilation
5.  **Post-Processing Cleanup:** After processing each video, it again manages the `working_dir_name` based on the `delete` setting, potentially prompting the user if `delete` is 'no'.

The interactive annotation mode, batch processing, and specific SAM2 model interactions previously described are now part of the workflow managed by `run_pipeline` and its associated modules, configured as per the YAML file.

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
For more information, see the [README](./DatasetManager/YolovDatasetManager/README.md).

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
- **Multiple Classes**: The pipeline supports multiple classes (e.g., `road`, `cars`, `trucks`) by mapping mask colors to class IDs in `DatasetCreator.py` via `color_to_label`. These colors must match the `label_colors` dictionary in `.\SAM2\sam3\sam3_video_predictor.py` (e.g., `(255, 255, 255)` for `road`, `(0, 0, 255)` for `cars`).
- **Instance ID Management**: During annotation in `.\SAM2\sam3\sam3_video_predictor.py`, instance IDs are managed per class using `Tab` (increment) and `Shift+Tab` (decrement). The encoded label is calculated as `class_id * 1000 + instance_id` (e.g., `1001` for class 1, instance 1; `2003` for class 2, instance 3), ensuring unique IDs for each instance within a class.
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
- **sam3 Pipeline Outputs**:
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
│   └── .\SAM2\sam3\sam3_video_predictor.py
```

## Notes
- Ensure GPU drivers and PyTorch are configured for CUDA if using a GPU.
- The sam3 pipeline requires user interaction for annotation and verification unless `--delete yes` is specified.
- Video filenames must follow the pattern specified in `--video_path_template` (e.g., `Video1.mp4`).
- Frame filenames must follow the pattern `<prefix>_<frame_index>.jpg` (e.g., `Img_00000.jpg`).
- The `color_to_label` mapping in `DatasetCreator.py` must match the `label_colors` dictionary used by the annotation tools within the processing pipeline (called by `sam3/sam3_video_predictor.py` via `run_pipeline`) (e.g., `(0, 0, 255): 1` for `cars`, `(255, 0, 0): 2` for `trucks`).
- The `class_to_id` and `class_names` in `DatasetCreator.py` should align with the classes used during annotation by the tools within the processing pipeline (called by `sam3/sam3_video_predictor.py` via `run_pipeline`).
- The SAM2 model, custom modules (`FrameExtractor`, etc.), and `create_yolo_structure` are critical dependencies; ensure they are correctly installed and accessible.
- Future implementations will add support for COCO and Pascal VOC formats to enable broader compatibility with object detection frameworks.

## Troubleshooting
- **Missing SAM2 Checkpoint**: Ensure `sam2_hiera_large.pt` and `sam2_hiera_l.yaml` are in the `checkpoints` and `sam2_configs` directories.
- **OpenCV Window Issues**: Verify that your system supports OpenCV's GUI (e.g., X11 on Linux or a compatible display server on Windows).
- **Invalid Frame Filenames**: Ensure extracted frames follow the expected naming pattern (`<prefix>_<frame_index>.jpg`).
- **GPU Errors**: Check GPU availability with `GPUtil.showUtilization()` and ensure PyTorch is CUDA-enabled.
- **Missing Custom Modules**: Verify that `FrameExtractor.py`, `ImageCopier.py`, `ImageOverlayProcessor.py`, `VideoCreator.py`, and `create_yolo_structure.py` are in the project directory.
- **YOLO Dataset Issues**: Ensure the `color_to_label` mapping matches the mask colors produced by the processing pipeline (called by `sam3/sam3_video_predictor.py` via `run_pipeline`). Check that source directories (`images` and `render`) exist and contain valid files.
- **Instance ID Conflicts**: Ensure instance IDs are unique within each class during annotation (managed via `Tab`/`Shift+Tab`).

## Acknowledgements
- [SAM2](https://github.com/facebookresearch/segment-anything) by Meta AI
- PyTorch, OpenCV, and the open-source community

**For questions or bug reports, please open an issue.**
