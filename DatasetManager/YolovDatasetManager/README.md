# YOLO Dataset Manager

## Overview
The **YOLO Dataset Manager** is a Python-based tool designed to create YOLO-compatible datasets for object detection tasks. It processes images and their corresponding segmentation masks, applies augmentations, and organizes them into a YOLO folder structure (`train`, `valid`, `test`) with images and labels. The tool supports multithreaded processing, various image augmentations (e.g., blur, noise, color jitter), and automatic generation of a `data.yaml` configuration file. It is ideal for preparing datasets for training YOLO models, particularly for applications like autonomous driving or road segmentation.

## Why It’s Used
This tool is used to:
- **Prepare YOLO Datasets**: Convert image and mask data into YOLO format with normalized polygon annotations.
- **Augment Data**: Generate multiple augmented versions of images and labels to improve model robustness.
- **Organize Data**: Automatically create a YOLO folder structure with train, validation, and test splits.
- **Support Object Detection**: Facilitate training YOLO models for tasks like road segmentation in autonomous driving.
- **Ensure Efficiency**: Use multithreading to process large datasets quickly and handle errors gracefully.

## Features
- **YOLO Folder Structure Creation**: Generates `train`, `valid`, and `test` directories with `images` and `labels` subfolders, plus a `data.yaml` file.
- **Image Augmentation**:
  - Gaussian and average blur with variable kernel sizes.
  - Gaussian noise with customizable mean and sigma.
  - Salt and pepper noise with adjustable probabilities.
  - Color jitter (brightness and contrast adjustments).
- **Multithreaded Processing**: Uses `ThreadPoolExecutor` to parallelize image and label processing, leveraging CPU cores.
- **Mask-to-YOLO Conversion**: Converts segmentation masks to YOLO polygon annotations with normalized coordinates.
- **Flexible Splitting**: Configurable train, validation, and test splits (e.g., 90% train, 10% validation).
- **Original Validation Option**: Optionally keeps validation images unaugmented for accurate evaluation.
- **Error Handling**: Robust logging for file not found, processing errors, and other issues.
- **Progress Tracking**: Displays a progress bar using `tqdm` for user feedback.
- **Configurability**: Supports custom class names, color-to-label mappings, file extensions, and augmentation parameters.
- **GPU Support**: Uses PyTorch with CUDA (if available) for faster image augmentations.

## How It Works
The tool is built with a modular architecture:
1. **create_yolo_structure.py**: Creates a YOLO folder structure (`train`, `valid`, `test`) with unique folder names to avoid overwrites and generates a `data.yaml` file with dataset configuration.
2. **DatasetCreatere.py**:
   - **YoloProcessor**: Orchestrates data processing, including folder setup, image loading, augmentation, and saving.
   - **ImageAugmentations**: Applies augmentations (blur, noise, color jitter) using PyTorch tensors, with GPU support.
   - **File Distribution**: Collects image paths, shuffles them, and processes files in parallel using threads.
   - **Mask Processing**: Converts segmentation masks to YOLO polygon annotations by extracting contours and normalizing coordinates.
   - **Saving**: Saves augmented images (`.jpg`) and YOLO labels (`.txt`) to the appropriate split directories.

### Workflow
- **Setup**: Reads configuration (e.g., dataset path, class names, augmentation times) and creates YOLO folder structure.
- **Load Data**: Collects images from the source directory (e.g., `images`) and corresponding masks (e.g., `render`).
- **Process Files**:
  - For each image-mask pair, generates multiple augmented versions (default: 10).
  - Applies augmentations (e.g., blur, noise, color jitter) to images while keeping labels unchanged.
  - Converts masks to YOLO polygon format (class ID + normalized coordinates).
- **Distribute**:
  - Randomly assigns augmented images and labels to `train`, `valid`, or `test` splits based on configured ratios.
  - Optionally keeps first augmentation of validation images as original.
- **Save**: Saves images as `.jpg` and labels as `.txt` in the respective `images` and `labels` subfolders.
- **Output**: Produces a YOLO-ready dataset with a `data.yaml` file for training.

## How to Use
### Prerequisites
- **Python**: Version 3.8 or higher.
- **Dependencies**:
  ```bash
  pip install numpy opencv-python pillow torch torchvision tqdm
  ```
- **Data**:
  - Images (e.g., `.jpg`, `.png`, `.jpeg`) in a source folder (e.g., `images`).
  - Corresponding segmentation masks (e.g., `.png`) in another folder (e.g., `render`), with pixel colors mapping to class IDs.
  - Example: White pixels `(255, 255, 255)` map to class `road` (ID 0).

### Prepare Data:
   - Place images in a source directory (e.g., `./SAM2/segment-anything-3/working_dir/images`).
   - Place corresponding masks in another directory (e.g., `./SAM2/segment-anything-3/working_dir/render`).
   - Ensure masks use consistent colors for classes (e.g., `(255, 255, 255)` for `road`).
   - Example structure:
     ```
     working_dir/
     ├── images/
     │   ├── image1.jpeg
     │   ├── image2.jpeg
     ├── render/
     │   ├── image1.png
     │   ├── image2.png
     ```

4. **Update Configuration**:
   - Open `DatasetCreatere.py` and modify the `CONFIG` dictionary to match your setup:
     ```python
     CONFIG = {
         "dataset_path": r"path/to/your/working_dir",
         "SOURCE_mask_folder_name": "render",
         "SOURCE_original_folder_name": "images",
         "SOURCE_mask_type_ext": ".png",
         "SOURCE_img_type_ext": ".jpeg",
         "augment_times": 10,  # Number of augmentations per image
         "test_split": 0.0,
         "val_split": 0.1,
         "train_split": 0.9,
         "Keep_val_dataset_original": True,
         "num_threads": os.cpu_count() - 2,
         "class_to_id": {"road": 0},
         "color_to_label": {(255, 255, 255): 0},
         "dataset_saving_working_dir": r"path/to/save/dataset",
         "folder_name": "road",
         "class_names": ["road"],
         "DESTINATION_img_type_ext": ".jpg",
         "DESTINATION_label_type_ext": ".txt",
         "FromDataType": "",
         "ToDataTypeFormate": "",
     }
     ```

### Running the Tool
1. **Run the Application**:
   ```bash
   python DatasetCreatere.py
   ```
   
2. **Output**:
   - A new dataset folder (e.g., `./SAM2/DatasetManager/road`) is created with:
     ```
     road/
     ├── data.yaml
     ├── train/
     │   ├── images/
     │   │   ├── image1_1.jpg
     │   │   ├── image1_2.jpg
     │   ├── labels/
     │   │   ├── image1_1.txt
     │   │   ├── image1_2.txt
     ├── valid/
     │   ├── images/
     │   ├── labels/
     ├── test/
     │   ├── images/
     │   ├── labels/
     ```
   - `data.yaml` contains:
     ```yaml
     train: ../train/images
     val: ../valid/images
     test: ../test/images
     nc: 1
     names:
       0: road
     ```

## Directory Structure
```
SAM2/
└── DatasetManager/
    └── YolovDatasetManager/
        ├── DatasetCreatere.py            # Main script: processes images, applies augmentations, and saves YOLO dataset
        ├── create_yolo_structure.py      # Creates YOLO folder structure and generates data.yaml
        ├── README.md                     # Overview, setup, and usage instructions
        │
        └── <dataset_output_dir>/         # Root directory for generated YOLO datasets (e.g., "road/")
            ├── data.yaml                 # YOLO dataset configuration file
            ├── train/
            │   ├── images/               # Training images
            │   └── labels/               # Corresponding YOLO labels
            ├── valid/
            │   ├── images/               # Validation images
            │   └── labels/               # Corresponding YOLO labels
            └── test/
                ├── images/               # Test images
                └── labels/               # Corresponding YOLO labels

```

## Troubleshooting
- **Error: Source directories not found**:
  - Ensure `dataset_path` + `SOURCE_original_folder_name` and `SOURCE_mask_folder_name` exist.
  - Example: `F:/RunningProjects/SAM2/segment-anything-3/working_dir/images` and `../render`.
- **Error: File not found**:
  - Verify that each image has a corresponding mask with the correct extension (e.g., `.png` for masks).
  - Check file paths in the `CONFIG` dictionary.
- **Slow Processing**:
  - Reduce `augment_times` or `num_threads` if memory or CPU usage is high.
  - Ensure CUDA is enabled if a compatible GPU is available (`torch.cuda.is_available()`).
- **Invalid Mask Colors**:
  - Ensure `color_to_label` in `CONFIG` matches the RGB colors in your masks (e.g., `(255, 255, 255)` for `road`).
- **Empty Output Folders**:
  - Check logs for errors (e.g., missing files, invalid masks).
  - Ensure `train_split`, `val_split`, and `test_split` sum to 1.0.
- **Memory Issues**:
  - For large datasets, reduce `num_threads` or process in smaller batches by limiting input images.

## Future Improvements
- Support for additional augmentation types (e.g., rotation, flipping).
- Batch processing for multiple dataset directories.
- Validation of mask colors before processing.
- Integration with other annotation formats (e.g., COCO, Pascal VOC).
- GUI for configuration and previewing augmentations.
- Support for video frame extraction as input.