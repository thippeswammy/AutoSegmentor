# Dataset Manager

This directory contains specialized tools for managing, converting, and synthesizing datasets for YOLO object detection and pose estimation models. It serves as the post-processing hub after the AutoSegmentor annotation pipeline.

[< Back to Project Root](../README.md)

## 📂 Components

### 1. YOLO Dataset Manager (`YolovDatasetManager/`)
A comprehensive toolset for creating YOLOv8/v11-ready datasets from raw segmentation masks.
- **Key Features**: 
    - Automated train/val/test splitting.
    - Mask-to-Polygon conversion for instance segmentation.
    - `data.yaml` generation for seamless training integration.
- **Documentation**: [Read More](./YolovDatasetManager/README.md)

### 2. Synthetic Engine (`SyntheticEngine/`)
An advanced augmentation pipeline that creates massive training datasets from a small set of annotated reference frames.
- **Key Features**: 
    - Copy-Paste augmentation with lighting adaptation.
    - Simulated occlusions and visibility tracking.
    - Geometric transformations synced across images, masks, and keypoints.
- **Documentation**: [Read More](./SyntheticEngine/README.md)

### 3. Dataset Handler (`DatasetHandler/`)
Low-level utilities for raw data manipulation and preprocessing.
- **`Video2images.py`**: A fast, standalone script for extracting frames from video files.
