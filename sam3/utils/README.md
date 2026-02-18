# Utilities and Core Logic

This directory contains the foundational building blocks of the AutoSegmentor pipeline. It is organized into logical components that handle specific responsibilities, from file management to model inference.

## 🏗️ Core Pipeline

### `pipeline.py`
The central orchestrator script.
- **Function**: `run_pipeline(config)`
- **Responsibility**: Connects all utility modules. It calls `FrameExtractor`, initiates `SAM2VideoProcessor`, manages the `UserInteraction` loop, and triggers `VideoCreator`.

---

## 📦 Component Details

### 1. Model Logic (`Model/`)
Wrappers around the SAM 2 architecture.

- **`SAM2Model.py`**:
    - Initializes the SAM 2 predictor.
    - Manages GPU/CPU device selection.
    - Sets bfloat16/float32 precision based on hardware.
- **`SAM2Config.py`**:
    - Central configuration class.
    - Resolves paths for checkpoints and configs.
    - Defines color palettes for segmentation masks.
- **`sam2_video_predictor.py`**:
    - The engine room of the project.
    - Manages the inference loop.
    - Handles prompt encoding (point clicks to mask inputs).
    - Manages the state of the SAM 2 inference session.

### 2. User Interface (`UserUI/`)
Tools for the interactive OpenCV GUI.

- **`UserInteraction.py`**:
    - Handles mouse callbacks (Left/Right clicks).
    - Captures keyboard inputs (Labels, Navigation).
    - Draws the UI overlay (Points, Text, Zoom window).
- **`AnnotationManager.py`**:
    - Saves and loads user prompts to JSON.
    - Ensures annotations are persistent across sessions.

### 3. File Management (`FileManagement/`)
Helpers for filesystem operations.

- **`FrameExtractor.py`**: Reads video files and saves frames as JPEGs.
- **`FrameHandler.py`**: Batches frames for processing to manage memory.
- **`MaskProcessor.py`**:
    - Runs the actual mask prediction on a batch.
    - Converts binary masks to color masks.
- **`ImageOverlayProcessor.py`**: Blends masks onto original images for verification.
- **`VideoCreator.py`**: Reconstructs video files from processed image frames.
- **`FileManager.py`**: General path manipulation and directory cleanup utilities.

---

## 🔄 Data Flow

1. **Extraction**: `FrameExtractor` -> `images/`
2. **Batching**: `FrameHandler` -> `temp/`
3. **Inference**: `sam2_video_predictor` + `MaskProcessor` -> `render/`
4. **Verification**: `ImageOverlayProcessor` -> `overlap/`
5. **Finalization**: verified images -> `verified/` -> `VideoCreator` -> `outputs/`
