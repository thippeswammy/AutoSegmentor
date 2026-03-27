# SAM 3 Core Module

The `sam3` directory contains the core logic for the AutoSegmentor pipeline. It houses the main execution script, configuration files, and the primary processing loop that integrates SAM 2 for video segmentation.

## 🎬 Main Execution

The primary script for running the pipeline is `sam3_video_predictor_demo.py`.

```powershell
python sam3_video_predictor_demo.py
```

### What logic does this script control?
1.  **Initialization**: Loads configuration, sets up logging, and prepares directories.
2.  **Frame Extraction**: Breaks down input videos into individual frames.
3.  **Batch Processing**: Feeds frames to the SAM 2 model in manageable batches.
4.  **Interactive Annotation**: Launches the GUI for user prompts (clicks/labels) if the model is uncertain.
5.  **Mask Generation**: Producing segmentation masks based on prompts.
6.  **Verification**: Overlays masks on original frames for user approval.
7.  **Video Compilation**: Reassembles processed frames into video previews.

---

## ⚙️ Configuration

The pipeline is controlled by `inputs/config/default_config.yaml`.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `video_start` / `video_end` | `int` | Range of videos to process from the inputs folder. |
| `prefix` | `str` | Prefix for generated filenames (e.g., "Img"). |
| `batch_size` | `int` | Number of frames to process in one GPU pass (Optimise for VRAM). |
| `fps` | `int` | Frame rate for the output video. |
| `delete` | `bool` | Whether to auto-delete temporary files after processing. |
| `auto_prompt_encoding` | `bool` | If `True`, propagates masks from previous frames without new prompts. |

**Example Config:**
```yaml
video_start: 1
video_end: 2
prefix: "Img"
batch_size: 8
fps: 24
delete: false
```

---

## 🎮 Interactive Annotation GUI

When the pipeline pauses for user input, an OpenCV window appears.

**Controls:**
- **Left Click**: Add a Foreground Point (Positive prompt).
- **Right Click**: Add a Background Point (Negative prompt).
- **Tab**: Switch to the next Object Instance ID.
- **1-9**: Select Class Label (e.g., 1=Car, 2=Road).
- **Enter**: Confirm annotations and continue processing.
- **Q**: Quit the annotator.

> **Tip**: Use the "Zoom" window to place precise points on small objects.

---

## 📂 Inputs & Outputs

### Inputs (`sam3/inputs/`)
- **`VideoInputs/`**: Place your raw `.mp4` files here.
- **`UserPrompts/`**: Stores JSON files containing your click history (`points_labels_*.json`).
- **`config/`**: Contains `default_config.yaml`.

### Outputs (`sam3/outputs/`)
After processing, three video files are generated for each input:
1.  **`OrgVideoX.mp4`**: The original video (reconstructed from frames).
2.  **`MaskVideoX.mp4`**: The segmentation masks visualized in color.
3.  **`OverlappedVideoX.mp4`**: The masks overlaid on the original video for verification.

### Working Directory (`sam3/working_dir/`)
Intermediate files are stored here.
- **`images/`**: Extracted raw frames.
- **`render/`**: Generated color masks.
- **`verified/`**: Final, approved images and masks ready for Dataset creation.

---

## 🧩 Architecture Details

For a deeper dive into the utility scripts that power this module (`FrameExtractor`, `MaskProcessor`, `SAM2Model`), please refer to the [Utilities Documentation](utils/README.md).
