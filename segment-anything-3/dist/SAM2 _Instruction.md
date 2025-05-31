
# SAM2 Video Processing Pipeline

This guide outlines the setup and usage instructions for running the **SAM2 video processing pipeline**. It includes video input preparation, execution steps, argument descriptions, and annotation/verification modes.

---

## Folder Structure

Prepare the following structure in **any folder** (referred to here as `anyFolder/`):

```plaintext
anyFolder/
├── sam2_video_predictor.exe
├── VideoInputs/
│   ├── Video1.mp4
│   ├── Video2.mp4
│   └── ...
```

Also, make sure to:

1. Press `Win + R`, type `%Temp%`, and press Enter.
2. Paste the following folders inside the temp directory:

   * `checkpoints`
   * `sam2_configs`
3. Place `VideoInputs/` and `sam2_video_predictor.exe` inside your working folder (`anyFolder/`).

---

## ▶ Running the SAM2 Pipeline

Open **Command Prompt** and navigate to the working directory:

```bash
cd path\to\anyFolder
```

### Simple Run Example

To process a single video:

```bash
sam2_video_predictor.exe --video_start 1 --batch_size 120
```

This will process `VideoInputs/Video1.mp4`.

---

## ⚙️ Full Configuration Example

```bash
sam2_video_predictor.exe ^
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

---

## 🧾 Argument Descriptions

| Argument                | Description                                                                |
| ----------------------- |----------------------------------------------------------------------------|
| `--video_start`         | Starting video number (inclusive, default: `99`)                           |
| `--video_end`           | Ending video number (exclusive, default: `1`)                              |
| `--prefix`              | Prefix for output filenames (default: `Img`)                               |
| `--batch_size`          | Number of frames to process per batch (default: `10`)                      |
| `--fps`                 | Frames per second for output videos (default: `30`)                        |
| `--delete`              | Auto-delete working directory without prompt (`yes`/`no`, default: `yes`)  |
| `--working_dir_name`    | Base directory for working files (default: `working_dir`)                  |
| `--video_path_template` | Template for input video file paths (default: `.\VideoInputs\Video{}.mp4`) |
| `--images_extract_dir`  | Directory for extracted frames                                             |
| `--temp_processing_dir` | Temporary directory for batch processing                                   |
| `--rendered_dir`        | Directory for rendered masks                                               |
| `--overlap_dir`         | Directory for mask overlays                                                |
| `--verified_img_dir`    | Directory for verified original images                                     |
| `--verified_mask_dir`   | Directory for verified mask images                                         |
| `--final_video_path`    | Output directory for processed videos                                      |

---

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

* Generated overlays are saved in:

  ```
  .\working_dir\overlap\
  ```
* If `--delete no`, verification is prompted before moving results to:

  ```
  .\working_dir\verified\images\
  .\working_dir\verified\mask\
  ```
* Generates 3 output videos:

  * Original Frames
  * Predicted Masks
  * Overlays (masks on frames)

---

## 📦 Dependencies & Notes

* No installation is needed for `.exe` usage.
* Ensure `checkpoints` and `sam2_configs` are correctly placed in `%Temp%`.
* Name your input files as: `Video1.mp4`, `Video2.mp4`, ..., `Video{n}.mp4`


## Output

Outputs will be saved in the following structure by default:

```plaintext
outputs/
├── Video1_original.mp4
├── Video1_masks.mp4
├── Video1_overlay.mp4
...
```