# Demos

AutoSegmentor ships with two bundled demos that run the entire pipeline — frame extraction,
SAM2 prompting, CoTracker3 keypoint tracking, pose export, YOLO export — on real footage,
with no configuration required.

```bash
python run_main.py --demo list           # cat, road
python run_main.py --demo                # default (cat)
python run_main.py --demo cat
python run_main.py --demo road
```

## What a full run looks like

![Full AutoSegmentor pipeline: annotate, auto-track, export](https://raw.githubusercontent.com/thippeswammy/AutoSegmentor/master/assets/cat_full_pipeline.gif)

This is a real, unedited session on the `cat` demo, start to finish:

1. Launch the annotation tool.
2. Click 5 foreground + 2 background points on the subject, in the first frame only.
3. Press **Enter** — SAM2 generates the mask, CoTracker3 starts tracking pose keypoints.
4. Navigate forward — the mask and skeleton follow the subject automatically. Drag a
   drifted keypoint back into place if needed; nothing else needs re-annotating.
5. Process the next batch, keep navigating — tracking continues without new prompts.
6. `Ctrl+S` to save, `Ctrl+E` to export.
7. Result: a YOLO dataset (`train` / `valid` / `test` + `data.yaml`) with detection,
   instance segmentation, and pose labels, all from the same handful of clicks.

The full-length narrated recording is published as a
[GitHub Release](https://github.com/thippeswammy/AutoSegmentor/releases) asset.

## The two bundled demos

| Demo | Footage | Shows off |
| :--- | :--- | :--- |
| `cat` (default) | Close-up clip of a cat | Classic single-subject segmentation + 6-point pose tracking. |
| `road` | Dashcam recording | Segmentation and pose tracking on a moving-camera outdoor scene. |

![Road dashcam demo: segmentation and pose tracking from a moving camera](https://raw.githubusercontent.com/thippeswammy/AutoSegmentor/master/assets/road_dashboard_1080.gif)

Each demo is just a session-state JSON file under `demo/` — the app discovers demos by
scanning that directory, so there's no hardcoded list to update when a new one is added.

## Running it on your own video

1. Drop your video into `demo/videos/`.
2. Duplicate an existing `*_session_state.json` in `demo/` and point its
   `video_inputs.template` at your file.
3. Edit `pose.classes` to match the keypoints you want tracked (for example, the corners of
   a box, or the joints of a person).
4. Give it a unique `demo.name`, then run:

   ```bash
   python run_main.py --demo <your_name>
   ```

This works for any subject — the classic use case beyond the bundled clips is
**warehouse and logistics automation**: pallets, forklifts, boxes, or rolls, tracked well
enough from a handful of clicks to train a custom pose-estimation model without hand-labeling
thousands of frames.

## Footage licensing

Both bundled clips are original project assets — see the
[repo's demo footage notes](https://github.com/thippeswammy/AutoSegmentor/blob/master/demo/videos/README.md)
for details.
