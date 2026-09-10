# Demos

AutoSegmentor ships with two bundled demos, no configuration required. `cat` runs the full
pipeline (SAM2 + CoTracker3 + pose export + YOLO export); `road` is a segmentation-only demo
carried over from earlier versions — see the table below.

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

**▶ Watch the full recording** (full quality, longer than the GIF above):

<video src="https://raw.githubusercontent.com/thippeswammy/AutoSegmentor/master/assets/AutoSegmenterCat.mp4" controls width="600"></video>

## The two bundled demos

| Demo | Footage | Shows off |
| :--- | :--- | :--- |
| `cat` (default) | Full-HD clip of a cat | Full pipeline: single-subject segmentation + 5-point CoTracker3 pose tracking. |
| `road` | Dashcam recording | SAM2 segmentation **only** — no pose tracking. Carried over from v1/v2; never updated to the full v3 SAM2+CoTracker pipeline. |

![Road dashcam demo: SAM2 segmentation on a moving camera](https://raw.githubusercontent.com/thippeswammy/AutoSegmentor/master/assets/road_dashboard_1080.gif)

**▶ Watch the full recording:**

<video src="https://raw.githubusercontent.com/thippeswammy/AutoSegmentor/master/assets/AutoSegmenterRoad.mp4" controls width="600"></video>

Each demo is just a session-state JSON file under `demo/` — the app discovers demos by
scanning that directory, so there's no hardcoded list to update when a new one is added.

## Running it on your own video

For everyday use on your own footage, skip demos entirely: drop it in
`workspace/VideoInputs/` and run `python run_main.py` (no `--demo`) — see
[Running on your own video](installation.md#running-on-your-own-video-not-a-demo) for what
the Setup Dialog that opens lets you configure.

The steps below are for turning your footage into a **named, repeatable demo** instead
(useful for sharing a fixed showcase, like the bundled `cat`/`road` demos, rather than a
one-off run):

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
