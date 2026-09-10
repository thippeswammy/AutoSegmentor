# Assets

Media referenced by the top-level [`README.md`](../README.md). Demo *input*
footage lives in [`demo/videos/`](../demo/videos/README.md); demo *output*
(datasets, masks, working files) is never committed.

## Contents

| File | Used in | Description |
| :--- | :--- | :--- |
| `cat_full_pipeline.gif` | README hero image, "Full Pipeline Demo (v3)" | End-to-end walkthrough on the bundled `cat` clip — UI annotation, SAM2 + CoTracker auto-tracking, a live correction, and export to a YOLO dataset. 640×360, 8fps, ~16MB. |
| `road_dashboard_1080.gif` | "Automated Demo" (road demo) | Segmentation + pose tracking on dashcam footage, from the `road` demo. |

## About the full pipeline demo

`cat_full_pipeline.gif` is a captioned recording of a single annotation
session, covering the whole workflow in one continuous pass:

| Step | Caption | What happens |
| :-: | :--- | :--- |
| 1 | Launch System | Launch AutoSegmentor |
| 2 | Initial Prompt: 5 Foreground + 2 Background Points | Click 5 foreground + 2 background points on the cat in frame 1 |
| 3 | Process Batch — SAM2 + CoTracker Take Over | SAM2 generates the mask; CoTracker starts tracking pose keypoints |
| 4 | Auto-Tracking Across Frames | Mask and skeleton follow the cat automatically across frames |
| 5 | One-Click Correction | A drifted keypoint is dragged back into place |
| 6 | Process Next Batch | Tracking continues into the next batch with no extra prompts |
| 7 | Save & Export | `Ctrl+S` to save, `Ctrl+E` to export |
| 8 | YOLO Dataset, Ready | Output: `train` / `valid` / `test` folders + `data.yaml` |
| 9 | Ready to train — for any application. | Closing line |

The source recording (1920×1080, 30fps, ~51s) is downsampled to 640×360 at
8fps for the GIF — small enough to load quickly in the README while keeping
the on-screen captions legible. A full-length, narrated version is attached
to the [`v3.0.0` release](https://github.com/thippeswammy/AutoSegmentor/releases).

## Guidelines for adding media here

- **GIFs**: short loops or downsampled full walkthroughs, kept well under
  the size of a typical page load — 30MB is a reasonable ceiling.
- **Full-length video**: attach it as a GitHub Release asset (up to 2GB,
  no external dependency) rather than committing an `.mp4` to this folder
  or the repository's history.
- Git has no meaningful diffing for binary media, so every file committed
  here stays in the repository's history permanently — keep only what the
  README actually displays.
