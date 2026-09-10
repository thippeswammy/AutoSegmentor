# Assets

Media referenced by the top-level [`README.md`](../README.md) and the docs site: GIFs,
and recorded showcase videos of running the app. Demo *input* footage (what the pipeline
actually processes) lives in [`demo/videos/`](../demo/videos/README.md) instead — generated
datasets, masks, and working files are never committed anywhere.

## Contents

| File | Used in | Description |
| :--- | :--- | :--- |
| `cat_full_pipeline.gif` | README hero image, "Full Pipeline Demo (v3)" | End-to-end walkthrough on the bundled `cat` clip — UI annotation, SAM2 + CoTracker auto-tracking, a live correction, and export to a YOLO dataset. 640×360, 8fps, ~16MB. |
| `road_dashboard_1080.gif` | "Automated Demo" (road demo) | SAM2 segmentation only (no pose tracking) on dashcam footage, from the `road` demo. ~45MB — see note below. |
| `AutoSegmenterCat.mp4` | README "Watch the full recordings", `docs/demos.md` | Full-quality, full-length recording the `cat_full_pipeline.gif` was downsampled from — the source video, not a raw pipeline input (that's `demo/videos/cat.mp4`, a separate, much shorter clip). 1920×1080, 30fps, ~58MB. |
| `AutoSegmenterRoad.mp4` | README "Watch the full recordings", `docs/demos.md` | Full-quality recording behind `road_dashboard_1080.gif`. Byte-identical to `demo/videos/road_dashboard.mp4` — the road demo's raw dashcam footage doubles as its own showcase video. ~88MB. |

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

The source recording (`AutoSegmenterCat.mp4`, 1920×1080, 30fps) is downsampled to 640×360
at 8fps for the GIF — small enough to load quickly in the README while keeping the
on-screen captions legible. The full-quality recording is embedded directly in the README
("▶ Watch the full recordings") and on the [Demos](https://thippeswammy.github.io/AutoSegmentor/demos/)
page.

## Guidelines for adding media here

- **Git LFS is required** for new media added to this repo: run `git lfs install`
  once, then `.gitattributes` (repo root) automatically routes any new `.mp4`/`.gif`
  through LFS instead of a plain git blob.
- **GIFs**: short loops or downsampled full walkthroughs, kept well under
  the size of a typical page load — 30MB is a reasonable ceiling.
  `road_dashboard_1080.gif` (~45MB) predates this guideline and LFS being set up; it's a
  known exception, not a template to follow for new additions.
- **Full-length video**: committing it here via Git LFS (like `AutoSegmenterCat.mp4` /
  `AutoSegmenterRoad.mp4`) is fine for files in the low hundreds of MB. For anything
  approaching GitHub's free LFS quota (1GB storage / 1GB bandwidth per month), prefer a
  [GitHub Release](https://github.com/thippeswammy/AutoSegmentor/releases) asset (up to
  2GB, doesn't count against LFS quota) instead.
- Files already committed before LFS was configured (`cat_full_pipeline.gif`,
  `road_dashboard_1080.gif`, and the clips under `demo/videos/`) remain plain git blobs —
  LFS only applies going forward, to avoid rewriting published repository history.
