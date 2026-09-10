# Assets — README Media

Media referenced by the top-level [`README.md`](../README.md). These are
display assets only (hero GIFs, badges' linked previews) — the actual demo
*input* footage lives in [`demo/videos/`](../demo/videos/README.md), and demo
*output* files are not committed here.

## Current files

| File | Used in | Notes |
| :--- | :--- | :--- |
| `cat_full_pipeline.gif` | README hero image + "Full Pipeline Demo" section | Full v3 walkthrough on the `cat` clip (see shot list below). 640×360, 8fps, ~16MB. |
| `road_dashboard_1080.gif` | "Automated Demo" section (road demo link) | Looping preview of the `road` demo (segmentation + pose on dashcam footage). ~45MB — kept because Git LFS isn't set up; see size guidance below. |

## The v3 "full pipeline" demo (cat clip) — shot list

Source recording: 1920×1080, 30fps, 51s, captions burned in during editing.
This is the exact sequence recorded; keep it as the reference if the clip is
ever re-recorded.

| # | On-screen caption | What it shows |
| :-: | :--- | :--- |
| 1 | Launch System | Launch AutoSegmentor |
| 2 | Initial Prompt: 5 Foreground + 2 Background Points | Click 5 foreground + 2 background points on the cat in frame 1 |
| 3 | Process Batch — SAM2 + CoTracker Take Over | Press Enter: SAM2 mask appears, CoTracker keypoints start tracking |
| 4 | Auto-Tracking Across Frames | Navigate forward (D / Right) — mask + skeleton follow the cat automatically |
| 4b | One-Click Correction | Drag a drifted keypoint back into place |
| 5 | Process Next Batch | Press Enter again to process the next batch |
| 6 | Keep Navigating — Still Tracking | Navigate forward through the remaining frames |
| 7 | Save & Export | Ctrl+S to save, then Ctrl+E to export |
| 8 | YOLO Dataset, Ready | Generated dataset: train / valid / test folders + data.yaml |
| 9 | Ready to train — for any application. | Closing line |

### Still open

- The full-length source recording (`Video Project 24.mp4`, ~58MB, not
  committed to the repo) should be attached as a **GitHub Release asset** on
  the `v3.0.0` release rather than hosted externally — no repo bloat, no
  third-party dependency, and GitHub Releases accept files up to 2GB.
  Once the release is published, replace the "Demo Video" badge URL at the
  top of `README.md` (currently an old Google Drive link) with the release
  asset URL, and fill in the "link pending" note in the README's "Full
  Pipeline Demo" section.

## Regenerating the GIF

No `ffmpeg` was available in this environment, so the GIF was built with
OpenCV (frame decode/resize) + `imageio` (GIF encode) instead — downsampled
from the 1920×1080/30fps source to 640×360/8fps. To reproduce or tune it,
adjust `target_fps` / `target_width` and re-run the same decode → resize →
`imageio.mimsave` steps against the source recording.

### Size guidance

Git has no native diffing for binary media, so every GIF/video committed here
permanently bloats the repo history. Prefer:
- GIFs: short loops or downsampled full walkthroughs, <30MB.
- Full-length videos: attach as a GitHub Release asset (preferred) or host
  externally and link via badge — do not commit MP4s to `assets/`.
