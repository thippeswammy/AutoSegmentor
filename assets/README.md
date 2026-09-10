# Assets — README Media

Media referenced by the top-level [`README.md`](../README.md). These are
display assets only (hero GIFs, badges' linked previews) — the actual demo
*input* footage lives in [`demo/videos/`](../demo/videos/README.md), and demo
*output* files are not committed here.

## Current files

| File | Used in | Notes |
| :--- | :--- | :--- |
| `road_dashboard_1080.gif` | README hero image | Looping preview of the `road` demo (segmentation + pose on dashcam footage). ~45MB — kept because Git LFS isn't set up; see size guidance below. |
| `cat_full_pipeline.gif` *(pending)* | README "Full Pipeline Demo" section | **Not added yet.** See checklist below. |

## Adding the new v3 "full pipeline" demo (cat clip)

This should show, in one pass: annotating the bundled `demo/videos/cat.mp4` in
the UI → CoTracker keypoint tracking across frames → SAM2 mask propagation →
exporting the verified session straight to YOLO format for **all three**
target model types (detection/bbox, instance segmentation, pose), via
`DatasetManager/YolovDatasetManager`.

1. **Record** a screen capture running through:
   - Launching `python run_main.py`, loading the `cat` demo (or a fresh
     session on `demo/videos/cat.mp4`).
   - A couple of manual annotation clicks (box/points), then CoTracker
     picking up keypoints across the batch and SAM2 propagating the mask.
   - `Ctrl+E` (export) producing the YOLO dataset folder, briefly showing the
     `train/valid/test` structure and `data.yaml`.
2. **Export two files**:
   - A short (10–20s) looping **GIF** for the README hero/section —
     name it `cat_full_pipeline.gif`, similar resolution/size to
     `road_dashboard_1080.gif` (trim it further if possible; keep under ~30MB).
   - The **full-length MP4** (with narration/captions if you have them) —
     host it externally (YouTube, Google Drive, etc.) rather than committing
     it here; the repo already links external video via the "Demo Video"
     badge at the top of the README.
3. **Drop the GIF** in this folder as `assets/cat_full_pipeline.gif`.
4. **Update `README.md`**:
   - Point the "🎬 Full Pipeline Demo" section's `![...]` image at the new
     GIF.
   - Replace/add a `Demo Video` shields.io badge pointing at the hosted
     full-length video link.
5. Update this table's "pending" row to reflect the real file.

### Size guidance

Git has no native diffing for binary media, so every GIF/video committed here
permanently bloats the repo history. Prefer:
- GIFs: short loops, trimmed/cropped to the relevant region, <30MB.
- Full-length videos: host externally, link via badge — do not commit MP4s
  to `assets/`.
