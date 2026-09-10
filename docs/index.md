# AutoSegmentor

**Turn raw video into a YOLO-ready dataset — with a handful of mouse clicks, not
frame-by-frame labeling.**

![AutoSegmentor: click a few points, SAM2 and CoTracker3 do the rest](https://raw.githubusercontent.com/thippeswammy/AutoSegmentor/master/assets/cat_full_pipeline.gif)

AutoSegmentor bridges raw footage and structured training data. Point at an object once —
Meta AI's **[Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2)**
propagates a pixel-accurate mask across every following frame, while
**[CoTracker3](https://github.com/facebookresearch/co-tracker)** follows its keypoints for
pose estimation — and export a complete dataset (detection, instance segmentation, and pose,
all at once) in one keystroke.

## Why it exists

Hand-labeling video for computer vision is slow: thousands of frames, each needing a mask
and a set of keypoints. AutoSegmentor collapses that into a few clicks on a single frame,
then lets state-of-the-art tracking models carry the annotation forward — a human stays in
the loop to correct drift, not to redraw every frame from scratch.

## What you get

- **An interactive PyQt5 annotation tool** — point-and-click prompts, live mask preview,
  undo/redo, per-object keyframe correction.
- **Automatic propagation** across long videos in GPU-friendly batches, so hour-long
  footage runs on consumer hardware.
- **One export, three label types** — YOLO-format detection (bbox), instance segmentation,
  and pose, generated together from the same verified annotations.
- **Two bundled demos** (a classic object clip and a dashcam road scene) that run the whole
  pipeline end to end with zero configuration.

## Get started

```bash
git clone --recursive https://github.com/thippeswammy/AutoSegmentor.git
cd AutoSegmentor
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\Activate.ps1 on Windows
python install.py
python run_main.py --demo cat
```

Head to **[Installation](installation.md)** for the full walkthrough (Windows and Ubuntu),
or straight to **[Demos](demos.md)** to see what running it actually looks like.

## Explore the docs

<div class="grid cards" markdown>

- **[Installation](installation.md)** — set up on Windows or Ubuntu, one command or manual.
- **[Demos](demos.md)** — watch the full pipeline run, then try it on your own footage.
- **[Architecture](architecture.md)** — how the UI, SAM2, CoTracker3, and export actually
  connect.
- **[Dataset Manager](dataset-manager.md)** — turning verified annotations into a
  training-ready YOLO dataset, plus large-scale synthetic augmentation.

</div>

---

Source on [GitHub](https://github.com/thippeswammy/AutoSegmentor).
