# AutoSegmentor Demos

AutoSegmentor ships with a classic automated demo that exercises the full
auto-labeling pipeline (frame extraction → SAM2 prompting → CoTracker3 keypoint
tracking → pose export → YOLO dataset export) without manual configuration.

## Running a demo

```bash
# List available demos
python run_main.py --demo list

# Run the default (cat) demo
python run_main.py --demo

# Run a specific demo
python run_main.py --demo cat
python run_main.py --demo road
```

## Available demos

| Name | Config file | Footage | What it showcases |
| :--- | :--- | :--- | :--- |
| `cat` | `demo_session_state.json` | `videos/cat.mp4` | Full pipeline: SAM2 segmentation + 5-point CoTracker3 pose tracking |
| `road` | `road_demo_session_state.json` | `videos/road_dashboard.mp4` | SAM2 segmentation **only** — no CoTracker pose tracking. Kept from the v1/v2 lineup; it was never updated to the full v3 SAM2+CoTracker pipeline. |

## Adding your own footage

To demo on your own video:

1. Copy your `.mp4` into `demo/videos/`.
2. Duplicate `demo_session_state.json` and point `video_inputs.template` at
   your file.
3. Adjust `pose.classes` keypoints to match the object you want to label.
4. Give the `demo.name` a unique id, then run
   `python run_main.py --demo <your_name>`.

## Video licensing

Demo clips are original project assets. Full attribution and licensing details
are in [`videos/README.md`](videos/README.md).