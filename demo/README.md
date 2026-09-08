# AutoSegmentor Demos

AutoSegmentor ships with named, automated demo scenarios that exercise the full
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
python run_main.py --demo industrial-warehouse
```

## Available demos

| Name | Config file | Footage | What it showcases |
| :--- | :--- | :--- | :--- |
| `cat` | `demo_session_state.json` | `videos/cat.mp4` | Classic segmentation + 6-point pose keypoint tracking |
| `industrial-warehouse` | `industrial_demo_session_state.json` | `videos/forklift_pallets_warehouse.mp4` | Moving forklift + pallets: pallet/forklift detection and 6-point pose tracking |
| `pallet-closeup` | `pallet_closeup_session_state.json` | `videos/pallet_stack_closeup.mp4` | **Static wooden pallet close-up** — create your own pallet labels to train a pose model |

## Material handling demo (industrial)

The `industrial-warehouse` demo is the flagship example for training a **pallet
pose model** used in warehouse / logistics automation. It processes real forklift
+ pallet footage and produces:

- Segmentations (SAM2) of pallets and the forklift
- 6-point pose keypoints per pallet (corners + center + top) tracked with CoTracker3
- 3-point pose keypoints for the forklift (forks, mast, top)
- A YOLO-compatible dataset ready to train a pose model

### Adding your own footage

To demo on your own warehouse / pallet / box videos:

1. Copy your `.mp4` into `demo/videos/`.
2. Duplicate `industrial_demo_session_state.json` and point
   `video_inputs.template` at your file.
3. Adjust `pose.classes` keypoints to match the object you want to label.
4. Give the `demo.name` a unique id, then run `python run_main.py --demo <your_name>`.

## Video licensing

Demo clips are royalty-free (Pexels License / Pixabay Content License). Full
attribution and licensing details are in [`videos/README.md`](videos/README.md).
