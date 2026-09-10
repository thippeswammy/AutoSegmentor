# Demo Videos — Sources & Licensing

This directory holds the footage used by the bundled AutoSegmentor demo.

## Included clips

| File | Source | Description | Resolution | License |
| :--- | :--- | :--- | :--- | :--- |
| `cat.mp4` | Original project demo asset | Cat video used for the classic segmentation/pose demo | — | Project asset |
| `road_dashboard.mp4` | Original project asset | Road scene recorded from a dashboard camera, used for the `road` demo | — | Project asset |

## Adding your own footage

Replace the `template` path in the demo session-state JSON (see
`demo/README.md`) with your own video and run `python run_main.py --demo <name>`.

New video files are tracked via **Git LFS** (`.gitattributes` at the repo root covers
`*.mp4`/`*.gif`) — run `git lfs install` once per machine before adding footage here.
The two clips already in this directory predate LFS and remain plain git blobs.