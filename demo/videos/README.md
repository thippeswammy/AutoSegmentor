# Demo Videos — Sources & Licensing

This directory holds the footage used by the bundled AutoSegmentor demo.

## Included clips

| File | Source | Description | Resolution | License |
| :--- | :--- | :--- | :--- | :--- |
| `cat.mp4` | Original project demo asset | Cat video used for the classic segmentation/pose demo | — | Project asset |
| `road_dashboard.mp4` | Original project asset | Road scene recorded from a dashboard camera (extra sample footage, not part of a demo) | — | Project asset |

## Adding your own footage

Replace the `template` path in the demo session-state JSON (see
`demo/README.md`) with your own video and run `python run_main.py --demo <name>`.