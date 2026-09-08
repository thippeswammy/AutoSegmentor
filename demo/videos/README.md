# Demo Videos — Sources & Licensing

The industrial / material-handling demo clips below are **royalty-free** and free to
download and use. They are included to demonstrate AutoSegmentor's auto-labeling
pipeline on realistic warehouse footage. When you publish derived models or datasets,
please follow the respective license terms and attribute the sources.

## Included clips

| File | Source | Description | Resolution | License |
| :--- | :--- | :--- | :--- | :--- |
| `forklift_pallets_warehouse.mp4` | [Pexels – 6194507](https://www.pexels.com/video/forklift-in-a-warehouse-6194507/) (Andi Farruku) | A forklift organizing large stacks of pallets in a warehouse | 1920×1080, 30 fps | Pexels License (free to use, no attribution required) |
| `forklift_loader_warehouse.mp4` | [Pixabay – 43628](https://pixabay.com/videos/forklift-loader-warehouse-vehicle-43628/) (GreenCardShow) | Forklift / loader operating in a warehouse | 1920×1080, 25 fps | Pixabay Content License |
| `pallet_stack_closeup.mp4` | [Pixabay – 187954](https://pixabay.com/videos/pallet-stack-timber-industry-wood-187954/) (Oleh_77) | **Static close-up of a wooden pallet stack** — ideal for creating your own pallet labels | 1920×1080, 30 fps | Pixabay Content License |
| `cat.mp4` | Original project demo asset | Cat video used for the classic segmentation/pose demo | — | Project asset |

## Adding your own footage

Replace the `template` path in the relevant demo session-state JSON (see
`demo/README.md`) with your own warehouse, pallet, forklift, or conveyor video and
run `python run_main.py --demo <name>`.
