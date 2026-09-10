# AutoSegmentor

[![GitHub](https://img.shields.io/github/stars/thippeswammy/AutoSegmentor?style=social)](https://github.com/thippeswammy/AutoSegmentor)
[![Demo Video](https://img.shields.io/badge/Demo-Video-blue)](https://drive.google.com/file/d/1Y19lwf_IIuzwVe-3j9vX0uicV_iWbrHZ/view?usp=sharing)

![AutoSegmentor full pipeline demo: UI annotation, SAM2 auto-tracking, and YOLO export](./assets/cat_full_pipeline.gif)

_AutoSegmentor is a state-of-the-art auto-labeling ecosystem that bridges the gap between raw video footage and structured AI datasets. By integrating Meta AI's **Segment Anything Model 2 (SAM2)** with high-precision tracking like **CoTracker**, it enables users to generate pixel-perfect masks and pose estimation data for long, complex videos with minimal manual interaction._

**Main purpose:**  
Build an end-to-end auto-labeling pipeline that converts raw videos into structured YOLO-compatible datasets using SAM2, with real-time segmentation enabled by CUDA acceleration, multithreading, and an interactive PyQt5 GUI.

---

## ✨ Features

- **Professional Desktop UI**: A fully-featured PyQt5 application with multi-window support, integrated property panels, and real-time visualization.
- **Automated Frame Extraction**: Robust extraction from any video format, handling long-form content with ease.
- **Interactive Annotation**: Point and box-based multi-class annotation with a high-fidelity zoom system for precision.
- **Advanced Tracking (CoTracker)**: Integrated CoTracker support for tracking keypoints across frames with high accuracy—a robust alternative to Optical Flow for complex scenes.
- **Real-time Mask Propagation**: Propagate annotations across batches of frames using SAM2's temporal memory.
- **Async Processing Engine**: Background execution of GPU tasks ensures the UI remains responsive even during heavy inference.
- **YOLO Dataset Creation**: Seamless conversion of verified masks into YOLOv8/v11 formats for **object detection (bbox)**, **instance segmentation**, and **pose estimation** simultaneously, with integrated data augmentation (blur, noise, color jitter).
- **Comprehensive Workspace Management**: Smart handling of project lifecycles, from raw input to verified output, with automatic directory cleanup.

---

## 🔧 Setup & Installation

Tested on **Windows 11** and **Ubuntu 22.04/24.04**.

### Prerequisites

- **Python**: 3.10+ (3.10 recommended)
- **GPU**: NVIDIA GPU with CUDA 12.x (required for SAM2/CoTracker performance)
- **RAM**: 16GB+ recommended
- **Git**: with [Git LFS](https://git-lfs.com/) installed (`git lfs install`) — required for
  contributors adding new demo videos/GIFs; not needed just to run the app
- **Disk space**: ~20GB (checkpoints + working directories)

### 1. Clone the Repository

Cloning with submodules fetches the vendored CoTracker3 dependency (SAM2 is vendored
directly in the repo, not a submodule):

```bash
git clone --recursive https://github.com/thippeswammy/AutoSegmentor.git
cd AutoSegmentor

# If you cloned without --recursive, initialize submodules manually:
# git submodule update --init --recursive
```

### 2. Create and Activate a Virtual Environment

| | Windows (PowerShell) | Ubuntu / Linux |
| :--- | :--- | :--- |
| Create | `python -m venv .venv` | `python3 -m venv .venv` |
| Activate | `.\.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |

(Windows `cmd.exe` instead of PowerShell: `.\.venv\Scripts\activate.bat`)

### 3. One-Shot Setup

```bash
python install.py
```

This single cross-platform script (no separate `.bat`/`.sh` needed) does everything else:
checks your Python version, initializes git submodules, installs
`requirements-core.txt`, downloads the SAM2 + CoTracker3 checkpoints, and runs a GPU/CUDA
diagnostic. Add `--cuda` to also install the optional `flash-attn` speedup (requires a
matching CUDA build toolchain — more commonly available on Linux; skip it on Windows unless
you already have that toolchain set up).

Useful flags: `--skip-checkpoints` / `--skip-gpu-check` / `--skip-deps` /
`--skip-submodules` to skip a step, or run just one part standalone:

```bash
python install.py --checkpoints-only            # download missing checkpoints
python install.py --checkpoints-only --check    # verify presence only
python install.py --gpu-check-only              # just the GPU/system diagnostic
```

<details>
<summary>Prefer to do it manually? (equivalent to what <code>install.py</code> automates)</summary>

```bash
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

The app needs two weight files at runtime (not bundled in the repo for size reasons):

| Model | Checkpoint | Download from (original source) |
| :--- | :--- | :--- |
| SAM2 | `sam2_hiera_large.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt (repo: [facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)) |
| CoTracker3 | `scaled_offline.pth` | https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth (repo: [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)) |

Place them at:

```bash
external/segment_anything_2/checkpoints/sam2_hiera_large.pt
external/co-tracker/checkpoints/scaled_offline.pth
```

Neither CoTracker3 nor SAM2 is `pip install`ed: both are vendored under `external/` and
`run_main.py` adds them to `sys.path` at launch (reading `external_libs` from
`workspace/inputs/config/default_config.yaml`), so just make sure the folders are present.

</details>

### 4. Install AutoSegmentor as a package (optional)

```bash
pip install -e .
```

This registers the `autosegmentor` package and the `autosegmentor` /
`autosegmentor-demo` console entry points.

### 5. Verify the installation

```bash
python run_main.py --version     # prints: AutoSegmentor 3.0.0
python run_main.py --demo list   # lists the bundled demos
```

### Linux troubleshooting: Qt platform plugin

On a minimal Ubuntu install, PyQt5 can fail to launch with
`"could not load the Qt platform plugin xcb"`. Install the missing system libraries:

```bash
sudo apt-get install libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0
```

---

## 🚀 Usage Guide

### 1. Preparing Workspace
- Place your videos in `workspace/VideoInputs/`.
- Ensure your configuration is set in `workspace/inputs/config/default_config.yaml`.

### 2. Launching the Application

Start the standard interactive GUI for annotation and project management:

```bash
python run_main.py
```

Check the version:

```bash
python run_main.py --version
```

Run the **Automated Demo Pipeline** on sample video data — list demos then run one:

```bash
python run_main.py --demo list           # List available demos
python run_main.py --demo                # Default (cat) demo
python run_main.py --demo cat            # Cat demo
python run_main.py --demo road           # Road/dashcam demo
```

---

## 🎬 Automated Demo & Full Pipeline Walkthrough

The GIF at the top of this README is a real, unedited annotation session on the bundled
`cat` clip (annotate → SAM2 auto-mask → CoTracker3 pose tracking → YOLO export) run via
`python run_main.py --demo cat` (see the Usage Guide above for the full `--demo` command
list). The `road` demo runs the same pipeline on a dashcam clip.

- [`demo/README.md`](demo/README.md) — what each demo is, session-state configs, adding
  your own footage.
- [`demo/videos/README.md`](demo/videos/README.md) — bundled footage and licensing.
- [`assets/README.md`](assets/README.md) — shot-by-shot GIF breakdown and the full-length
  narrated recording (published as a [`v3.0.0` release](https://github.com/thippeswammy/AutoSegmentor/releases) asset).

---

## 🏭 Material Handling & Industrial Automation

AutoSegmentor is a good fit for **warehouse / logistics automation** —
detecting and tracking industrial objects (pallets, forklifts, boxes, rolls)
so you can train custom **pose estimation** models from video with almost no
manual labeling.

No industrial demo video is bundled, but the workflow below applies to any
object — pallets, stillage, boxes, EPAL (euro-pallets), forklifts, etc.:

1. Drop your warehouse / production-line video into `demo/videos/`.
2. Make a copy of the demo session-state JSON in `demo/` and point
   `video_inputs.template` at your file.
3. Edit `pose.classes` → add your class ids and keypoint names (e.g. the 8
   corners of a euro-pallet, or box corners for pick-and-place).
4. Register the demo by giving it a unique `demo.name`:
   ```bash
   python run_main.py --demo <your_name>
   ```
5. Launch the app in that demo, label the first object instance, let
   SAM2 + CoTracker3 auto-propagate, then export the pose labels and train
   with `DatasetManager/` (or `ultralytics` YOLO-Pose) to get an inference
   model for your picking / placement / inspection automation.

### 3. Annotation Controls (Keyboard & Mouse)

| Action | Control |
| :--- | :--- |
| **Foreground Point** | Left Click |
| **Background Point** | Right Click |
| **Undo Action** | `Ctrl + Z` or `U` |
| **Redo Action** | `Ctrl + Y` |
| **Navigate Frames** | `A` / `D` or `Left` / `Right` |
| **Turbo Scroll** | `Shift + A` / `Shift + D` |
| **Batch Navigation** | `[` / `]` |
| **Change Class (1-10)** | Keys `1` to `0` |
| **Instance Management** | `Tab` (Next) / `Shift + Tab` (Prev) |
| **Toggle Mask Overlay** | `M` |
| **Toggle Corner Zoom** | `Z` |
| **Reset Frame** | `R` |
| **Process Batch** | `Enter` / `Return` |
| **Save Progress** | `Ctrl + S` |
| **Export Dataset** | `Ctrl + E` |

---

## 🏗️ System Architecture

AutoSegmentor is a reactive, UI-driven desktop application: a PyQt5 annotation UI drives a
background engine that wraps SAM2 (mask propagation) and CoTracker3 (keypoint tracking),
with a separate downstream toolchain (`DatasetManager/`) turning verified annotations into
YOLO-format training data.

For the full data/control-flow diagram, the real call path from `run_main.py` down to model
inference, and a breakdown of every subpackage, see the
**[System Architecture & Workflow Guide](./docs/architecture_and_workflow.md)** — that
document is the single source of truth, kept in sync with the code (this README doesn't
duplicate it).

### Directory Map

```text
AutoSegmentor/
├── run_main.py                # Main entry point
├── install.py                 # One-shot setup (deps, submodules, checkpoints, GPU check)
├── autosegmentor/              # Core application package
│   ├── core/                  # Pipeline orchestration (AutoSegmentorEngine)
│   ├── ui/                    # PyQt5 windows & widgets
│   ├── models/                # SAM2 & CoTracker wrappers
│   ├── file_management/       # Disk ETL & data handling
│   └── tools/                 # App bootstrap, demo registry
├── DatasetManager/             # Dataset export & synthesis (see READMEs below)
│   ├── SyntheticEngine/        # Offline augmentation pipeline
│   └── YolovDatasetManager/    # YOLO format creation
├── workspace/                  # Project workspace (videos in, datasets/logs out)
├── external/                   # Vendored SAM2 + CoTracker3 (see docs for details)
├── assets/                     # Media assets for README
├── demo/                       # Bundled demo footage + session configs
├── outputs/                    # Logs
└── docs/                       # Detailed technical documentation
```

---

## 📖 Detailed Documentation Index

For in-depth guides on every part of the AutoSegmentor ecosystem, refer to the following documents:

### 🏛️ Core Architecture
- **[System Architecture & Workflow Guide](./docs/architecture_and_workflow.md)**: Deep dive into the PyQt5 design, async threading, and technical pipeline.

### 📊 Dataset Management
- **[Dataset Manager Overview](./DatasetManager/README.md)**: Entry point for post-processing tools.
- **[YOLO Dataset Creator](./DatasetManager/YolovDatasetManager/README.md)**: Guide for converting verified masks into YOLOv8/v11 training data.
- **[Synthetic Data Engine](./DatasetManager/SyntheticEngine/README.md)**: Instructions for creating large-scale synthetic datasets using copy-paste augmentation.

---

## ❓ Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **VRAM Out of Memory** | Reduce `batch_size` in the config (e.g., to 8 or 16). |
| **SAM2 Missing** | Ensure `external/segment_anything_2/` exists (it is vendored in the repo, not a submodule) and `external_libs` in `workspace/inputs/config/default_config.yaml` lists it. |
| **Slow Preview** | Check if `torch.cuda.is_available()` is True. CPU inference is extremely slow. |
| **GUI Not Opening** | Verify your PyQt5 installation and display drivers. |

---

## Acknowledgements

- [Meta AI's SAM2](https://github.com/facebookresearch/segment-anything-2)
- [CoTracker Team](https://github.com/facebookresearch/co-tracker)
- All open-source contributors to the PyTorch and PyQt ecosystems.

---

**Built with ❤️ for the Computer Vision community.**
