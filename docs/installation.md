# Installation

Tested on **Windows 11** and **Ubuntu 22.04 / 24.04**.

## Prerequisites

| | Requirement |
| :--- | :--- |
| Python | 3.10+ |
| GPU | NVIDIA GPU with CUDA 12.x (SAM2/CoTracker are far too slow on CPU alone) |
| RAM | 16GB+ recommended |
| Disk | ~20GB (model checkpoints + working directories) |
| Git | [Git LFS](https://git-lfs.com/) — only needed if you plan to contribute new demo footage |

## 1. Clone the repository

Cloning with `--recursive` fetches the vendored CoTracker3 submodule. (SAM2 is vendored
directly in the repo, not as a submodule.)

```bash
git clone --recursive https://github.com/thippeswammy/AutoSegmentor.git
cd AutoSegmentor
```

Forgot `--recursive`? Fetch the submodule after the fact:

```bash
git submodule update --init --recursive
```

## 2. Create a virtual environment

=== "Windows (PowerShell)"

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

=== "Ubuntu / Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

## 3. Run the one-shot setup

```bash
python install.py
```

This single, cross-platform script — no separate `.bat`/`.sh` files to keep in sync —
takes care of everything else:

1. Checks your Python version.
2. Initializes git submodules.
3. Installs dependencies from `requirements-core.txt`.
4. Downloads the SAM2 and CoTracker3 model checkpoints.
5. Runs a GPU/CUDA diagnostic so you know before you launch whether inference will be fast
   or painfully slow.

Add `--cuda` to also install the optional `flash-attn` speedup — it needs a matching CUDA
build toolchain, which is more commonly already set up on Linux than on Windows, so treat it
as optional either way.

Only need part of it? Every step also runs standalone:

```bash
python install.py --checkpoints-only            # download missing checkpoints
python install.py --checkpoints-only --check    # just verify they're present
python install.py --gpu-check-only              # just the GPU/system diagnostic
```

## 4. Verify

```bash
python run_main.py --version     # AutoSegmentor 3.0.0
python run_main.py --demo list   # cat, road
```

If that prints a version and two demo names, you're ready — head to [Demos](demos.md).

## Doing it by hand

Everything `install.py` automates, spelled out, in case you want full control:

```bash
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

Neither SAM2 nor CoTracker3 is `pip install`ed — both are vendored under `external/`, and
`run_main.py` adds them to `sys.path` at launch. You just need the folders present (the
clone step above handles that).

Download the two checkpoints manually if you'd rather not run the script:

| Model | File | Source |
| :--- | :--- | :--- |
| SAM2 | `sam2_hiera_large.pt` | [facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2) |
| CoTracker3 | `scaled_offline.pth` | [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker) |

Place them at:

```text
external/segment_anything_2/checkpoints/sam2_hiera_large.pt
external/co-tracker/checkpoints/scaled_offline.pth
```

Optionally install AutoSegmentor as a package for the `autosegmentor` / `autosegmentor-demo`
console commands:

```bash
pip install -e .
```

## Troubleshooting

**Linux: `could not load the Qt platform plugin "xcb"`** — a minimal Ubuntu install is
missing PyQt5's system dependencies:

```bash
sudo apt-get install libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0
```

**VRAM out of memory** — lower `batch_size` in your config (try 8 or 16).

**Everything runs, but painfully slowly** — check `python install.py --gpu-check-only`
reports `CUDA available: True`. CPU-only inference works but is not a realistic way to run
this tool on real footage.
